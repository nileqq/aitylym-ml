"""KazEGA-HuBERT: эмоция по отрезкам речи.

Почему на фразу, а не на слово
------------------------------
Модель обучалась на клипах 1-20 секунд, и на отрезке в одно слово
(300-500 мс) она неработоспособна. Причина измерена: смена текста двигает
эмбеддинг HuBERT в 2.29 раза сильнее, чем смена эмоции. На коротком куске
усреднять нечего, и «эмоция» слова определялась бы тем, из каких звуков оно
состоит, а не тем, как его произнесли — цвет менялся бы стабильно и
убедительно, и при этом неверно.

Поэтому эмоция считается на фразе, а пословную динамику даёт просодия.

Точность
--------
На своём тесте KazEGA (тот же домен, что и у нас — казахский YouTube):
accuracy 52.6%, macro F1 45.9%, каппа 0.391 при семи классах.
На актёрской студийной начитке KazEmoTTS — 24.5% и каппа 0.082, то есть
почти неотличимо от случайного. Разница доменная: там просодия сжата.
"""

from __future__ import annotations

import json

import numpy as np

SAMPLE_RATE = 16_000
MODEL_REPO = "kazega0/KazEGA-HuBERT"
BASE_MODEL = "facebook/hubert-base-ls960"

# Так модель обучалась: ровно 10 секунд, короткое дополняется нулями.
MAX_LENGTH = 160_000

# Целевая длина фразы. Нижняя граница — потому что ниже модель ломается,
# верхняя — чтобы в одну фразу не попало несколько разных эмоций.
PHRASE_MIN_SEC = 3.0
PHRASE_MAX_SEC = 8.0
PHRASE_GAP_SEC = 0.45

# Доли классов в обучающей выборке KazEGA (train, 88 977 записей).
# Модель видела neutral в 37% случаев и тянет туда на любых данных.
# Деление на эти доли выравнивает шансы: на KazEmoTTS приём поднимал
# macro F1 с 17.1% до 27.0%.
TRAIN_PRIOR = {
    "neutral": 0.3728,
    "sad": 0.1227,
    "happy": 0.1184,
    "angry": 0.1156,
    "fearful": 0.1133,
    "disgusted": 0.0914,
    "surprised": 0.0658,
}

# Координаты в пространстве valence-arousal. Значения литературные —
# уточнить их должен опрос.
VALENCE_AROUSAL = {
    "neutral": (0.0, 0.0),
    "happy": (0.8, 0.6),
    "angry": (-0.7, 0.8),
    "fearful": (-0.7, 0.7),
    "sad": (-0.7, -0.5),
    "disgusted": (-0.6, 0.2),
    "surprised": (0.2, 0.8),
}

_model = None
_processor = None
_id2emotion = None
_device = None


# ----------------------------------------------------------------------------
# Фразы
# ----------------------------------------------------------------------------


def group_words(spans: list) -> list:
    """Слова -> фразы. Возвращает список списков индексов слов.

    Режем по паузам, но следим за длиной: слишком короткая фраза для модели
    бесполезна, слишком длинная смешивает разные эмоции.
    """
    if not spans:
        return []

    groups, current = [], [0]

    for i in range(1, len(spans)):
        gap = spans[i][0] - spans[i - 1][1]
        span = spans[i][1] - spans[current[0]][0]

        if (gap >= PHRASE_GAP_SEC or span > PHRASE_MAX_SEC) and \
                spans[i - 1][1] - spans[current[0]][0] >= PHRASE_MIN_SEC:
            groups.append(current)
            current = [i]
        else:
            current.append(i)

    groups.append(current)

    # Короткий хвост приклеиваем к предыдущей фразе, а не оставляем огрызком.
    if len(groups) > 1:
        last = groups[-1]
        if spans[last[-1]][1] - spans[last[0]][0] < PHRASE_MIN_SEC:
            groups[-2].extend(groups.pop())

    return groups


# ----------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------


def _build_class():
    """Архитектура KazEGA-HuBERT.

    Повторяет kazega/model.py из репозитория модели. Имена полей должны
    совпадать с ключами чекпоинта, иначе strict=True не пройдёт.
    """
    import torch
    import torch.nn as nn
    from transformers import HubertModel

    class MultiTaskHubert(nn.Module):
        def __init__(self, num_emotions, num_genders, num_ages):
            super().__init__()
            self.backbone = HubertModel.from_pretrained(
                BASE_MODEL, output_hidden_states=True
            )
            hidden = self.backbone.config.hidden_size
            layers = self.backbone.config.num_hidden_layers + 1

            self.emotion_weights = nn.Parameter(torch.ones(layers))
            self.gender_weights = nn.Parameter(torch.ones(layers))
            self.age_weights = nn.Parameter(torch.ones(layers))

            def head(out_dim, dropout):
                return nn.Sequential(
                    nn.Linear(hidden, 256), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(256, out_dim),
                )

            self.emotion_head = head(num_emotions, 0.2)
            self.gender_head = head(num_genders, 0.1)
            self.age_head = head(num_ages, 0.1)

        def forward(self, input_values, input_lengths):
            outputs = self.backbone(input_values, output_hidden_states=True)
            hidden = torch.stack(outputs.hidden_states, dim=0)

            feat_lengths = self.backbone._get_feat_extract_output_lengths(
                input_lengths
            ).to(input_values.device)

            frames = hidden.shape[2]
            index = torch.arange(frames, device=input_values.device)
            # Маска по реальной длине: паддинг не должен попасть в среднее.
            mask = (index.unsqueeze(0) < feat_lengths.unsqueeze(1)).float()

            def pool(weights):
                w = torch.softmax(weights, dim=0)
                merged = (w.view(-1, 1, 1, 1) * hidden).sum(dim=0)
                m = mask.unsqueeze(-1)
                return (merged * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

            return (
                self.emotion_head(pool(self.emotion_weights)),
                self.gender_head(pool(self.gender_weights)),
                self.age_head(pool(self.age_weights)),
            )

    return MultiTaskHubert


def load():
    """Грузит модель. Вызывать при старте: около 4 секунд."""
    global _model, _processor, _id2emotion, _device

    if _model is not None:
        return _model

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        hf_hub_download(MODEL_REPO, "model.pt"),
        map_location="cpu", weights_only=False,
    )
    encoders = json.load(
        open(hf_hub_download(MODEL_REPO, "label_encoders.json"), encoding="utf-8")
    )
    _id2emotion = {v: k for k, v in encoders["emotion"].items()}

    cls = _build_class()
    _model = cls(checkpoint["num_emotions"], checkpoint["num_genders"],
                 checkpoint["num_ages"])
    _model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _model.to(_device).eval()

    _processor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL)
    return _model


def debias(probs: dict) -> dict:
    """Снимает перекос модели в сторону частых классов обучения."""
    adjusted = {e: p / TRAIN_PRIOR.get(e, 1.0) for e, p in probs.items()}
    total = sum(adjusted.values()) or 1.0
    return {e: p / total for e, p in adjusted.items()}


def predict(y: np.ndarray, sr: int, spans: list) -> list:
    """Отрезки -> список словарей {эмоция: вероятность}, уже без смещения."""
    import torch

    if not spans:
        return []

    load()

    batch, lengths = [], []
    for start, end in spans:
        segment = y[int(start * sr): int(end * sr)][:MAX_LENGTH]
        length = len(segment)
        if length < MAX_LENGTH:
            segment = np.pad(segment, (0, MAX_LENGTH - length))
        batch.append(segment)
        lengths.append(max(length, 1))

    inputs = _processor(
        batch, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_values.to(_device)

    with torch.inference_mode():
        logits, _, _ = _model(
            inputs, torch.tensor(lengths, dtype=torch.long, device=_device)
        )
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    return [
        debias({_id2emotion[i]: float(row[i]) for i in range(len(row))})
        for row in probs
    ]


def valence_arousal(probs: dict) -> tuple:
    """Взвешенная точка в VA-пространстве — по всем классам, не по argmax."""
    v = sum(VALENCE_AROUSAL[e][0] * p for e, p in probs.items())
    a = sum(VALENCE_AROUSAL[e][1] * p for e, p in probs.items())
    return (v, a)
