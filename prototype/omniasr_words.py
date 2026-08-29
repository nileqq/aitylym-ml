"""Пословные тайминги из OmniASR CTC.

Как это работает, в терминах классического МО
---------------------------------------------
Никакой магии тут нет. Модель — это **поканальный классификатор кадров**.
Она режет сигнал на кадры по 20.05 мс и для каждого кадра НЕЗАВИСИМО выдаёт
распределение вероятностей по 10288 токенам (символы + несколько служебных).
То есть на выходе просто матрица T x V: T кадров, V классов. Ровно как если бы
ты обучил логрегрессию предсказывать букву по окну сигнала — только признаки
считает свёрточно-трансформерный энкодер, а не ты руками.

Два служебных токена важны:
    id 0  — "blank": в этом кадре новой буквы нет
    id 4  — " "    : в этом кадре разделитель слов

Стандартный жадный CTC-декод — три шага:
    1. argmax по каждому кадру          -> один id на кадр
    2. схлопнуть подряд идущие одинаковые -> убирает дубли от буквы,
                                            растянутой на несколько кадров
    3. выбросить blank                   -> получился текст

Шаги 1-2 сохраняют НОМЕР КАДРА, и вот из-за этого тайминги достаются бесплатно:
кадр i — это i * 20.05 мс аудио. Разрезав уцелевшие токены по пробелу,
получаем границы слов в кадрах, а значит и в секундах.

Одна тонкость, которую надо знать
---------------------------------
CTC не "тянет" букву всю её длительность — он выдаёт ПИК где-то внутри неё,
обычно ближе к концу. Поэтому первая буква слова срабатывает чуть позже, чем
слово реально началось. Если брать границы по буквам, все слова уедут вправо
на 40-80 мс.

Поэтому границы слова берутся по РАЗДЕЛИТЕЛЯМ: слово занимает кадры между
предыдущим и следующим пиком пробела. Так точнее, и слова гарантированно не
накладываются друг на друга.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16_000

# Служебные id в словаре omniASR_tokenizer_written_v2
BLANK_ID = 0
SPACE_ID = 4

# 0/1/2 — blank/bos/eos, декодируются в пустую строку.
# 3 — unk, и он декодируется как " ⁇ ", то есть с пробелами ВНУТРИ. Если его
# не выбросить, он разорвёт слово и попадёт в текст. Проверено на словаре
# omniASR_tokenizer_written_v2: id 4 — единственный, дающий чистый пробел.
_SPECIAL_IDS = (0, 1, 2, 3)

# CTC не растягивает букву на всю её длительность, а выдаёт пик где-то внутри,
# обычно ближе к концу. Поэтому начало первого слова и конец последнего
# сдвигаем на эту величину, иначе они прилипают к границам куска.
_PEAK_LAG_FRAMES = 3  # ~60 мс

# Длинное аудио режем: у трансформера внимание квадратично по длине,
# на нескольких минутах разом кончится память.
MAX_CHUNK_SEC = 20.0
MIN_CHUNK_SEC = 1.0


@dataclass
class Word:
    """Одно слово с границами в секундах."""

    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 3)
        return d


# ----------------------------------------------------------------------------
# Загрузка модели
# ----------------------------------------------------------------------------

_PIPELINE_CACHE: dict = {}


def load_pipeline(model_card: str = "omniASR_CTC_300M_v2"):
    """Грузит пайплайн один раз и кеширует.

    Загрузка занимает несколько секунд, а в Gradio коллбек вызывается на каждый
    клик — без кеша прототип будет невыносимо медленным.
    """
    if model_card not in _PIPELINE_CACHE:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        _PIPELINE_CACHE[model_card] = ASRInferencePipeline(model_card=model_card)
    return _PIPELINE_CACHE[model_card]


# ----------------------------------------------------------------------------
# Подготовка аудио
# ----------------------------------------------------------------------------


def prepare_audio(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Моно + 16 кГц + float32. Модель обучена только на таком входе."""
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
        )

    # Некоторые wav приходят как int16, отмасштабированный в float
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak

    return waveform.astype(np.float32)


def _split_into_chunks(waveform: np.ndarray) -> List[tuple]:
    """Режет длинное аудио на куски <= MAX_CHUNK_SEC по паузам.

    Возвращает список (offset_samples, chunk). Режем именно по тишине, чтобы
    граница куска не попала в середину слова.
    """
    max_len = int(MAX_CHUNK_SEC * SAMPLE_RATE)

    if len(waveform) <= max_len:
        return [(0, waveform)]

    import librosa

    # Границы речи; top_db=35 — умеренно агрессивно, паузы между фразами ловит
    intervals = librosa.effects.split(waveform, top_db=35)
    if len(intervals) == 0:
        intervals = np.array([[0, len(waveform)]])

    chunks = []
    start = 0
    last_end = None  # конец последней паузы, влезающей в лимит

    for _, seg_end in intervals:
        if seg_end - start > max_len:
            if last_end is not None and last_end > start:
                # режем по последней паузе, которая ещё влезала
                chunks.append((start, waveform[start:last_end]))
                start = last_end
            else:
                # одна непрерывная реплика длиннее лимита — паузы нет,
                # приходится резать посередине слова
                hard_end = start + max_len
                chunks.append((start, waveform[start:hard_end]))
                start = hard_end
        last_end = seg_end

    if start < len(waveform):
        chunks.append((start, waveform[start:]))

    # Слишком короткие хвосты приклеиваем к предыдущему куску
    merged: List[tuple] = []
    for offset, chunk in chunks:
        if merged and len(chunk) < MIN_CHUNK_SEC * SAMPLE_RATE:
            prev_offset, prev_chunk = merged[-1]
            merged[-1] = (prev_offset, waveform[prev_offset : offset + len(chunk)])
        else:
            merged.append((offset, chunk))

    return merged


# ----------------------------------------------------------------------------
# Главное: логиты -> слова с таймингами
# ----------------------------------------------------------------------------


def _frame_tokens(pipeline, chunk: np.ndarray):
    """Прогоняет кусок через модель и возвращает (кадр, id) для непустых пиков.

    Здесь и происходят шаги 1-2 жадного CTC-декода, но с сохранением
    номера кадра.
    """
    import torch
    from fairseq2.nn import BatchLayout

    model = pipeline.model
    dtype = next(model.parameters()).dtype

    x = torch.from_numpy(chunk).unsqueeze(0).to(device=pipeline.device, dtype=dtype)
    layout = BatchLayout(x.shape, seq_lens=[x.shape[1]], device=pipeline.device)

    with torch.inference_mode():
        logits, out_layout = model(x, layout)

    num_frames = int(out_layout.seq_lens[0])
    ids = logits[0, :num_frames].argmax(dim=-1)

    # шаг 2: схлопнуть повторы
    keep = torch.ones(num_frames, dtype=torch.bool, device=ids.device)
    keep[1:] = ids[1:] != ids[:-1]

    frames = torch.nonzero(keep).squeeze(1).tolist()
    kept = [(f, int(ids[f])) for f in frames if int(ids[f]) not in _SPECIAL_IDS]

    # сколько секунд приходится на кадр — считаем из фактических длин,
    # а не хардкодим 0.02005
    sec_per_frame = (len(chunk) / SAMPLE_RATE) / num_frames

    return kept, sec_per_frame


def _tokens_to_words(pipeline, kept, sec_per_frame, offset_sec, total_sec) -> List[Word]:
    """Разрезает поток (кадр, id) по пробелам и собирает слова."""
    import torch

    words: List[Word] = []
    buf: List[tuple] = []  # (кадр, id)

    # None означает "разделителя перед этим словом ещё не было", то есть слово
    # первое в куске. Тогда опереться не на что и берём сам токен.
    prev_sep_frame: Optional[int] = None

    def to_sec(frame: int) -> float:
        # float() обязателен: offset_sec приходит из numpy-индексов и тянет за
        # собой np.float64, из-за чего у части слов типы разъезжаются
        return float(frame * sec_per_frame + offset_sec)

    def flush(sep_frame: Optional[int]):
        nonlocal buf, prev_sep_frame
        if buf:
            text = pipeline.token_decoder(
                torch.tensor([token_id for _, token_id in buf])
            ).strip()
            if text:
                # границы по разделителям, а не по первой/последней букве
                if prev_sep_frame is None:
                    start = to_sec(max(buf[0][0] - _PEAK_LAG_FRAMES, 0))
                else:
                    start = to_sec(prev_sep_frame)

                if sep_frame is None:
                    end = min(to_sec(buf[-1][0] + _PEAK_LAG_FRAMES), total_sec)
                else:
                    end = to_sec(sep_frame)

                if end > start:
                    words.append(
                        Word(word=text, start=round(start, 3), end=round(end, 3))
                    )
        buf = []
        if sep_frame is not None:
            prev_sep_frame = sep_frame

    for frame, token_id in kept:
        if token_id == SPACE_ID:
            flush(frame)
        else:
            buf.append((frame, token_id))

    flush(None)
    return words


def transcribe_words(
    waveform: np.ndarray,
    sample_rate: int,
    pipeline=None,
) -> List[Word]:
    """Аудио -> список слов с границами в секундах.

    Это единственная функция, которая нужна снаружи. Пример:

        import soundfile as sf
        from omniasr_words import transcribe_words

        wav, sr = sf.read("tale.wav")
        for w in transcribe_words(wav, sr):
            print(w.word, w.start, w.end)
    """
    if pipeline is None:
        pipeline = load_pipeline()

    waveform = prepare_audio(waveform, sample_rate)

    words: List[Word] = []
    for offset_samples, chunk in _split_into_chunks(waveform):
        if len(chunk) < 0.05 * SAMPLE_RATE:
            continue

        offset_sec = offset_samples / SAMPLE_RATE
        chunk_sec = len(chunk) / SAMPLE_RATE

        kept, sec_per_frame = _frame_tokens(pipeline, chunk)
        words.extend(
            _tokens_to_words(
                pipeline, kept, sec_per_frame, offset_sec, offset_sec + chunk_sec
            )
        )

    return words


def plain_text(words: List[Word]) -> str:
    return " ".join(w.word for w in words)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    import soundfile as sf

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--json", help="куда сохранить результат")
    args = parser.parse_args()

    wav, sr = sf.read(args.audio, dtype="float32")
    result = transcribe_words(wav, sr)

    print(f"\n{plain_text(result)}\n")
    print(f"{'слово':<22s} {'начало':>8s} {'конец':>8s} {'длит':>7s}")
    print("-" * 50)
    for w in result:
        print(f"{w.word:<22s} {w.start:8.3f} {w.end:8.3f} {w.duration:7.3f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([w.to_dict() for w in result], f, ensure_ascii=False, indent=2)
        print(f"\nсохранено: {args.json}")
