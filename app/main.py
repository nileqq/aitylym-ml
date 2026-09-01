"""AItylym — каркас приложения.

Принимает аудиофайл, отдаёт результат обработки. Сама обработка пока не
реализована: process() возвращает None, интерфейс это переживает и
показывает заглушку.

Запуск:
    gradio app/main.py     # с горячей перезагрузкой
    python app/main.py
"""

import gradio as gr
import vad, asr, emotion, prosody, librosa
from html import escape
from pathlib import Path
import numpy as np

DEFAULT_SR = 16000

def process(audio_path):
    """
    Аудио -> результат. Здесь будет ASR, эмоции и просодия.

    Args:
        audio_path: путь к файлу, либо None если ничего не загружено.

    Returns:
        Список отрезков с просодией. Пока без текста и эмоции.
    """
    if audio_path is None:
        return None

    # Грузим один раз и передаём дальше сэмплы, а не путь: get_speech_timestamps
    # и librosa оба работают с массивом, повторное чтение файла ни к чему.
    y, sr = librosa.load(audio_path, sr=DEFAULT_SR)

    # VAD / нарезка
    tmstamps = vad.tmstamps(y)
    segments = [(s["start"], s["end"]) for s in tmstamps]

    if not segments:
        return None

    # ASR -> слова с таймингами. Границы слов точнее отрезков VAD, поэтому
    # просодия дальше считается именно по ним.
    found = asr.words(y, sr)

    if not found:
        return None

    labels = [word for word, _, _ in found]
    spans = [(start, end) for _, start, end in found]

    # Эмоция — на фразе, не на слове. На 300-500 мс модель неработоспособна:
    # там побеждает фонетика, а не подача (замерено — смена текста двигает
    # эмбеддинг в 2.29 раза сильнее, чем смена эмоции).
    groups = emotion.group_words(spans)
    phrase_spans = [(spans[g[0]][0], spans[g[-1]][1]) for g in groups]
    phrase_probs = emotion.predict(y, sr, phrase_spans)

    # Каждому слову — эмоция его фразы
    per_word = [None] * len(spans)
    for group, probs in zip(groups, phrase_probs):
        top = max(probs, key=probs.get)
        for i in group:
            per_word[i] = (top, probs[top])

    # Просодия пословно
    values = prosody.analyze(y, sr, spans)

    return [
        {
            "text": label,
            "start": start,
            "end": end,
            "loud": p.loud,
            "pitch": p.pitch,
            "voiced": p.voiced,
            "emotion": per_word[i][0] if per_word[i] else "neutral",
            "confidence": per_word[i][1] if per_word[i] else 0.0,
        }
        for i, (label, (start, end), p) in enumerate(zip(labels, spans, values))
    ]


# Оттенок под каждую эмоцию. Какой цвет какой эмоции соответствует у
# казахоязычных зрителей — как раз то, что должен показать опрос; пока это
# гипотеза. neutral без тона: серый.
EMOTION_HUE = {
    "neutral": None,
    "happy": 48,
    "angry": 5,
    "fearful": 275,
    "sad": 210,
    "disgusted": 110,
    "surprised": 25,
}

EMOTION_LABEL = {
    "neutral": "бейтарап", "happy": "қуаныш", "angry": "ашу",
    "fearful": "қорқыныш", "sad": "қайғы", "disgusted": "жиіркеніш",
    "surprised": "таңданыс",
}


def render_card(result):
    """Слова -> карточка, как будет выглядеть субтитр.

    Маппинг:
        тон         <- эмоция ФРАЗЫ
        насыщенность<- уверенность модели
        светлота    <- громкость СЛОВА
        размер      <- громкость СЛОВА  (у CuCap самое устойчивое
                       соответствие: 29% NA / 38% KOR)

    Тон крутим в HSL, а не смешиваем цвета в RGB: у смеси середина уходит
    в грязь. Высоту голоса отдельно не кодируем — 79% участников CuCap не
    сопоставили ей ничего.
    """
    if not result:
        return "<div style='padding:32px;text-align:center;opacity:.5'>—</div>"

    blocks = []
    for item in result:
        size = 18 + 26 * item["loud"]

        hue = EMOTION_HUE.get(item.get("emotion", "neutral"))
        if hue is None:
            # Нейтраль остаётся серой: приписывать ей цвет значило бы
            # выдавать «модель ничего не нашла» за содержательный ответ.
            light = 55 + 20 * item["loud"]
            color = f"hsl(0, 0%, {light:.0f}%)"
        else:
            sat = 35 + 45 * min(1.0, item.get("confidence", 0.0) * 2)
            light = 48 + 22 * item["loud"]
            color = f"hsl({hue}, {sat:.0f}%, {light:.0f}%)"

        weight = 400 + 400 * item["loud"]
        label = item.get("text") or f'{item["start"]:.1f}–{item["end"]:.1f}'

        title = EMOTION_LABEL.get(item.get("emotion", ""), "")
        if item.get("confidence"):
            title += f' {item["confidence"]:.0%}'

        blocks.append(
            f'<span title="{escape(title)}" style="color:{color};font-size:{size:.1f}px;'
            f'font-weight:{int(weight // 100 * 100)};'
            f'margin:0 .3em;display:inline-block;'
            f'text-shadow:0 1px 3px rgba(0,0,0,.8)">{escape(label)}</span>'
        )

    return (
        '<div style="background:#111418;border-radius:14px;'
        'padding:30px 24px;text-align:center;line-height:1.7;'
        'font-family:system-ui,sans-serif">'
        + "".join(blocks) +
        '</div>'
    )


def on_click(audio_path):
    """
    Прослойка между process() и интерфейсом.

    Нужна, чтобы None не ломал вывод: пока process() ничего не возвращает,
    показываем заглушку вместо пустоты.
    """
    result = process(audio_path)

    if result is None:
        return "_Нәтиже жоқ_", render_card(None)

    lines = [
        "| сөз | уақыт | эмоция | қаттылық | биіктік |",
        "|---|---|---|---|---|",
    ]
    for item in result:
        pitch = f"{item['pitch']:.2f}" if item["voiced"] else "—"
        label = EMOTION_LABEL.get(item.get("emotion", ""), "")
        conf = f" {item['confidence']:.0%}" if item.get("confidence") else ""
        lines.append(
            f"| {item.get('text', '')} "
            f"| {item['start']:.2f}–{item['end']:.2f} с "
            f"| {label}{conf} "
            f"| {item['loud']:.2f} | {pitch} |"
        )

    return "\n".join(lines), render_card(result)


with gr.Blocks(title="AItylym") as demo:
    gr.Markdown("# AItylym\n### Қазақ тіліндегі динамикалық субтитрлер")

    audio = gr.Audio(label="Аудио", type="filepath")
    run = gr.Button("Талдау", variant="primary")
    output = gr.Markdown()

    gr.Markdown("### Вывод")
    card = gr.HTML(value=render_card(None))

    run.click(on_click, audio, [output, card])


if __name__ == "__main__":
    demo.launch()
