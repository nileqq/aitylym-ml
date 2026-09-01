"""AItylym — каркас приложения.

Принимает аудиофайл, отдаёт результат обработки. Сама обработка пока не
реализована: process() возвращает None, интерфейс это переживает и
показывает заглушку.

Запуск:
    gradio app/main.py     # с горячей перезагрузкой
    python app/main.py
"""

import gradio as gr
import vad, asr, prosody, librosa
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

    # TODO: эмоция
    # даем это HUBert
    #
    # Эмоцию считать на отрезок VAD целиком, не на слово: на 300-500 мс
    # модель неработоспособна, там побеждает фонетика, а не подача.
    # Отрезки для этого уже есть — segments.

    # Просодия пословно
    values = prosody.analyze(y, sr, spans)

    # TODO: покрасить по эмоции

    return [
        {
            "text": label,
            "start": start,
            "end": end,
            "loud": p.loud,
            "pitch": p.pitch,
            "voiced": p.voiced,
        }
        for label, (start, end), p in zip(labels, spans, values)
    ]


def render_card(result):
    """Отрезки -> карточка, как будет выглядеть субтитр.

    Подпись — само слово; если текста нет, показывается интервал времени.

    Маппинг сейчас временный:
        размер <- громкость   (останется: у CuCap это самое устойчивое
                               соответствие, 29% NA / 38% KOR)
        цвет   <- высота      (заглушка: в итоге цвет несёт эмоцию, а
                               высоту CuCap кодировать не советует —
                               79% участников не сопоставили ей ничего)
    """
    if not result:
        return "<div style='padding:32px;text-align:center;opacity:.5'>—</div>"

    blocks = []
    for item in result:
        size = 18 + 26 * item["loud"]

        if item["voiced"]:
            # Тон от синего (низкий голос) к оранжевому (высокий).
            # Крутим оттенок в HSL, а не смешиваем два цвета в RGB:
            # у смеси середина всегда уходит в грязь.
            hue = 210 - 190 * item["pitch"]
            color = f"hsl({hue:.0f}, 70%, 62%)"
        else:
            color = "#9AA3AC"

        weight = 400 + 400 * item["loud"]
        label = item.get("text") or f'{item["start"]:.1f}–{item["end"]:.1f}'

        blocks.append(
            f'<span style="color:{color};font-size:{size:.1f}px;'
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
        "| сөз | уақыт | қаттылық | биіктік |",
        "|---|---|---|---|",
    ]
    for item in result:
        pitch = f"{item['pitch']:.2f}" if item["voiced"] else "—"
        lines.append(
            f"| {item.get('text', '')} "
            f"| {item['start']:.2f}–{item['end']:.2f} с "
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
