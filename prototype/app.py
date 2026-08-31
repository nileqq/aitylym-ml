"""AItylym — прототип кинетических субтитров.

Что это
-------
Текст + параметры подачи для каждого слова -> раскрашенный субтитр.

Аудио здесь ещё нет: просодия задаётся вручную в таблице. Это сделано
намеренно. Так прототип проверяет ровно одну вещь — **алгоритм оформления**,
отдельно от качества ASR и распознавания эмоций. Когда
`omniasr_words.py` начнёт отдавать слова с таймингами, а DSP — громкость и
высоту, они просто заполнят ту же таблицу, и рендер не изменится.

Маппинг взят из CuCap (ASSETS'25), а не придуман:
    цвет   <- эмоция (валентность)
    размер <- громкость   — самое устойчивое соответствие, 29% NA / 38% KOR
    жирность <- arousal
Высота голоса отдельно не кодируется: у CuCap 79% участников из Северной
Америки выбрали для неё «никак».

Запуск:
    python prototype/app.py            # локально
    python prototype/app.py --share    # публичная ссылка gradio.live
"""

from __future__ import annotations

import argparse
import html
import math
import re

import gradio as gr
import pandas as pd

# ----------------------------------------------------------------------------
# Эмоции и их координаты в пространстве valence-arousal
# ----------------------------------------------------------------------------

# Валентность решает цвет, arousal — жирность. Значения примерные: их и должен
# уточнить опрос, ради него всё и затевалось.
EMOTIONS = {
    "бейтарап": (0.0, 0.0),
    "қуаныш": (0.8, 0.6),
    "ашу": (-0.7, 0.8),
    "қорқыныш": (-0.7, 0.7),
    "қайғы": (-0.7, -0.5),
    "жиркену": (-0.6, 0.2),
    "таңданыс": (0.2, 0.8),
}

DEFAULT_COLORS = {
    "бейтарап": "#D9D9D9",
    "қуаныш": "#F5C518",
    "ашу": "#E23B3B",
    "қорқыныш": "#8B5CF6",
    "қайғы": "#4A90E2",
    "жиркену": "#4CAF50",
    "таңданыс": "#FF8A3D",
}

COLUMNS = ["сөз", "эмоция", "қаттылық", "екпін"]

SAMPLE_TEXT = "Сен бүгін келдің деп ойламадым мен"


# ----------------------------------------------------------------------------
# Oklab: смешивание цветов без грязи
# ----------------------------------------------------------------------------
#
# Интерполяция в RGB тащит переход между насыщенными цветами через мутную
# середину — красный с синим дают грязно-серый. Oklab перцептивно равномерен,
# и та же середина выглядит чистой. Это буквально три матрицы, но разница на
# глаз большая.


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055


def hex_to_oklab(value: str) -> tuple:
    value = (value or "#000000").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r, g, b = (int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        r = g = b = 0.0

    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))

    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def oklab_to_hex(lab: tuple) -> str:
    big_l, a, b = lab

    l_ = big_l + 0.3963377774 * a + 0.2158037573 * b
    m_ = big_l - 0.1055613458 * a - 0.0638541728 * b
    s_ = big_l - 0.0894841775 * a - 1.2914855480 * b

    l, m, s = l_**3, m_**3, s_**3

    rgb = (
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    )

    out = []
    for channel in rgb:
        channel = _linear_to_srgb(max(0.0, min(1.0, channel)))
        out.append(f"{int(round(max(0.0, min(1.0, channel)) * 255)):02X}")

    return "#" + "".join(out)


def mix_oklab(first: tuple, second: tuple, amount: float) -> tuple:
    return tuple(a + (b - a) * amount for a, b in zip(first, second))


# ----------------------------------------------------------------------------
# Текст -> таблица
# ----------------------------------------------------------------------------


def text_to_table(text: str) -> pd.DataFrame:
    """Разбивает текст на слова и заводит для каждого строку со значениями
    по умолчанию. Дальше человек правит их руками — или, в будущем, их
    заполнит DSP."""
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    if not words:
        words = ["—"]

    return pd.DataFrame(
        [[w, "бейтарап", 0.5, 0.5] for w in words],
        columns=COLUMNS,
    )


def _clamp01(value, fallback=0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


# ----------------------------------------------------------------------------
# Рендер
# ----------------------------------------------------------------------------


def render(
    table,
    smoothing,
    size_min,
    size_max,
    weight_min,
    weight_max,
    dark,
    *colors,
):
    palette = dict(zip(EMOTIONS.keys(), colors))

    if isinstance(table, pd.DataFrame):
        rows = table.values.tolist()
    else:
        rows = list(table or [])

    words = []
    for row in rows:
        row = list(row) + [None] * (4 - len(row))
        word = str(row[0] or "").strip()
        if not word or word == "—":
            continue

        emotion = str(row[1] or "бейтарап").strip().lower()
        if emotion not in EMOTIONS:
            emotion = "бейтарап"

        words.append(
            {
                "word": word,
                "emotion": emotion,
                "loud": _clamp01(row[2]),
                "stress": _clamp01(row[3]),
                "lab": hex_to_oklab(palette.get(emotion, "#D9D9D9")),
            }
        )

    if not words:
        return "<p style='opacity:.6'>Мәтін енгізіңіз</p>"

    # Сглаживание: подмешиваем к слову цвета соседей. Требование из
    # algorithm.docx — «цвета соседние смешиваются, чтобы не было резкого
    # перехода». Делается в Oklab, иначе середина уходит в грязь.
    smoothed = []
    for i, item in enumerate(words):
        neighbours = []
        if i > 0:
            neighbours.append(words[i - 1]["lab"])
        if i < len(words) - 1:
            neighbours.append(words[i + 1]["lab"])

        lab = item["lab"]
        if neighbours and smoothing > 0:
            mean = tuple(
                sum(n[k] for n in neighbours) / len(neighbours) for k in range(3)
            )
            lab = mix_oklab(lab, mean, float(smoothing))

        smoothed.append(oklab_to_hex(lab))

    bg = "#111418" if dark else "#FFFFFF"
    shadow = (
        "0 1px 3px rgba(0,0,0,.85)" if dark else "0 1px 2px rgba(255,255,255,.9)"
    )

    spans = []
    for item, color in zip(words, smoothed):
        # размер <- громкость, жирность <- arousal эмоции, усиленный "екпін"
        size = size_min + (size_max - size_min) * item["loud"]
        arousal = (EMOTIONS[item["emotion"]][1] + 1) / 2
        weight_t = min(1.0, arousal * (0.5 + item["stress"]))
        weight = int(round(weight_min + (weight_max - weight_min) * weight_t))
        weight = max(100, min(900, weight // 100 * 100))

        spans.append(
            f'<span style="color:{color};font-size:{size:.1f}px;'
            f"font-weight:{weight};line-height:1.5;"
            f'text-shadow:{shadow};margin-right:.28em;display:inline-block">'
            # экранируем: текст приходит от пользователя и вставляется в HTML
            f"{html.escape(item['word'])}"
            f"</span>"
        )

    return (
        f'<div style="background:{bg};padding:34px 28px;border-radius:14px;'
        f'text-align:center;font-family:Inter,Arial,sans-serif">'
        + "".join(spans)
        + "</div>"
    )


# ----------------------------------------------------------------------------
# Интерфейс
# ----------------------------------------------------------------------------


def build() -> gr.Blocks:
    with gr.Blocks(title="AItylym — кинетические субтитры") as demo:
        gr.Markdown(
            "# AItylym — динамикалық субтитрлер\n"
            "Мәтін жазыңыз, әр сөздің айтылу мәнерін кестеде көрсетіңіз "
            "және оформлениені баптаңыз.\n\n"
            "*Прототип: просодия әзірге қолмен енгізіледі.*"
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Мәтін",
                    value=SAMPLE_TEXT,
                    lines=2,
                )

                table = gr.Dataframe(
                    value=text_to_table(SAMPLE_TEXT),
                    headers=COLUMNS,
                    datatype=["str", "str", "number", "number"],
                    interactive=True,
                    label="Әр сөз: эмоция, қаттылық (0-1), екпін (0-1)",
                )

                gr.Markdown(
                    "Эмоциялар: " + ", ".join(f"`{e}`" for e in EMOTIONS)
                )

            with gr.Column(scale=2):
                dark = gr.Checkbox(value=True, label="Қараңғы фон (бейне сияқты)")

                with gr.Accordion("Түстер", open=True):
                    pickers = [
                        gr.ColorPicker(value=DEFAULT_COLORS[name], label=name)
                        for name in EMOTIONS
                    ]

                with gr.Accordion("Оформление", open=False):
                    smoothing = gr.Slider(
                        0, 0.5, value=0.18, step=0.02,
                        label="Көршілес түстерді араластыру (Oklab)",
                    )
                    size_min = gr.Slider(14, 40, value=22, step=1, label="Ең кіші қаріп")
                    size_max = gr.Slider(20, 80, value=44, step=1, label="Ең үлкен қаріп")
                    weight_min = gr.Slider(100, 900, value=300, step=100, label="Ең жіңішке")
                    weight_max = gr.Slider(100, 900, value=800, step=100, label="Ең қалың")

        preview = gr.HTML(label="Нәтиже")

        inputs = [
            table, smoothing, size_min, size_max,
            weight_min, weight_max, dark, *pickers,
        ]

        text.change(text_to_table, text, table).then(render, inputs, preview)
        for component in inputs:
            component.change(render, inputs, preview)

        demo.load(render, inputs, preview)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="публичная ссылка")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    build().launch(share=args.share, server_port=args.port)
