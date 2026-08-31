"""AItylym — прототип кинетических субтитров.

Что это
-------
Пишешь текст — сразу под ним видишь субтитр. Кликаешь по слову — настраиваешь
именно его. Слева палитра: какой эмоции какой цвет.

Аудио здесь ещё нет: просодия задаётся руками. Это намеренно — прототип
проверяет ровно **алгоритм оформления**, отдельно от качества ASR и
распознавания эмоций. Когда `omniasr_words.py` начнёт отдавать слова с
таймингами, а DSP — громкость и высоту, они просто заполнят то же состояние,
и рендер не изменится.

Маппинг взят из CuCap (ASSETS'25), а не придуман:
    цвет     <- эмоция (валентность)
    размер   <- громкость   — самое устойчивое соответствие, 29% NA / 38% KOR
    жирность <- arousal
Высота голоса отдельно не кодируется: у CuCap 79% участников из Северной
Америки выбрали для неё «никак».

Слова — настоящие кнопки Gradio, а не HTML: клик по <span> внутри gr.HTML
до Python не доходит, а по кнопке доходит. Внешний вид кнопкам задаётся
инъекцией <style> с правилами по elem_id.

Запуск:
    python prototype/app.py            # локально
    python prototype/app.py --share    # публичная ссылка gradio.live
"""

from __future__ import annotations

import argparse
import math
import re

import gradio as gr

# ----------------------------------------------------------------------------
# Эмоции и их координаты в пространстве valence-arousal
# ----------------------------------------------------------------------------

# Валентность решает цвет, arousal — жирность. Значения примерные: уточнить их
# и должен опрос, ради него всё затевалось.
EMOTIONS = {
    "бейтарап": (0.0, 0.0),
    "қуаныш": (0.8, 0.6),
    "ашу": (-0.7, 0.8),
    "қорқыныш": (-0.7, 0.7),
    "қайғы": (-0.7, -0.5),
    "жиіркеніш": (-0.6, 0.2),
    "таңданыс": (0.2, 0.8),
}

DEFAULT_COLORS = {
    "бейтарап": "#D9D9D9",
    "қуаныш": "#F5C518",
    "ашу": "#E23B3B",
    "қорқыныш": "#8B5CF6",
    "қайғы": "#4A90E2",
    "жиіркеніш": "#4CAF50",
    "таңданыс": "#FF8A3D",
}

SAMPLE_TEXT = "Сен бүгін келдің деп ойламадым мен"


# ----------------------------------------------------------------------------
# Oklab: смешивание цветов без грязи
# ----------------------------------------------------------------------------
#
# Интерполяция в RGB тащит переход между насыщенными цветами через мутную
# середину — красный с синим дают грязно-серый. Oklab перцептивно равномерен,
# и та же середина выглядит чистой.


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
    except (ValueError, IndexError):
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
# Состояние
# ----------------------------------------------------------------------------


def text_to_words(text: str, previous=None) -> list:
    """Разбирает текст на слова. Настройки уже размеченных слов сохраняются:
    иначе каждая опечатка в конце строки сбрасывала бы всю работу."""
    previous = previous or []
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]

    out = []
    for i, word in enumerate(words):
        if i < len(previous) and previous[i].get("word") == word:
            out.append(previous[i])
        else:
            out.append(
                {"word": word, "emotion": "бейтарап", "loud": 0.5, "stress": 0.5}
            )
    return out


def _clamp01(value, fallback=0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return fallback


def styles_for(words, palette, smoothing, size_min, size_max,
               weight_min, weight_max) -> list:
    """Считает финальные цвет/размер/жирность для каждого слова."""
    labs = [
        hex_to_oklab(palette.get(w.get("emotion", "бейтарап"), "#D9D9D9"))
        for w in words
    ]

    out = []
    for i, word in enumerate(words):
        # Сглаживание: подмешиваем цвета соседей. Требование из
        # algorithm.docx — «цвета соседние смешиваются, чтобы не было
        # резкого перехода». В Oklab, иначе середина уходит в грязь.
        neighbours = []
        if i > 0:
            neighbours.append(labs[i - 1])
        if i < len(labs) - 1:
            neighbours.append(labs[i + 1])

        lab = labs[i]
        if neighbours and smoothing > 0:
            mean = tuple(
                sum(n[k] for n in neighbours) / len(neighbours) for k in range(3)
            )
            lab = mix_oklab(lab, mean, float(smoothing))

        loud = _clamp01(word.get("loud"))
        stress = _clamp01(word.get("stress"))
        arousal = (EMOTIONS.get(word.get("emotion"), (0, 0))[1] + 1) / 2

        size = size_min + (size_max - size_min) * loud
        weight_t = min(1.0, arousal * (0.5 + stress))
        weight = int(weight_min + (weight_max - weight_min) * weight_t)
        weight = max(100, min(900, weight // 100 * 100))

        out.append(
            {"color": oklab_to_hex(lab), "size": size, "weight": weight}
        )
    return out


def caption_css(styles, selected, dark: bool) -> str:
    """CSS для кнопок-слов. Gradio не даёт задать кнопке инлайновый стиль,
    поэтому правила вешаются на elem_id."""
    shadow = "0 1px 3px rgba(0,0,0,.85)" if dark else "0 1px 2px rgba(255,255,255,.9)"

    bg = "#111418" if dark else "#FFFFFF"

    # В Gradio 6 elem_classes и elem_id ставятся ПРЯМО на <button>, а не на
    # обёртку. Поэтому селекторы одноуровневые: ".aitylym-word button" не
    # совпал бы ни с чем, и кнопки остались бы серыми плашками.
    rules = [
        # полоса субтитра
        f".aitylym-strip{{background:{bg}!important;border-radius:14px!important;"
        "padding:24px 16px!important;display:flex!important;flex-wrap:wrap!important;"
        "justify-content:center!important;align-items:baseline!important;"
        "gap:0!important;border:none!important}",
        # слово: убираем всё кнопочное и запрещаем флексу его тянуть,
        # иначе слова наезжают друг на друга
        ".aitylym-word{background:none!important;border:none!important;"
        "box-shadow:none!important;padding:2px 6px!important;"
        "flex:0 0 auto!important;width:auto!important;min-width:0!important;"
        "max-width:none!important;white-space:nowrap!important;"
        "line-height:1.4!important;transition:none!important}",
        ".aitylym-word:hover{background:rgba(127,127,127,.18)!important;"
        "border-radius:6px!important}",
    ]

    for i, style in enumerate(styles):
        ring = (
            "outline:2px solid #4A90E2!important;border-radius:6px!important;"
            if i == selected
            else ""
        )
        rules.append(
            f"#aitylym-w{i}{{"
            f"color:{style['color']}!important;"
            f"font-size:{style['size']:.1f}px!important;"
            f"font-weight:{style['weight']}!important;"
            f"text-shadow:{shadow};{ring}}}"
        )

    return "<style>" + "".join(rules) + "</style>"


# ----------------------------------------------------------------------------
# Интерфейс
# ----------------------------------------------------------------------------


def build() -> gr.Blocks:
    with gr.Blocks(title="AItylym — динамикалық субтитрлер") as demo:
        words_state = gr.State(text_to_words(SAMPLE_TEXT))
        selected_state = gr.State(None)

        gr.Markdown(
            "# AItylym — динамикалық субтитрлер\n"
            "Мәтін жазыңыз, төменде нәтижені көресіз. "
            "Сөзді басып, оның айтылу мәнерін баптаңыз."
        )

        with gr.Row():
            # ---------------- Слева: палитра ----------------
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### Түстер")
                pickers = [
                    gr.ColorPicker(value=DEFAULT_COLORS[name], label=name)
                    for name in EMOTIONS
                ]

            # ---------------- Справа: текст и результат ----------------
            with gr.Column(scale=3):
                text = gr.Textbox(label="Мәтін", value=SAMPLE_TEXT, lines=2)

                dark = gr.Checkbox(value=True, label="Қараңғы фон (бейне сияқты)")

                gr.Markdown("### Нәтиже")

                # Без явного triggers: тогда gr.render сам перерисовывается
                # и при загрузке страницы, и на любое изменение входов.
                # Со списком triggers первичная отрисовка не происходила —
                # subtitle появлялся только после первого действия.
                @gr.render(
                    inputs=[words_state, selected_state, dark, *pickers],
                    show_progress="hidden",
                )
                def draw_caption(words, selected, is_dark, *colors):
                    palette = dict(zip(EMOTIONS.keys(), colors))
                    styles = styles_for(
                        words, palette, 0.18, 22, 46, 300, 800
                    )

                    gr.HTML(caption_css(styles, selected, is_dark))

                    if not words:
                        gr.Markdown("*Мәтін енгізіңіз*")
                        return

                    with gr.Row(elem_classes=["aitylym-strip"]):
                        for i, word in enumerate(words):
                            btn = gr.Button(
                                word["word"],
                                elem_id=f"aitylym-w{i}",
                                elem_classes=["aitylym-word"],
                                scale=0,
                                min_width=1,
                            )
                            btn.click(
                                lambda i=i: i, None, selected_state,
                                show_progress="hidden",
                            )


                # ---------------- Настройка выбранного слова ----------------
                with gr.Group():
                    picked = gr.Markdown("*Сөзді басыңыз*")
                    emotion = gr.Radio(
                        list(EMOTIONS), label="Эмоция", interactive=True
                    )
                    loud = gr.Slider(0, 1, value=0.5, step=0.05, label="Қаттылық")
                    stress = gr.Slider(0, 1, value=0.5, step=0.05, label="Екпін")

        # -------------------- Логика --------------------

        def on_text(new_text, words):
            return text_to_words(new_text, words), None

        text.change(
            on_text, [text, words_state], [words_state, selected_state]
        )

        def on_select(index, words):
            """Клик по слову — подставляем его значения в контролы."""
            if index is None or index >= len(words):
                return "*Сөзді басыңыз*", None, 0.5, 0.5
            word = words[index]
            return (
                f"**{word['word']}**",
                word.get("emotion", "бейтарап"),
                word.get("loud", 0.5),
                word.get("stress", 0.5),
            )

        selected_state.change(
            on_select,
            [selected_state, words_state],
            [picked, emotion, loud, stress],
        )

        def edit(field):
            def apply(value, index, words):
                if index is None or index >= len(words) or value is None:
                    return gr.skip()
                words = [dict(w) for w in words]
                words[index][field] = value
                return words

            return apply

        emotion.change(edit("emotion"), [emotion, selected_state, words_state], words_state)
        loud.change(edit("loud"), [loud, selected_state, words_state], words_state)
        stress.change(edit("stress"), [stress, selected_state, words_state], words_state)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="публичная ссылка")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    build().launch(share=args.share, server_port=args.port)
