"""AItylym — каркас приложения.

Принимает аудиофайл, отдаёт результат обработки. Сама обработка пока не
реализована: process() возвращает None, интерфейс это переживает и
показывает заглушку.

Запуск:
    gradio app/main.py     # с горячей перезагрузкой
    python app/main.py
"""

import gradio as gr


def process(audio_path):
    """Аудио -> результат. Здесь будет ASR, эмоции и просодия.

    Args:
        audio_path: путь к файлу, либо None если ничего не загружено.

    Returns:
        Пока None. Дальше — то, что нужно показать пользователю.
    """
    if audio_path is None:
        return None

    # TODO: VAD / нарезка
    # TODO: ASR -> слова с таймингами
    # TODO: эмоция
    # TODO: просодия пословно
    # TODO: покрасить

    return None


def on_click(audio_path):
    """Прослойка между process() и интерфейсом.

    Нужна, чтобы None не ломал вывод: пока process() ничего не возвращает,
    показываем заглушку вместо пустоты.
    """
    result = process(audio_path)

    if result is None:
        return "_Нәтиже жоқ_"

    return result


with gr.Blocks(title="AItylym") as demo:
    gr.Markdown("# AItylym\n### Қазақ тіліндегі динамикалық субтитрлер")

    audio = gr.Audio(label="Аудио", type="filepath")
    run = gr.Button("Талдау", variant="primary")
    output = gr.Markdown()

    run.click(on_click, audio, output)


if __name__ == "__main__":
    demo.launch()
