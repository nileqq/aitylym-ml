"""OmniASR: аудио -> слова с границами.

Почему тайминги вообще есть
---------------------------
Высокоуровневый метод пайплайна, transcribe(), возвращает только список
строк — таймингов там нет. Но модель CTC-шная: она классифицирует каждый
кадр отдельно, и если взять логиты вместо готовой строки, номера кадров
сохраняются.

    частота кадров : 20.05 мс (49.9 в секунду)
    токенайзер     : посимвольный
    пробел         : отдельный токен, id 4

Жадный CTC-декод — argmax по кадрам, схлопнуть повторы, выбросить blank.
Первые два шага сохраняют номер кадра, поэтому, разрезав последовательность
по токену-пробелу, получаем границы слов: кадр i это i * 20.05 мс.

Реализация лежит в prototype/omniasr_words.py и там же проверена:
57.6 с аудио -> 94 слова, ноль перекрытий, RTF 0.0053.

Модель грузится около 24 секунд, поэтому пайплайн кешируется в модуле.
"""

from __future__ import annotations

import sys
from pathlib import Path

# omniasr_words пока живёт в prototype/. Если переедет — менять здесь.
_PROTOTYPE = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPE))

import omniasr_words

_pipeline = None


def load():
    """Грузит модель. Вызывать при старте приложения, а не в обработчике:
    24 секунды ожидания по клику выглядят как зависание."""
    global _pipeline
    if _pipeline is None:
        _pipeline = omniasr_words.load_pipeline()
    return _pipeline


def words(y, sampling_rate: int = 16000) -> list:
    """Аудио -> [(слово, начало, конец), ...] в секундах."""
    pipeline = load()
    return [
        (w.word, w.start, w.end)
        for w in omniasr_words.transcribe_words(y, sampling_rate, pipeline=pipeline)
    ]


def text(y, sampling_rate: int = 16000) -> str:
    """Аудио -> сплошной транскрипт."""
    return " ".join(word for word, _, _ in words(y, sampling_rate))
