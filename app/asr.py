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

import os
import sys
from pathlib import Path

# omniasr_words пока живёт в prototype/. Если переедет — менять здесь.
_PROTOTYPE = Path(__file__).resolve().parent.parent / "prototype"
if str(_PROTOTYPE) not in sys.path:
    sys.path.insert(0, str(_PROTOTYPE))

import omniasr_words

# Какую модель брать. Измерено на своём корпусе (см. reports/REPORT_asr.md):
#
#   300M  WER 63.2%   чекпоинт 1.3 ГБ   24 клип/с
#   1B    WER 49.0%   чекпоинт 3.9 ГБ   16 клип/с
#
# 1B заметно лучше, но требует GPU и свопа под пик загрузки. На бесплатном
# HF Spaces видеокарты нет и два ядра CPU, поэтому там остаётся 300M —
# иначе холодный старт тянет 3.9 ГБ, а инференс идёт минутами.
#
# Переключается переменной окружения, чтобы одно и то же приложение
# работало и локально на GPU, и на Spaces:
#     AITYLYM_ASR_MODEL=omniASR_CTC_1B_v2
DEFAULT_MODEL = os.environ.get("AITYLYM_ASR_MODEL", "omniASR_CTC_300M_v2")

_pipeline = None
_loaded_card = None


def load(model_card: str = DEFAULT_MODEL):
    """Грузит модель. Вызывать при старте приложения, а не в обработчике:
    загрузка занимает 20-25 секунд и по клику выглядит как зависание."""
    global _pipeline, _loaded_card
    if _pipeline is None or _loaded_card != model_card:
        _pipeline = omniasr_words.load_pipeline(model_card)
        _loaded_card = model_card
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
