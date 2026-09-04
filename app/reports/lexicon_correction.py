"""Притягивание распознанных слов к словарю по расстоянию Левенштейна.

Идея
----
CER заметно ниже WER (12.7% против 35% на длинных репликах), значит
большинство неверных слов ошибаются на одну-две буквы. Если заменить такое
слово ближайшим словарным, часть ошибок должна исчезнуть.

Честность замера
----------------
Словарь строится по ПЕРВОЙ половине корпуса, а WER считается на ВТОРОЙ.
Иначе словарь содержал бы ровно те слова, которые надо угадать, и выигрыш
был бы фиктивным.

Порог обязателен: если верного слова в словаре нет, притягивание
превращает «почти правильное» слово в уверенно неправильное. Поэтому
заменяем только при малом расстоянии, а далёкие оставляем как есть.

Запуск:
    python app/reports/lexicon_correction.py asr_wer_1b.csv
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(__file__).resolve().parent

# Максимальное расстояние, при котором подмена допустима. Считается
# относительно длины слова: для короткого слова одна ошибка — это много,
# для длинного мало.
MAX_REL_DISTANCE = 0.34

# Слова короче этого не трогаем: у них любое расстояние велико
# относительно длины, и подмена почти всегда портит.
MIN_LEN = 4

# Сколько раз слово должно встретиться в первой половине, чтобы попасть
# в словарь. Единичные вхождения — чаще всего сами ошибки распознавания
# или опечатки в разметке.
MIN_COUNT = 2


def levenshtein(a: str, b: str, limit: int) -> int:
    """Расстояние Левенштейна с ранним выходом.

    limit нужен для скорости: если минимальное значение в строке уже
    превысило порог, дальше считать незачем — слово всё равно не подойдёт.
    """
    if abs(len(a) - len(b)) > limit:
        return limit + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def build_vocabulary(rows: list) -> dict:
    """Словарь из эталонов: слово -> частота. Сгруппирован по длине,
    чтобы не сравнивать со всеми подряд."""
    counter = Counter()
    for row in rows:
        counter.update(row["reference"].split())

    vocabulary = {w for w, n in counter.items() if n >= MIN_COUNT}

    by_length = {}
    for word in vocabulary:
        by_length.setdefault(len(word), []).append(word)
    return by_length


def correct(word: str, by_length: dict) -> str:
    """Ближайшее словарное слово, если оно достаточно близко."""
    if len(word) < MIN_LEN:
        return word

    limit = int(len(word) * MAX_REL_DISTANCE)
    if limit < 1:
        return word

    best, best_distance = word, limit + 1
    # Кандидаты только той длины, что вообще может уложиться в порог
    for length in range(len(word) - limit, len(word) + limit + 1):
        for candidate in by_length.get(length, ()):
            if candidate == word:
                return word
            distance = levenshtein(word, candidate, limit)
            if distance < best_distance:
                best, best_distance = candidate, distance

    return best if best_distance <= limit else word


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "asr_wer_1b.csv"
    rows = list(csv.DictReader(open(OUT_DIR / name, encoding="utf-8")))
    print(f"файл: {name}, строк: {len(rows)}")

    half = len(rows) // 2
    vocabulary = build_vocabulary(rows[:half])
    test = rows[half:]

    size = sum(len(v) for v in vocabulary.values())
    print(f"словарь из первой половины: {size} слов")
    print(f"тест на второй половине  : {len(test)} клипов\n")

    before, after, weights = [], [], []
    changed = 0
    total_words = 0

    for row in test:
        reference = row["reference"]
        hypothesis = row["hypothesis"]
        if not reference or not hypothesis:
            continue

        words = hypothesis.split()
        fixed = []
        for word in words:
            new = correct(word, vocabulary)
            fixed.append(new)
            if new != word:
                changed += 1
        total_words += len(words)

        corrected = " ".join(fixed)
        before.append(jiwer.wer(reference, hypothesis))
        after.append(jiwer.wer(reference, corrected))
        weights.append(len(reference.split()))

    weights = np.array(weights, dtype=float)

    def weighted(values):
        return float(np.sum(np.array(values) * weights) / np.sum(weights))

    print(f"подменено слов: {changed} из {total_words} "
          f"({changed / max(total_words, 1) * 100:.1f}%)\n")

    print(f"{'':22s} {'WER':>9s}")
    print(f"{'без словаря':22s} {weighted(before) * 100:8.2f}%")
    print(f"{'со словарём':22s} {weighted(after) * 100:8.2f}%")

    delta = weighted(after) - weighted(before)
    verdict = "помогает" if delta < -0.002 else (
        "вредит" if delta > 0.002 else "не меняет")
    print(f"\nразница: {delta * 100:+.2f} пункта — {verdict}")

    improved = sum(1 for b, a in zip(before, after) if a < b)
    worsened = sum(1 for b, a in zip(before, after) if a > b)
    print(f"клипов лучше: {improved}, хуже: {worsened}, "
          f"без изменений: {len(before) - improved - worsened}")


if __name__ == "__main__":
    main()
