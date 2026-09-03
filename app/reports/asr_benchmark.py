"""WER/CER OmniASR на собственном корпусе (6610 клипов).

Зачем
-----
WER 22% у OmniASR измерен на FLEURS — это чистая студийная начитка. Реальный
материал AItylym другой: неформальная речь с YouTube. Насколько модель
проседает на нём, до сих пор не измерялось, а от этой цифры зависит, хватит
ли OmniASR или нужен MMS.

Тестовый набор
--------------
data/metadata/metadata.csv: 6610 клипов, у каждого эталонный транскрипт.
Домен ровно тот, ради которого всё делается.

Оговорка, которую нужно держать в голове
----------------------------------------
Эталоны пришли вместе с исходным датасетом, и написаны они неровно —
заглавные буквы посреди фразы, разнобой в передаче заимствований. Похоже,
это не выверенная вручную разметка. Значит WER здесь меряет расхождение с
этими транскриптами, а не с идеальной истиной, и абсолютное значение будет
завышено. Для сравнения моделей между собой это не мешает: обе меряются об
один и тот же эталон.

Запуск:
    python app/reports/asr_benchmark.py --limit 200      # быстрая прикидка
    python app/reports/asr_benchmark.py                  # весь корпус
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent

# Чекпоинт свой на каждую модель, иначе результаты смешаются.
def checkpoint_for(model_card):
    short = model_card.replace("omniASR_CTC_", "").replace("_v2", "")
    return OUT_DIR / f"asr_wer_{short.lower()}.csv"
FIELDS = ["sample_id", "reference", "hypothesis", "wer", "cer", "n_words"]

# Пунктуация и регистр к делу не относятся: модель их не предсказывает,
# и наказывать её за это значит мерить не то.
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def load_rows(limit=None) -> list:
    import pandas as pd

    meta = pd.read_csv(ROOT / "data" / "metadata" / "metadata.csv")

    rows = []
    for record in meta.itertuples():
        reference = normalize(record.text)
        if not reference:
            continue

        # audio_path в метаданных записан с обратными слэшами Windows
        path = ROOT / str(record.audio_path).replace("\\", "/")
        if not path.exists():
            continue

        rows.append((record.sample_id, str(path), reference))
        if limit and len(rows) >= limit:
            break

    return rows


# OmniASR отказывается принимать аудио длиннее этого — жёсткая проверка
# внутри пайплайна, не наша настройка.
MAX_AUDIO_SEC = 40.0


def drop_too_long(rows: list) -> tuple:
    """Отсеивает клипы, которые модель не примет.

    Их 12 из 6610 (0.18%), так что на итог они не влияют, но одна такая
    запись роняет весь пакет целиком.
    """
    import soundfile as sf

    keep, dropped = [], []
    for row in rows:
        try:
            if sf.info(row[1]).duration > MAX_AUDIO_SEC:
                dropped.append(row[0])
            else:
                keep.append(row)
        except Exception:
            dropped.append(row[0])

    return keep, dropped


def already_done() -> set:
    if not CHECKPOINT.exists():
        return set()
    with CHECKPOINT.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="начать заново")
    parser.add_argument("--model", default=None,
                        help="карточка модели, например omniASR_CTC_1B_v2")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "app"))
    import asr

    model_card = args.model or asr.DEFAULT_MODEL
    global CHECKPOINT
    CHECKPOINT = checkpoint_for(model_card)

    if args.fresh and CHECKPOINT.exists():
        CHECKPOINT.unlink()

    import jiwer
    import numpy as np

    rows = load_rows(args.limit)
    rows, dropped = drop_too_long(rows)
    if dropped:
        print(f"пропущено (длиннее {MAX_AUDIO_SEC:.0f} с или битые): {len(dropped)}")

    done = already_done()
    todo = [r for r in rows if str(r[0]) not in done]

    print(f"клипов всего: {len(rows)}, уже обработано: {len(done)}, "
          f"осталось: {len(todo)}", flush=True)

    if todo:
        print(f"загрузка модели {model_card}...", flush=True)
        pipeline = asr.load(model_card)

        new_file = not CHECKPOINT.exists()
        handle = CHECKPOINT.open("a", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        started = time.perf_counter()
        for offset in range(0, len(todo), args.batch):
            chunk = todo[offset:offset + args.batch]

            # Для WER тайминги не нужны, поэтому берём пакетный transcribe():
            # он гораздо быстрее, чем прогон по одному файлу.
            hypotheses = pipeline.transcribe([p for _, p, _ in chunk],
                                             batch_size=len(chunk))

            batch_rows = []
            for (sample_id, _, reference), raw in zip(chunk, hypotheses):
                hypothesis = normalize(raw)
                batch_rows.append({
                    "sample_id": sample_id,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": jiwer.wer(reference, hypothesis) if hypothesis else 1.0,
                    "cer": jiwer.cer(reference, hypothesis) if hypothesis else 1.0,
                    "n_words": len(reference.split()),
                })

            writer.writerows(batch_rows)
            handle.flush()
            os.fsync(handle.fileno())

            processed = offset + len(chunk)
            if processed % (args.batch * 20) == 0 or processed == len(todo):
                rate = processed / (time.perf_counter() - started)
                left = (len(todo) - processed) / max(rate, 1e-6)
                print(f"  {processed}/{len(todo)}  {rate:.1f} клип/с  "
                      f"осталось ~{left / 60:.1f} мин", flush=True)

        handle.close()

    # ---------------- итоги ----------------

    with CHECKPOINT.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))

    wer = np.array([float(r["wer"]) for r in results])
    cer = np.array([float(r["cer"]) for r in results])
    words = np.array([int(r["n_words"]) for r in results])

    # Взвешенный WER — правильная агрегация: длинные фразы должны весить
    # больше, чем однословные. Среднее по клипам их уравнивает и обычно
    # завышает результат.
    weighted = float(np.sum(wer * words) / np.sum(words))

    print(f"\n{'=' * 46}")
    print(f"{model_card} на собственном корпусе ({len(results)} клипов)")
    print("=" * 46)
    print(f"  WER взвешенный по словам : {weighted * 100:6.2f}%")
    print(f"  WER среднее по клипам    : {wer.mean() * 100:6.2f}%")
    print(f"  WER медиана              : {np.median(wer) * 100:6.2f}%")
    print(f"  CER среднее              : {cer.mean() * 100:6.2f}%")
    print()
    print(f"  клипов с WER = 0         : {(wer == 0).sum():5d} "
          f"({(wer == 0).mean() * 100:.1f}%)")
    print(f"  клипов с WER > 100%      : {(wer > 1).sum():5d} "
          f"({(wer > 1).mean() * 100:.1f}%)")
    print()
    print("  распределение WER:")
    for low, high in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 10)]:
        share = ((wer >= low) & (wer < high)).mean()
        bar = "#" * int(share * 40)
        print(f"    {low * 100:3.0f}-{min(high, 1) * 100:3.0f}%  "
              f"{share * 100:5.1f}%  {bar}")

    print(f"\n  подробности: {CHECKPOINT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
