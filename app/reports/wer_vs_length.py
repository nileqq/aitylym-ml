"""Проверка гипотезы: страдает ли OmniASR именно от коротких отрезков.

Если WER резко падает с ростом длительности — значит резать поток на
короткие фразы перед ASR вредно, и OmniASR надо кормить длинными кусками,
а границы слов брать из CTC-кадров.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
rows = list(csv.DictReader(
    open(Path(__file__).parent / "asr_wer_checkpoint.csv", encoding="utf-8")
))

dur, wer, cer, words = [], [], [], []
for row in rows:
    path = ROOT / "data" / "audio" / f"{row['sample_id']}.wav"
    try:
        duration = sf.info(str(path)).duration
    except Exception:
        continue
    dur.append(duration)
    wer.append(float(row["wer"]))
    cer.append(float(row["cer"]))
    words.append(int(row["n_words"]))

dur = np.array(dur)
wer = np.array(wer)
cer = np.array(cer)
words = np.array(words)

print(f"проанализировано клипов: {len(dur)}\n")

header = f"{'длительность':>14s} {'клипов':>7s} {'WER':>8s} {'CER':>8s}"
print(header)
print("-" * len(header))

for low, high in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 45)]:
    mask = (dur >= low) & (dur < high)
    if mask.sum() < 5:
        continue
    # WER взвешиваем по числу слов: иначе однословные клипы, где одна
    # ошибка даёт сразу 100%, перекашивают картину
    weighted = float(np.sum(wer[mask] * words[mask]) / max(np.sum(words[mask]), 1))
    print(f"{low:5.0f}-{high:<4.0f} с {mask.sum():10d} "
          f"{weighted * 100:7.1f}% {cer[mask].mean() * 100:7.1f}%")

try:
    from scipy.stats import spearmanr
    rho, pval = spearmanr(dur, wer)
    print(f"\nкорреляция длительность-WER: rho={rho:+.3f}, p={pval:.2e}")
except ImportError:
    pass

short = dur < 3
print(f"\nклипы короче 3 с: {short.sum()} ({short.mean() * 100:.0f}% корпуса)")
print(f"  WER на коротких : "
      f"{np.sum(wer[short] * words[short]) / np.sum(words[short]) * 100:.1f}%")
print(f"  WER на длинных  : "
      f"{np.sum(wer[~short] * words[~short]) / np.sum(words[~short]) * 100:.1f}%")
