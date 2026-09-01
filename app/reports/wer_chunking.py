"""Причинный тест: одно и то же аудио целиком против него же, порезанного.

Корреляция «короткие клипы -> высокий WER» может объясняться и тем, что
короткие клипы просто труднее: обрывки, междометия, неполные слова. Чтобы
отделить длину от содержания, берём один и тот же материал и распознаём
двумя способами:

    A) склеенный поток целиком
    B) те же клипы по отдельности

Содержание идентично, отличается только нарезка. Если A заметно лучше —
резать перед ASR действительно вредно.
"""

import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
import asr

SR = 16000
MAX_STREAM_SEC = 35.0          # лимит модели 40 с, оставляем запас
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text):
    return " ".join(_PUNCT.sub(" ", str(text or "").lower()).split())


meta = pd.read_csv(ROOT / "data" / "metadata" / "metadata.csv")

# Берём подряд идущие клипы одного видео и одного говорящего: так склейка
# остаётся связной речью, а не нарезкой из разных источников.
groups = []
for (video, speaker), part in meta.groupby(["video_id", "speaker_id"], sort=False):
    part = part.head(40)
    picked, total = [], 0.0
    for record in part.itertuples():
        path = ROOT / str(record.audio_path).replace("\\", "/")
        if not path.exists():
            continue
        try:
            duration = sf.info(str(path)).duration
        except Exception:
            continue
        if total + duration > MAX_STREAM_SEC:
            break
        picked.append((str(path), normalize(record.text), duration))
        total += duration

    if len(picked) >= 4 and total >= 12.0:
        groups.append(picked)
    if len(groups) >= 30:
        break

print(f"групп: {len(groups)}, клипов в них: {sum(len(g) for g in groups)}")

pipeline = asr.load()

whole_wer, split_wer, whole_cer, split_cer, weights = [], [], [], [], []

for i, group in enumerate(groups):
    reference = " ".join(ref for _, ref, _ in group)
    if not reference:
        continue

    # A: склеиваем и распознаём одним куском
    chunks = []
    for path, _, _ in group:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        chunks.append(np.asarray(wav, dtype=np.float32))
    stream = np.concatenate(chunks)

    whole = normalize(pipeline.transcribe([{"waveform": stream, "sample_rate": SR}])[0])

    # B: те же клипы по отдельности
    parts = pipeline.transcribe([p for p, _, _ in group], batch_size=8)
    split = normalize(" ".join(parts))

    whole_wer.append(jiwer.wer(reference, whole) if whole else 1.0)
    split_wer.append(jiwer.wer(reference, split) if split else 1.0)
    whole_cer.append(jiwer.cer(reference, whole) if whole else 1.0)
    split_cer.append(jiwer.cer(reference, split) if split else 1.0)
    weights.append(len(reference.split()))

    if i == 0:
        print(f"\nпример (группа из {len(group)} клипов, {sum(d for _,_,d in group):.1f} с):")
        print(f"  эталон : {reference[:90]}")
        print(f"  целиком: {whole[:90]}")
        print(f"  порезан: {split[:90]}")

weights = np.array(weights, dtype=float)


def weighted(values):
    return float(np.sum(np.array(values) * weights) / np.sum(weights))


print(f"\n{'=' * 50}")
print(f"{len(whole_wer)} групп, одно и то же аудио двумя способами")
print("=" * 50)
print(f"{'':22s} {'WER':>9s} {'CER':>9s}")
print(f"{'целиком':22s} {weighted(whole_wer) * 100:8.1f}% {weighted(whole_cer) * 100:8.1f}%")
print(f"{'порезано на клипы':22s} {weighted(split_wer) * 100:8.1f}% {weighted(split_cer) * 100:8.1f}%")

delta = weighted(split_wer) - weighted(whole_wer)
print(f"\nразница: {delta * 100:+.1f} пункта WER "
      f"({'резать вредно' if delta > 0 else 'резать не вредит'})")

better = sum(1 for w, s in zip(whole_wer, split_wer) if w < s)
print(f"групп, где целиком лучше: {better} из {len(whole_wer)}")
