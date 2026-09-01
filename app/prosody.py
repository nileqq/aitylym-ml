"""Просодия: громкость и высота голоса по отрезкам аудио.

Работает с любыми отрезками — сейчас это куски от VAD, потом будут слова
с таймингами от OmniASR. Интерфейс один и тот же.

Четыре решения, которые здесь приняты, и почему
-----------------------------------------------
1. Громкость в дБ, а не в линейном RMS. Слух логарифмический: на линейной
   шкале тихая речь выглядит ровнее, чем звучит.

2. Высота в полутонах относительно медианы говорящего, а не в герцах.
   Мужской голос около 110 Гц, женский около 210 — в герцах их шкалы
   несравнимы, а в полутонах от собственной медианы обе становятся
   «насколько говорящий отклонился от своей нормы».

3. Нормализация по перцентилям 5/95, а не min-max. Один вскрик или один
   щелчок микрофона задирают максимум, и всё остальное сплющивается в
   нижнюю треть шкалы.

4. F0 считается ОДИН раз на весь файл, дальше нарезается по отрезкам.
   librosa.pyin очень медленный; вызывать его на каждое слово — значит
   ждать минуты вместо секунд.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Шаг анализа: 20 мс при 16 кГц. Ровно столько же длится кадр OmniASR
# (20.05 мс), поэтому сетка просодии и сетка слов совпадают без пересчёта.
HOP_LENGTH = 320

# Границы поиска высоты. Для речи этого достаточно: 70 Гц — низкий мужской
# голос, 400 Гц — высокий женский. Более широкий диапазон только замедляет
# pyin и добавляет ложные срабатывания на шуме.
F0_MIN, F0_MAX = 70.0, 400.0


@dataclass
class Prosody:
    """Просодия одного отрезка."""

    loud: float          # 0..1, нормализовано по всей записи
    pitch: float         # 0..1, нормализовано; 0.5 если отрезок глухой
    voiced: bool         # была ли вообще различима высота
    loud_db: float       # сырое значение, дБ относительно максимума
    pitch_semitones: float   # сырое значение, полутоны от медианы говорящего


def _robust_scale(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Нормализация по перцентилям 5/95 с обрезкой по краям."""
    out = np.full(len(values), 0.5, dtype=np.float32)

    if mask.sum() < 2:
        return out

    low, high = np.percentile(values[mask], [5, 95])
    if high - low < 1e-6:
        return out

    out[mask] = np.clip((values[mask] - low) / (high - low), 0.0, 1.0)
    return out


def analyze(y: np.ndarray, sr: int, segments: list) -> list:
    """Считает просодию для списка отрезков.

    Args:
        y: моно-аудио.
        sr: частота дискретизации.
        segments: список (начало, конец) в секундах. Подходят и отрезки
            VAD, и границы слов от ASR.

    Returns:
        Список Prosody — по одному на каждый отрезок, в том же порядке.
    """
    import librosa

    if not segments:
        return []

    # --- считаем один раз на всю запись ---

    rms = librosa.feature.rms(
        y=y, frame_length=HOP_LENGTH * 2, hop_length=HOP_LENGTH
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    f0, _, _ = librosa.pyin(
        y, sr=sr, fmin=F0_MIN, fmax=F0_MAX,
        frame_length=HOP_LENGTH * 8, hop_length=HOP_LENGTH,
    )

    voiced_frames = ~np.isnan(f0)

    if voiced_frames.any():
        median_f0 = float(np.median(f0[voiced_frames]))
        semitones = np.zeros(len(f0), dtype=np.float32)
        semitones[voiced_frames] = 12.0 * np.log2(f0[voiced_frames] / median_f0)
    else:
        semitones = np.zeros(len(f0), dtype=np.float32)

    n_frames = min(len(rms_db), len(semitones))
    frame_sec = HOP_LENGTH / sr

    # --- собираем сырые значения по отрезкам ---

    raw_loud, raw_pitch, has_pitch = [], [], []

    for start, end in segments:
        a = int(start / frame_sec)
        b = max(a + 1, int(end / frame_sec))
        a = min(max(a, 0), n_frames - 1)
        b = min(b, n_frames)

        window_db = rms_db[a:b]
        raw_loud.append(float(np.mean(window_db)) if len(window_db) else -80.0)

        window_voiced = voiced_frames[a:b]
        if window_voiced.any():
            raw_pitch.append(float(np.mean(semitones[a:b][window_voiced])))
            has_pitch.append(True)
        else:
            # Глухой отрезок: высоты нет. Ставим 0 полутонов и отдельно
            # помечаем флагом — иначе «нет высоты» станет неотличимо от
            # «очень низкая высота», и такое слово покрасится как басовое.
            raw_pitch.append(0.0)
            has_pitch.append(False)

    # --- нормализуем общей шкалой на всю запись ---
    #
    # Именно общей: если нормализовать каждый отрезок отдельно, в каждом
    # найдётся своё «самое громкое слово», и разница между отрезками
    # исчезнет.

    raw_loud = np.array(raw_loud, dtype=np.float32)
    raw_pitch = np.array(raw_pitch, dtype=np.float32)
    has_pitch = np.array(has_pitch, dtype=bool)

    loud_norm = _robust_scale(raw_loud, np.ones(len(raw_loud), dtype=bool))
    pitch_norm = _robust_scale(raw_pitch, has_pitch)

    return [
        Prosody(
            loud=float(loud_norm[i]),
            pitch=float(pitch_norm[i]),
            voiced=bool(has_pitch[i]),
            loud_db=float(raw_loud[i]),
            pitch_semitones=float(raw_pitch[i]),
        )
        for i in range(len(segments))
    ]


if __name__ == "__main__":
    import argparse
    import sys
    import time

    import librosa

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Просодия по отрезкам VAD")
    parser.add_argument("audio")
    args = parser.parse_args()

    import vad

    y, sr = librosa.load(args.audio, sr=16000)
    segments = [(s["start"], s["end"]) for s in vad.tmstamps(y)]

    started = time.perf_counter()
    result = analyze(y, sr, segments)
    elapsed = time.perf_counter() - started

    print(f"\n{len(y) / sr:.2f} с аудио, {len(segments)} отрезков, "
          f"просодия за {elapsed:.2f} с\n")

    print(f"{'отрезок':>18s} {'loud':>6s} {'pitch':>6s} {'дБ':>7s} {'полутон':>8s}")
    print("-" * 50)
    for (start, end), p in zip(segments, result):
        mark = "" if p.voiced else "  (глухой)"
        print(f"{start:7.2f}-{end:7.2f} {p.loud:6.2f} {p.pitch:6.2f} "
              f"{p.loud_db:7.1f} {p.pitch_semitones:8.2f}{mark}")
