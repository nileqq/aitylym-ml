"""Сравнение лёгких VAD на казахском аудио.

Задача, под которую меряем
--------------------------
В AItylym у VAD одна работа: найти паузы, по которым резать поток на фразы.
Не «отделить речь от тишины вообще», а именно попасть в границы. Поэтому и
метрики здесь про границы, а не общая покадровая точность.

Откуда разметка
---------------
Готовых VAD-корпусов с покадровой разметкой для казахского нет, а мерить на
английских — мерить не тот домен. Поэтому тест строится из собственных
клипов: они склеиваются паузами известной длины, и правильные границы
известны по построению.

Ограничение честно назовём: внутри клипов могут быть свои микропаузы,
которые засчитаются как ложные срабатывания. Чтобы это не искажало
сравнение, ложные границы считаются отдельной метрикой, а не смешиваются
с пропущенными.

Запуск:
    python app/reports/vad_benchmark.py
    python app/reports/vad_benchmark.py --clips 60 --repeats 3
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000

# Паузы такой длины вставляем между клипами. Диапазон выбран под задачу:
# в pipeline.py фразы режутся по паузе от 0.45 с.
GAP_MIN, GAP_MAX = 0.35, 1.10

# Насколько близко к истинной паузе должна лечь найденная граница,
# чтобы засчитать попадание.
TOLERANCE_SEC = 0.20


# ----------------------------------------------------------------------------
# Тестовый материал
# ----------------------------------------------------------------------------


def trim_silence(wav: np.ndarray, top_db: float = 45.0) -> np.ndarray:
    """Убирает тишину по краям клипа.

    Нужно, чтобы вставленная пауза не сливалась с собственным хвостом клипа
    и истинная граница оставалась точной. Порог мягкий: режем только явную
    тишину, речь не трогаем.
    """
    import librosa

    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed if len(trimmed) > SAMPLE_RATE * 0.2 else wav


def build_test_audio(paths: list, rng) -> tuple:
    """Склеивает клипы паузами известной длины.

    Возвращает (аудио, список пауз [(начало, конец)]).
    """
    import soundfile as sf
    import librosa

    pieces, gaps = [], []
    position = 0.0

    for i, path in enumerate(paths):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

        wav = trim_silence(np.asarray(wav, dtype=np.float32))
        pieces.append(wav)
        position += len(wav) / SAMPLE_RATE

        if i < len(paths) - 1:
            gap = float(rng.uniform(GAP_MIN, GAP_MAX))
            # Не абсолютный ноль: в реальной записи пауза — это тихий фон,
            # а на цифровой тишине VAD ведут себя нереалистично хорошо.
            silence = rng.normal(0, 1e-4, int(gap * SAMPLE_RATE)).astype("float32")
            pieces.append(silence)
            gaps.append((position, position + gap))
            position += gap

    return np.concatenate(pieces), gaps


def add_noise(wav: np.ndarray, snr_db, rng) -> np.ndarray:
    if snr_db is None:
        return wav
    signal = np.sqrt(np.mean(wav ** 2) + 1e-12)
    noise = rng.standard_normal(len(wav)).astype("float32")
    noise *= (signal / (10 ** (snr_db / 20.0))) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return (wav + noise).astype("float32")


# ----------------------------------------------------------------------------
# Обёртки над VAD. Каждая возвращает список отрезков РЕЧИ в секундах.
# ----------------------------------------------------------------------------


class EnergyVAD:
    """librosa.effects.split — то, что сейчас стоит в omniasr_words.py."""

    name = "energy (librosa)"
    size_mb = 0.0
    kind = "порог по энергии"

    def __init__(self, top_db: float = 35.0):
        self.top_db = top_db

    def __call__(self, wav):
        import librosa

        intervals = librosa.effects.split(wav, top_db=self.top_db)
        return [(a / SAMPLE_RATE, b / SAMPLE_RATE) for a, b in intervals]


def _import_webrtcvad():
    """webrtcvad 2.0.10 импортирует pkg_resources только чтобы узнать свою
    версию, а setuptools >= 81 его больше не поставляет. Подставляем
    заглушку вместо того, чтобы править чужой файл или откатывать
    setuptools."""
    import sys
    import types

    if "pkg_resources" not in sys.modules:
        import importlib.metadata as metadata

        stub = types.ModuleType("pkg_resources")
        stub.get_distribution = lambda name: types.SimpleNamespace(
            version=metadata.version(name)
        )
        sys.modules["pkg_resources"] = stub

    import webrtcvad

    return webrtcvad


class WebRTCVAD:
    """Классический GMM из WebRTC. Без нейросети, на C."""

    name = "WebRTC VAD"
    size_mb = 0.0
    kind = "GMM, без обучения"

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30):
        webrtcvad = _import_webrtcvad()

        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.aggressiveness = aggressiveness

    def __call__(self, wav):
        pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()
        step = int(SAMPLE_RATE * self.frame_ms / 1000) * 2

        flags = []
        for offset in range(0, len(pcm) - step + 1, step):
            flags.append(self.vad.is_speech(pcm[offset:offset + step], SAMPLE_RATE))

        return _flags_to_segments(flags, self.frame_ms / 1000)


class SileroVAD:
    """Silero VAD — маленькая нейросеть, де-факто стандарт."""

    name = "Silero VAD"
    kind = "нейросеть"

    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.threshold = threshold

        import silero_vad
        weights = Path(silero_vad.__file__).parent / "data"
        total = sum(f.stat().st_size for f in weights.rglob("*") if f.is_file())
        self.size_mb = total / 1e6

    def __call__(self, wav):
        import torch
        from silero_vad import get_speech_timestamps

        stamps = get_speech_timestamps(
            torch.from_numpy(wav),
            self.model,
            sampling_rate=SAMPLE_RATE,
            threshold=self.threshold,
            return_seconds=True,
        )
        return [(s["start"], s["end"]) for s in stamps]


def _flags_to_segments(flags: list, frame_sec: float) -> list:
    segments, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start * frame_sec, i * frame_sec))
            start = None
    if start is not None:
        segments.append((start * frame_sec, len(flags) * frame_sec))
    return segments


# ----------------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------------


def score(speech: list, gaps: list, total_sec: float) -> dict:
    """Считает, насколько найденные паузы совпали с настоящими."""
    # Паузы = всё, что между отрезками речи
    speech = sorted(speech)
    silences = []
    cursor = 0.0
    for a, b in speech:
        if a - cursor > 1e-6:
            silences.append((cursor, a))
        cursor = max(cursor, b)
    if total_sec - cursor > 1e-6:
        silences.append((cursor, total_sec))

    found, errors = 0, []
    for gap_start, gap_end in gaps:
        centre = (gap_start + gap_end) / 2

        matched = None
        for sil_start, sil_end in silences:
            if sil_start <= centre <= sil_end:
                matched = (sil_start, sil_end)
                break

        if matched is None:
            continue

        # Ошибка границы: насколько начало найденной паузы разошлось с
        # настоящим. Мерить расстояние до центра бессмысленно — оно нулевое
        # по самому условию попадания.
        offset = abs(matched[0] - gap_start)
        if offset <= TOLERANCE_SEC:
            found += 1
            errors.append(offset)

    # Ложные паузы: детектированные тишины, не совпавшие ни с одной настоящей
    false_gaps = 0
    for sil_start, sil_end in silences:
        if sil_end - sil_start < 0.1:
            continue
        centre = (sil_start + sil_end) / 2
        if not any(g0 - TOLERANCE_SEC <= centre <= g1 + TOLERANCE_SEC
                   for g0, g1 in gaps):
            false_gaps += 1

    speech_sec = sum(b - a for a, b in speech)

    return {
        "recall": found / len(gaps) if gaps else 0.0,
        "false_per_min": false_gaps / (total_sec / 60),
        "boundary_err_ms": float(np.mean(errors) * 1000) if errors else float("nan"),
        "speech_ratio": speech_sec / total_sec,
    }


# ----------------------------------------------------------------------------
# Прогон
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=40, help="клипов в одном тесте")
    parser.add_argument("--repeats", type=int, default=3, help="разных склеек")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    files = sorted(glob.glob(str(root / "data" / "audio" / "*.wav")))
    if not files:
        raise SystemExit("не найдено data/audio/*.wav")

    rng = np.random.default_rng(0)

    vads = [EnergyVAD(), WebRTCVAD(aggressiveness=2), WebRTCVAD(aggressiveness=3),
            SileroVAD()]
    vads[2].name = "WebRTC VAD (agg=3)"
    vads[1].name = "WebRTC VAD (agg=2)"

    conditions = [None, 20, 10, 5]
    results = {}

    for repeat in range(args.repeats):
        picks = list(rng.choice(files, size=args.clips, replace=False))
        clean, gaps = build_test_audio(picks, rng)
        total = len(clean) / SAMPLE_RATE
        print(f"[{repeat + 1}/{args.repeats}] тест: {total:.1f} с, "
              f"{len(gaps)} пауз", flush=True)

        for snr in conditions:
            wav = add_noise(clean, snr, np.random.default_rng(42))
            for vad in vads:
                started = time.perf_counter()
                speech = vad(wav)
                elapsed = time.perf_counter() - started

                metrics = score(speech, gaps, total)
                metrics["rtf"] = elapsed / total

                key = (vad.name, "чистое" if snr is None else f"SNR {snr}")
                results.setdefault(key, []).append(metrics)

    # усредняем по повторам
    table = {}
    for (name, cond), runs in results.items():
        table[f"{name}|{cond}"] = {
            k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]
        }

    print()
    header = (f"{'VAD':22s} {'условие':10s} {'recall':>8s} {'лишних/мин':>11s} "
              f"{'ошибка мс':>10s} {'речь %':>8s} {'RTF':>9s}")
    print(header)
    print("-" * len(header))
    for key, m in table.items():
        name, cond = key.split("|")
        print(f"{name:22s} {cond:10s} {m['recall'] * 100:7.1f}% "
              f"{m['false_per_min']:11.1f} {m['boundary_err_ms']:10.0f} "
              f"{m['speech_ratio'] * 100:7.1f}% {m['rtf']:9.5f}")

    print()
    for vad in vads:
        print(f"  {vad.name:22s} {vad.kind:20s} {vad.size_mb:6.2f} МБ")

    if args.json:
        Path(args.json).write_text(
            json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nсохранено: {args.json}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
