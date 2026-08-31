import json
import librosa
import numpy as np
import whisper


def normalize(values):
    val_array = np.array(values)
    min_val = np.min(val_array)
    max_val = np.max(val_array)
    if max_val - min_val == 0:
        return np.zeros_like(val_array)
    return (val_array - min_val) / (max_val - min_val)


def generate_kinetic_json(audio_path, output_json="kinetic_subtitles.json"):
    print("1. Извлечение акустических параметров (DSP)...")
    y, sr = librosa.load(audio_path, sr=None)
    frame_ms = 20
    hop_length = int(sr * (frame_ms / 1000.0))

    # Извлечение RMS (Громкость)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_norm = normalize(rms)

    # Извлечение Pitch (F0)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sr,
        hop_length=hop_length,
    )

    # Нормализуем только вокализованные кадры (где F0 != NaN)
    valid_f0_mask = ~np.isnan(f0)
    f0_norm = np.full_like(f0, np.nan)
    if np.any(valid_f0_mask):
        f0_min = np.min(f0[valid_f0_mask])
        f0_max = np.max(f0[valid_f0_mask])
        if f0_max - f0_min != 0:
            f0_norm[valid_f0_mask] = (f0[valid_f0_mask] - f0_min) / (
                f0_max - f0_min
            )
        else:
            f0_norm[valid_f0_mask] = 0.5

    times = librosa.frames_to_time(
        range(len(rms)), sr=sr, hop_length=hop_length
    )

    print("2. Распознавание казахской речи (Whisper ASR)...")
    # Для казахского языка рекомендуется модель не ниже 'small' или 'medium'
    model = whisper.load_model("small")

    # Передаем параметры языка и контекстную подсказку кириллицы
    result = model.transcribe(
        audio_path,
        language="kk",
        initial_prompt="Бұл қазақ тіліндегі аудиожазба.",
        word_timestamps=True,
    )

    words_data = []

    # 3. Агрегация DSP-параметров по временным интервалам слов
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word_str = word_info["word"].strip()
            w_start = word_info["start"]
            w_end = word_info["end"]
            duration = max(w_end - w_start, 0.05)

            indices = np.where((times >= w_start) & (times <= w_end))[0]

            if len(indices) > 0:
                avg_loudness = float(np.mean(rms_norm[indices]))

                # Фильтруем NaN для точного расчета высоты тона
                word_f0 = f0_norm[indices]
                voiced_f0 = word_f0[~np.isnan(word_f0)]

                if len(voiced_f0) > 0:
                    avg_pitch = float(np.mean(voiced_f0))
                else:
                    avg_pitch = (
                        0.0  # Глухой согласный звук или фоновый шум/пауза
                    )
            else:
                avg_loudness = 0.5
                avg_pitch = 0.5

            char_per_sec = len(word_str) / duration
            speed_factor = float(np.clip(char_per_sec / 15.0, 0.5, 2.0))

            words_data.append({
                "word": word_str,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "duration": round(duration, 3),
                "loudness": round(avg_loudness, 3),
                "pitch": round(avg_pitch, 3),
                "speed_factor": round(speed_factor, 3),
            })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(words_data, f, ensure_ascii=False, indent=2)

    print(f"Готово! Кинетический JSON сохранен в {output_json}")
    return words_data


if __name__ == "__main__":
    generate_kinetic_json("tale.wav")