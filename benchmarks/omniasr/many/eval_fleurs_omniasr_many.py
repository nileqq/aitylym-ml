import csv
import json
import math
import os
import re
import time
from pathlib import Path

import torch
from datasets import Audio, load_dataset
from jiwer import (
    cer,
    mer,
    process_words,
    wer,
    wil,
    wip,
)
from tqdm import tqdm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


# ============================================================
# CONFIG
# ============================================================

MODEL = "omniASR_CTC_300M_v2"

# Логический большой блок.
# Нужен только для организации прогресса.
CHUNK_SIZE = 50

# Реальное количество аудио, одновременно идущих в GPU.
# Для RTX 4060 8 GB оставляем 8.
MODEL_BATCH_SIZE = 8

# OmniASR имеет hard limit 40 sec.
MAX_AUDIO_SECONDS = 39.9

FLEURS_TEST_FILE = (
    "hf://datasets/google/fleurs/"
    "parquet-data/kk_kz/"
    "test-00000-of-00001.parquet"
)

OUTPUT_DIR = Path(__file__).resolve().parent

OUTPUT_CSV = (
    OUTPUT_DIR
    / "fleurs_omniasr_300m_full_results.csv"
)

CHECKPOINT_CSV = (
    OUTPUT_DIR
    / "fleurs_omniasr_300m_checkpoint.csv"
)

STATE_FILE = (
    OUTPUT_DIR
    / "fleurs_omniasr_300m_state.json"
)

SKIPPED_CSV = (
    OUTPUT_DIR
    / "fleurs_omniasr_300m_skipped.csv"
)


CHECKPOINT_FIELDS = [
    "recording_idx",
    "source_id",
    "duration_sec",
    "reference",
    "hypothesis",
    "wer",
    "cer",
    "exact_match",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Одинаковая normalization для reference и hypothesis.

    - lowercase
    - punctuation -> space
    - multiple spaces -> one

    Казахские Unicode-буквы сохраняются.
    """

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = " ".join(text.split())

    return text


# ============================================================
# HELPERS
# ============================================================

def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)

    if n % 2 == 1:
        return values[n // 2]

    return (
        values[n // 2 - 1]
        + values[n // 2]
    ) / 2


def save_state(
    processed_samples: int,
    cumulative_inference_time: float,
):
    """
    Сохраняем state атомарно:
    сначала .tmp, потом replace.
    """

    state = {
        "model": MODEL,
        "processed_samples": processed_samples,
        "cumulative_inference_time":
            cumulative_inference_time,
    }

    tmp_file = STATE_FILE.with_suffix(".tmp")

    with tmp_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())

    tmp_file.replace(STATE_FILE)


def append_checkpoint(rows):
    """
    Сразу физически дописываем batch на диск.
    """

    with CHECKPOINT_CSV.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CHECKPOINT_FIELDS,
        )

        writer.writerows(rows)

        f.flush()
        os.fsync(f.fileno())


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("OMNIASR — FLEURS KAZAKH FULL MULTI-RECORDING BENCHMARK")
print("=" * 80)


# ============================================================
# 1. LOAD FLEURS
# ============================================================

print()
print("[1/6] Loading ONLY FLEURS kk_kz TEST...")

dataset = load_dataset(
    "parquet",
    data_files={
        "test": FLEURS_TEST_FILE,
    },
    split="test",
)

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False),
)

print(
    f"Raw recordings in test split: "
    f"{len(dataset)}"
)


# ============================================================
# 2. USE ALL RECORDINGS
# ============================================================

print()
print("[2/6] Using ALL recordings...")

# ВАЖНО:
# sample["id"] во FLEURS НЕ уникален для recording.
#
# Один и тот же текст может иметь несколько аудиозаписей.
# Поэтому ничего через seen не удаляем.
#
# Уникальным ключом будет recording_idx = индекс строки dataset.

print(
    f"Total recordings: {len(dataset)}"
)


# ============================================================
# 3. PREPARE DATA
# ============================================================

print()
print("[3/6] Preparing audio + references...")


recording_indices = []
source_ids = []
audio_inputs = []
references = []
durations = []

too_long_rows = []

total_audio_seconds = 0.0


for recording_idx in tqdm(
    range(len(dataset)),
    desc="Preparing",
    unit="recording",
):

    sample = dataset[recording_idx]

    source_id = sample["id"]

    num_samples = sample.get(
        "num_samples"
    )

    if num_samples is None:
        raise RuntimeError(
            f"No num_samples for "
            f"recording_idx={recording_idx}, "
            f"source_id={source_id}"
        )

    duration = (
        num_samples / 16000.0
    )

    # --------------------------------------------------------
    # OmniASR hard limit
    # --------------------------------------------------------

    if duration >= MAX_AUDIO_SECONDS:

        too_long_rows.append({
            "recording_idx": recording_idx,
            "source_id": source_id,
            "duration_sec": duration,
            "reason": "audio >= 40 sec",
        })

        continue

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    audio = sample["audio"]

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes is not None:
        audio_input = audio_bytes

    elif audio_path is not None:
        audio_input = audio_path

    else:
        raise RuntimeError(
            f"No audio bytes/path for "
            f"recording_idx={recording_idx}, "
            f"source_id={source_id}"
        )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = normalize_text(
        sample["transcription"]
    )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    recording_indices.append(
        recording_idx
    )

    source_ids.append(
        source_id
    )

    audio_inputs.append(
        audio_input
    )

    references.append(
        reference
    )

    durations.append(
        duration
    )

    total_audio_seconds += duration


# ============================================================
# SAVE SKIPPED
# ============================================================

with SKIPPED_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "recording_idx",
            "source_id",
            "duration_sec",
            "reason",
        ],
    )

    writer.writeheader()
    writer.writerows(
        too_long_rows
    )


print()

print(
    f"Raw recordings      : "
    f"{len(dataset)}"
)

print(
    f"Usable recordings   : "
    f"{len(recording_indices)}"
)

print(
    f"Skipped >= 40 sec   : "
    f"{len(too_long_rows)}"
)

if too_long_rows:

    print()
    print("Skipped long recordings:")

    for row in too_long_rows:

        print(
            f"  recording_idx="
            f"{row['recording_idx']} | "
            f"source_id="
            f"{row['source_id']} | "
            f"{row['duration_sec']:.2f} sec"
        )


print()

print(
    f"Total audio duration: "
    f"{total_audio_seconds:.2f} sec"
)

print(
    f"                    : "
    f"{total_audio_seconds / 60:.2f} min"
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

print()
print("[4/6] Loading OmniASR...")
print(f"Model: {MODEL}")


model_load_start = time.perf_counter()


pipeline = ASRInferencePipeline(
    model_card=MODEL,
)


model_load_time = (
    time.perf_counter()
    - model_load_start
)


print(
    f"Model loaded in "
    f"{model_load_time:.2f} sec"
)


# ============================================================
# GPU
# ============================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    print(
        "WARNING: CUDA is NOT available!"
    )


# ============================================================
# 5. RESTORE CHECKPOINT
# ============================================================

print()
print("[5/6] Running full inference...")

print(
    f"Chunk size       : "
    f"{CHUNK_SIZE}"
)

print(
    f"Model batch size : "
    f"{MODEL_BATCH_SIZE}"
)


processed_recordings = set()

cumulative_inference_time = 0.0


# ------------------------------------------------------------
# Existing checkpoint
# ------------------------------------------------------------

if CHECKPOINT_CSV.exists():

    print()
    print(
        f"Found checkpoint:\n"
        f"{CHECKPOINT_CSV}"
    )

    with CHECKPOINT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        # Защита от старого checkpoint,
        # где был только id.
        if (
            reader.fieldnames is None
            or "recording_idx"
            not in reader.fieldnames
        ):
            raise RuntimeError(
                "\nOld checkpoint format detected.\n"
                "Delete this file before running:\n"
                f"{CHECKPOINT_CSV}\n"
            )

        for row in reader:

            processed_recordings.add(
                int(
                    row["recording_idx"]
                )
            )


    print(
        f"Restored predictions : "
        f"{len(processed_recordings)}"
    )


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

if STATE_FILE.exists():

    with STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        state = json.load(f)


    state_model = state.get(
        "model"
    )

    if (
        state_model is not None
        and state_model != MODEL
    ):

        raise RuntimeError(
            "Checkpoint model mismatch:\n"
            f"checkpoint = {state_model}\n"
            f"current    = {MODEL}"
        )


    cumulative_inference_time = (
        state.get(
            "cumulative_inference_time",
            0.0,
        )
    )


    print(
        f"Restored inference time: "
        f"{cumulative_inference_time:.2f} sec"
    )


# ============================================================
# CREATE CHECKPOINT HEADER
# ============================================================

if not CHECKPOINT_CSV.exists():

    with CHECKPOINT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=CHECKPOINT_FIELDS,
        )

        writer.writeheader()


# ============================================================
# REMAINING RECORDINGS
# ============================================================

remaining_positions = [
    position
    for position, recording_idx
    in enumerate(recording_indices)
    if recording_idx
    not in processed_recordings
]


print()

print(
    f"Already processed : "
    f"{len(processed_recordings)}"
)

print(
    f"Remaining         : "
    f"{len(remaining_positions)}"
)


# ============================================================
# CHUNKS
# ============================================================

num_chunks = math.ceil(
    len(remaining_positions)
    / CHUNK_SIZE
)


# ============================================================
# INFERENCE
# ============================================================

for chunk_number, chunk_offset in enumerate(

    tqdm(
        range(
            0,
            len(remaining_positions),
            CHUNK_SIZE,
        ),
        total=num_chunks,
        desc="OmniASR chunks",
        unit="chunk",
    ),

    start=1,
):

    chunk_positions = (
        remaining_positions[
            chunk_offset:
            chunk_offset + CHUNK_SIZE
        ]
    )


    # ========================================================
    # BATCHES INSIDE THIS CHUNK
    # ========================================================

    for batch_offset in range(
        0,
        len(chunk_positions),
        MODEL_BATCH_SIZE,
    ):

        batch_positions = (
            chunk_positions[
                batch_offset:
                batch_offset
                + MODEL_BATCH_SIZE
            ]
        )


        batch_recording_indices = [
            recording_indices[position]
            for position
            in batch_positions
        ]


        batch_source_ids = [
            source_ids[position]
            for position
            in batch_positions
        ]


        batch_audio = [
            audio_inputs[position]
            for position
            in batch_positions
        ]


        batch_references = [
            references[position]
            for position
            in batch_positions
        ]


        batch_durations = [
            durations[position]
            for position
            in batch_positions
        ]


        # ====================================================
        # RUN GPU BATCH
        # ====================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()


        batch_start = (
            time.perf_counter()
        )


        batch_predictions = (
            pipeline.transcribe(
                batch_audio,

                # Мы уже сами сформировали
                # batch <= MODEL_BATCH_SIZE.
                batch_size=len(
                    batch_audio
                ),
            )
        )


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        batch_elapsed = (
            time.perf_counter()
            - batch_start
        )


        cumulative_inference_time += (
            batch_elapsed
        )


        # ====================================================
        # NORMALIZE OUTPUT
        # ====================================================

        batch_hypotheses = [
            normalize_text(prediction)
            for prediction
            in batch_predictions
        ]


        if (
            len(batch_hypotheses)
            != len(batch_references)
        ):

            raise RuntimeError(
                "Batch prediction count mismatch: "
                f"{len(batch_hypotheses)} vs "
                f"{len(batch_references)}"
            )


        # ====================================================
        # BUILD CHECKPOINT ROWS
        # ====================================================

        batch_rows = []


        for (
            recording_idx,
            source_id,
            duration,
            reference,
            hypothesis,
        ) in zip(
            batch_recording_indices,
            batch_source_ids,
            batch_durations,
            batch_references,
            batch_hypotheses,
        ):

            sample_wer = wer(
                reference,
                hypothesis,
            )

            sample_cer = cer(
                reference,
                hypothesis,
            )

            exact_match = int(
                reference
                == hypothesis
            )


            batch_rows.append({
                "recording_idx":
                    recording_idx,

                "source_id":
                    source_id,

                "duration_sec":
                    duration,

                "reference":
                    reference,

                "hypothesis":
                    hypothesis,

                "wer":
                    sample_wer,

                "cer":
                    sample_cer,

                "exact_match":
                    exact_match,
            })


        # ====================================================
        # SAVE IMMEDIATELY AFTER EACH GPU BATCH
        # ====================================================

        append_checkpoint(
            batch_rows
        )


        for recording_idx in (
            batch_recording_indices
        ):

            processed_recordings.add(
                recording_idx
            )


        save_state(
            processed_samples=len(
                processed_recordings
            ),
            cumulative_inference_time=(
                cumulative_inference_time
            ),
        )


    # Chunk completed
    tqdm.write(
        f"Chunk {chunk_number}/{num_chunks} "
        f"complete | "
        f"processed="
        f"{len(processed_recordings)} | "
        f"saved ✓"
    )


# ============================================================
# RELOAD CHECKPOINT
# ============================================================

checkpoint_rows = []


with CHECKPOINT_CSV.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        checkpoint_rows.append(row)


# ============================================================
# CHECK FOR ACCIDENTAL DUPLICATES
# ============================================================

checkpoint_recording_indices = [
    int(row["recording_idx"])
    for row in checkpoint_rows
]


if (
    len(checkpoint_recording_indices)
    != len(
        set(
            checkpoint_recording_indices
        )
    )
):

    raise RuntimeError(
        "Duplicate recording_idx found "
        "inside checkpoint!"
    )


# ============================================================
# SORT INTO ORIGINAL DATASET ORDER
# ============================================================

checkpoint_rows.sort(
    key=lambda row:
        int(row["recording_idx"])
)


# ============================================================
# COMPLETENESS CHECK
# ============================================================

if (
    len(checkpoint_rows)
    != len(recording_indices)
):

    raise RuntimeError(
        "Benchmark incomplete:\n"
        f"predictions = "
        f"{len(checkpoint_rows)}\n"
        f"expected    = "
        f"{len(recording_indices)}"
    )


# ============================================================
# FINAL ARRAYS
# ============================================================

final_references = [
    row["reference"]
    for row in checkpoint_rows
]


final_hypotheses = [
    row["hypothesis"]
    for row in checkpoint_rows
]


final_durations = [
    float(
        row["duration_sec"]
    )
    for row in checkpoint_rows
]


# ============================================================
# 6. GLOBAL METRICS
# ============================================================

print()
print("[6/6] Calculating metrics...")


overall_wer = wer(
    final_references,
    final_hypotheses,
)


overall_cer = cer(
    final_references,
    final_hypotheses,
)


overall_mer = mer(
    final_references,
    final_hypotheses,
)


overall_wil = wil(
    final_references,
    final_hypotheses,
)


overall_wip = wip(
    final_references,
    final_hypotheses,
)


word_stats = process_words(
    final_references,
    final_hypotheses,
)


# ============================================================
# PER-RECORDING METRICS
# ============================================================

sample_wers = []
sample_cers = []

exact_matches = 0


for row in tqdm(
    checkpoint_rows,
    desc="Per-recording metrics",
    unit="recording",
):

    sample_wer = float(
        row["wer"]
    )

    sample_cer = float(
        row["cer"]
    )

    exact_match = int(
        row["exact_match"]
    )

    sample_wers.append(
        sample_wer
    )

    sample_cers.append(
        sample_cer
    )

    exact_matches += exact_match


exact_match_rate = (
    exact_matches
    / len(checkpoint_rows)
)


# ============================================================
# SPEED
# ============================================================

final_audio_seconds = sum(
    final_durations
)


if final_audio_seconds > 0:

    rtf = (
        cumulative_inference_time
        / final_audio_seconds
    )

    realtime_speed = (
        1.0 / rtf
    )

else:

    rtf = None
    realtime_speed = None


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    peak_vram_bytes = (
        torch.cuda.max_memory_allocated()
    )

    peak_vram_gb = (
        peak_vram_bytes
        / (1024 ** 3)
    )

else:

    peak_vram_gb = None


# ============================================================
# FINAL CSV
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=CHECKPOINT_FIELDS,
    )

    writer.writeheader()

    writer.writerows(
        checkpoint_rows
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL RESULTS")
print("=" * 80)


print()
print("DATASET")
print("-" * 80)

print(
    "Dataset              : "
    "FLEURS kk_kz TEST"
)

print(
    "Evaluation mode      : "
    "ALL recordings"
)

print(
    f"Raw recordings       : "
    f"{len(dataset)}"
)

print(
    f"Evaluated recordings : "
    f"{len(checkpoint_rows)}"
)

print(
    f"Too long excluded    : "
    f"{len(too_long_rows)}"
)

print(
    f"Audio duration       : "
    f"{final_audio_seconds / 60:.2f} min"
)


print()
print("MODEL")
print("-" * 80)

print(
    f"Model                : "
    f"{MODEL}"
)

print(
    f"Chunk size           : "
    f"{CHUNK_SIZE}"
)

print(
    f"GPU batch size       : "
    f"{MODEL_BATCH_SIZE}"
)


print()
print("ACCURACY")
print("-" * 80)

print(
    f"WER ↓                : "
    f"{overall_wer:.4f} "
    f"({overall_wer * 100:.2f}%)"
)

print(
    f"CER ↓                : "
    f"{overall_cer:.4f} "
    f"({overall_cer * 100:.2f}%)"
)

print(
    f"MER ↓                : "
    f"{overall_mer:.4f} "
    f"({overall_mer * 100:.2f}%)"
)

print(
    f"WIL ↓                : "
    f"{overall_wil:.4f} "
    f"({overall_wil * 100:.2f}%)"
)

print(
    f"WIP ↑                : "
    f"{overall_wip:.4f} "
    f"({overall_wip * 100:.2f}%)"
)

print(
    f"Exact sentences ↑    : "
    f"{exact_matches}/"
    f"{len(checkpoint_rows)} "
    f"({exact_match_rate * 100:.2f}%)"
)


print()
print("WORD ERRORS")
print("-" * 80)

print(
    f"Hits                 : "
    f"{word_stats.hits}"
)

print(
    f"Substitutions        : "
    f"{word_stats.substitutions}"
)

print(
    f"Deletions            : "
    f"{word_stats.deletions}"
)

print(
    f"Insertions           : "
    f"{word_stats.insertions}"
)


print()
print("PER-RECORDING")
print("-" * 80)

print(
    f"Mean WER             : "
    f"{mean(sample_wers) * 100:.2f}%"
)

print(
    f"Median WER           : "
    f"{median(sample_wers) * 100:.2f}%"
)

print(
    f"Mean CER             : "
    f"{mean(sample_cers) * 100:.2f}%"
)

print(
    f"Median CER           : "
    f"{median(sample_cers) * 100:.2f}%"
)


print()
print("PERFORMANCE")
print("-" * 80)

print(
    f"Model load time      : "
    f"{model_load_time:.2f} sec"
)

print(
    f"Inference time       : "
    f"{cumulative_inference_time:.2f} sec"
)

if rtf is not None:

    print(
        f"RTF ↓                : "
        f"{rtf:.4f}"
    )

    print(
        f"Realtime speed ↑     : "
        f"{realtime_speed:.2f}x"
    )


if peak_vram_gb is not None:

    print(
        f"Peak VRAM            : "
        f"{peak_vram_gb:.2f} GB"
    )


print()
print("OUTPUT")
print("-" * 80)

print(
    f"Final CSV:\n"
    f"{OUTPUT_CSV}"
)

print()

print(
    f"Checkpoint:\n"
    f"{CHECKPOINT_CSV}"
)

print()

print(
    f"Skipped recordings:\n"
    f"{SKIPPED_CSV}"
)


print()
print("=" * 80)
print("DONE")
print("=" * 80)