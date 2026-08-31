import csv
import math
import re
import time
from pathlib import Path
import json, os

import torch
from datasets import Audio, load_dataset
from jiwer import (
    wer,
    cer,
    mer,
    wil,
    wip,
    process_words,
)
from tqdm import tqdm

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


# ============================================================
# CONFIG
# ============================================================

MODEL = "omniASR_CTC_300M_v2"

# Сколько samples обрабатываем как один логический chunk
CHUNK_SIZE = 50

MAX_AUDIO_SECONDS = 39.9

# Реальный batch внутри модели.
# Для RTX 4060 8 GB начинаем с 8.
MODEL_BATCH_SIZE = 8

FLEURS_TEST_FILE = (
    "hf://datasets/google/fleurs/"
    "parquet-data/kk_kz/"
    "test-00000-of-00001.parquet"
)

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "fleurs_omniasr_300m_full_results.csv"

CHECKPOINT_CSV = (
    OUTPUT_DIR / "fleurs_omniasr_300m_checkpoint.csv"
)

STATE_FILE = (
    OUTPUT_DIR / "fleurs_omniasr_300m_state.json"
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Одинаковая normalization для reference и hypothesis.

    - lowercase
    - punctuation -> space
    - multiple spaces -> one
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
# HELPER
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


# ============================================================
# LOAD FLEURS
# ============================================================

print()
print("=" * 80)
print("OMNIASR — FLEURS KAZAKH FULL BENCHMARK")
print("=" * 80)

print("\n[1/6] Loading ONLY FLEURS kk_kz TEST...")

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

print(f"Raw samples in test split: {len(dataset)}")


# ============================================================
# UNIQUE IDs ONLY
# ============================================================

print("\n[2/6] Removing duplicate IDs...")

seen = set()

samples = []
duplicate_count = 0


for sample in tqdm(
    dataset,
    total=len(dataset),
    desc="Deduplicating",
    unit="sample",
):
    sample_id = sample["id"]

    if sample_id in seen:
        duplicate_count += 1
        continue

    seen.add(sample_id)
    samples.append(sample)


print()
print(f"Unique samples : {len(samples)}")
print(f"Duplicates     : {duplicate_count}")


# ============================================================
# PREPARE DATA
# ============================================================
print("\n[3/6] Preparing audio + references...")

audio_inputs = []
references = []
sample_ids = []
durations = []

total_audio_seconds = 0.0

too_long_ids = []


for sample in tqdm(
    samples,
    desc="Preparing",
    unit="sample",
):
    sample_id = sample["id"]

    num_samples = sample.get("num_samples")

    if num_samples is None:
        raise RuntimeError(
            f"No num_samples for ID={sample_id}"
        )

    duration = num_samples / 16000.0

    # OmniASR has a hard 40-second limit.
    if duration >= MAX_AUDIO_SECONDS:
        too_long_ids.append(
            (sample_id, duration)
        )
        continue

    audio = sample["audio"]

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes is not None:
        audio_input = audio_bytes

    elif audio_path is not None:
        audio_input = audio_path

    else:
        raise RuntimeError(
            f"No audio bytes/path for ID={sample_id}"
        )

    reference = normalize_text(
        sample["transcription"]
    )

    sample_ids.append(sample_id)
    audio_inputs.append(audio_input)
    references.append(reference)
    durations.append(duration)

    total_audio_seconds += duration


print()
print(f"Usable samples       : {len(audio_inputs)}")
print(f"Skipped >= 40 sec    : {len(too_long_ids)}")

if too_long_ids:
    print("\nSkipped long samples:")

    for sample_id, duration in too_long_ids:
        print(
            f"  ID={sample_id}: "
            f"{duration:.2f} sec"
        )

print(
    f"\nTotal audio duration : "
    f"{total_audio_seconds:.2f} sec "
    f"({total_audio_seconds / 60:.2f} min)"
)

# ============================================================
# LOAD MODEL
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
    f"Model loaded in {model_load_time:.2f} sec"
)


# ============================================================
# GPU METRICS
# ============================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    print("WARNING: CUDA is NOT available!")

# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

print()
print("[5/6] Running full inference...")

print(f"Chunk size       : {CHUNK_SIZE}")
print(f"Model batch size : {MODEL_BATCH_SIZE}")

saved_rows = []
processed_ids = set()

cumulative_inference_time = 0.0


# ------------------------------------------------------------
# Restore previous checkpoint if it exists
# ------------------------------------------------------------

if CHECKPOINT_CSV.exists():

    print()
    print(f"Found checkpoint: {CHECKPOINT_CSV}")

    with CHECKPOINT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            saved_rows.append(row)
            processed_ids.add(
                str(row["id"])
            )

    print(
        f"Restored predictions: "
        f"{len(saved_rows)}"
    )


if STATE_FILE.exists():

    with STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        state = json.load(f)

    cumulative_inference_time = state.get(
        "cumulative_inference_time",
        0.0,
    )

    print(
        f"Restored inference time: "
        f"{cumulative_inference_time:.2f} sec"
    )


# ============================================================
# DETERMINE REMAINING SAMPLES
# ============================================================

remaining_indices = [
    i
    for i, sample_id in enumerate(sample_ids)
    if str(sample_id) not in processed_ids
]


print(
    f"Already processed : {len(processed_ids)}"
)

print(
    f"Remaining         : {len(remaining_indices)}"
)


# ============================================================
# CHECKPOINT CSV HEADER
# ============================================================

if not CHECKPOINT_CSV.exists():

    with CHECKPOINT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "duration_sec",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "exact_match",
            ],
        )

        writer.writeheader()


# ============================================================
# INFERENCE CHUNK BY CHUNK
# ============================================================

num_chunks = math.ceil(
    len(remaining_indices) / CHUNK_SIZE
)


for chunk_number, offset in enumerate(
    tqdm(
        range(
            0,
            len(remaining_indices),
            CHUNK_SIZE,
        ),
        total=num_chunks,
        desc="OmniASR",
        unit="chunk",
    ),
    start=1,
):

    chunk_indices = remaining_indices[
        offset:
        offset + CHUNK_SIZE
    ]

    chunk_audio = [
        audio_inputs[i]
        for i in chunk_indices
    ]

    chunk_refs = [
        references[i]
        for i in chunk_indices
    ]

    chunk_ids = [
        sample_ids[i]
        for i in chunk_indices
    ]

    chunk_durations = [
        durations[i]
        for i in chunk_indices
    ]


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    chunk_start = time.perf_counter()


    chunk_predictions = pipeline.transcribe(
        chunk_audio,
        batch_size=MODEL_BATCH_SIZE,
    )


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    chunk_elapsed = (
        time.perf_counter()
        - chunk_start
    )

    cumulative_inference_time += chunk_elapsed


    chunk_hypotheses = [
        normalize_text(prediction)
        for prediction in chunk_predictions
    ]


    # --------------------------------------------------------
    # SAVE THIS CHUNK IMMEDIATELY
    # --------------------------------------------------------

    chunk_rows = []


    for (
        sample_id,
        duration,
        reference,
        hypothesis,
    ) in zip(
        chunk_ids,
        chunk_durations,
        chunk_refs,
        chunk_hypotheses,
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
            reference == hypothesis
        )


        row = {
            "id": sample_id,
            "duration_sec": duration,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": sample_wer,
            "cer": sample_cer,
            "exact_match": exact_match,
        }

        chunk_rows.append(row)


    # Append predictions to CSV
    with CHECKPOINT_CSV.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "duration_sec",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "exact_match",
            ],
        )

        writer.writerows(chunk_rows)

        # Force write to disk
        f.flush()
        os.fsync(f.fileno())


    # Save cumulative state
    state = {
        "model": MODEL,
        "processed_samples": (
            len(processed_ids)
            + len(chunk_rows)
        ),
        "cumulative_inference_time":
            cumulative_inference_time,
    }


    tmp_state = STATE_FILE.with_suffix(
        ".tmp"
    )

    with tmp_state.open(
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


    # Atomic replacement
    tmp_state.replace(
        STATE_FILE
    )


    # Update current run state
    for row in chunk_rows:

        saved_rows.append(row)

        processed_ids.add(
            str(row["id"])
        )


    tqdm.write(
        f"Chunk {chunk_number}: "
        f"{len(chunk_rows)} samples | "
        f"{chunk_elapsed:.2f}s | "
        f"saved ✓"
    )


# ============================================================
# RELOAD COMPLETE CHECKPOINT
# ============================================================

rows = []

with CHECKPOINT_CSV.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


# Reconstruct complete references / hypotheses
final_references = [
    row["reference"]
    for row in rows
]

final_hypotheses = [
    row["hypothesis"]
    for row in rows
]


if len(final_references) != len(sample_ids):

    raise RuntimeError(
        "Benchmark incomplete: "
        f"{len(final_references)} predictions "
        f"for {len(sample_ids)} samples."
    )


# Use these for all final metrics
references = final_references
hypotheses = final_hypotheses

inference_time = cumulative_inference_time


# ------------------------------------------------------------
# Restore previous checkpoint if it exists
# ------------------------------------------------------------

if CHECKPOINT_CSV.exists():

    print()
    print(f"Found checkpoint: {CHECKPOINT_CSV}")

    with CHECKPOINT_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            saved_rows.append(row)
            processed_ids.add(
                str(row["id"])
            )

    print(
        f"Restored predictions: "
        f"{len(saved_rows)}"
    )


if STATE_FILE.exists():

    with STATE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        state = json.load(f)

    cumulative_inference_time = state.get(
        "cumulative_inference_time",
        0.0,
    )

    print(
        f"Restored inference time: "
        f"{cumulative_inference_time:.2f} sec"
    )


# ============================================================
# DETERMINE REMAINING SAMPLES
# ============================================================

remaining_indices = [
    i
    for i, sample_id in enumerate(sample_ids)
    if str(sample_id) not in processed_ids
]


print(
    f"Already processed : {len(processed_ids)}"
)

print(
    f"Remaining         : {len(remaining_indices)}"
)


# ============================================================
# CHECKPOINT CSV HEADER
# ============================================================

if not CHECKPOINT_CSV.exists():

    with CHECKPOINT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "duration_sec",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "exact_match",
            ],
        )

        writer.writeheader()


# ============================================================
# INFERENCE CHUNK BY CHUNK
# ============================================================

num_chunks = math.ceil(
    len(remaining_indices) / CHUNK_SIZE
)


for chunk_number, offset in enumerate(
    tqdm(
        range(
            0,
            len(remaining_indices),
            CHUNK_SIZE,
        ),
        total=num_chunks,
        desc="OmniASR",
        unit="chunk",
    ),
    start=1,
):

    chunk_indices = remaining_indices[
        offset:
        offset + CHUNK_SIZE
    ]

    chunk_audio = [
        audio_inputs[i]
        for i in chunk_indices
    ]

    chunk_refs = [
        references[i]
        for i in chunk_indices
    ]

    chunk_ids = [
        sample_ids[i]
        for i in chunk_indices
    ]

    chunk_durations = [
        durations[i]
        for i in chunk_indices
    ]


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    chunk_start = time.perf_counter()


    chunk_predictions = pipeline.transcribe(
        chunk_audio,
        batch_size=MODEL_BATCH_SIZE,
    )


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    chunk_elapsed = (
        time.perf_counter()
        - chunk_start
    )

    cumulative_inference_time += chunk_elapsed


    chunk_hypotheses = [
        normalize_text(prediction)
        for prediction in chunk_predictions
    ]


    # --------------------------------------------------------
    # SAVE THIS CHUNK IMMEDIATELY
    # --------------------------------------------------------

    chunk_rows = []


    for (
        sample_id,
        duration,
        reference,
        hypothesis,
    ) in zip(
        chunk_ids,
        chunk_durations,
        chunk_refs,
        chunk_hypotheses,
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
            reference == hypothesis
        )


        row = {
            "id": sample_id,
            "duration_sec": duration,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": sample_wer,
            "cer": sample_cer,
            "exact_match": exact_match,
        }

        chunk_rows.append(row)


    # Append predictions to CSV
    with CHECKPOINT_CSV.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "duration_sec",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "exact_match",
            ],
        )

        writer.writerows(chunk_rows)

        # Force write to disk
        f.flush()
        os.fsync(f.fileno())


    # Save cumulative state
    state = {
        "model": MODEL,
        "processed_samples": (
            len(processed_ids)
            + len(chunk_rows)
        ),
        "cumulative_inference_time":
            cumulative_inference_time,
    }


    tmp_state = STATE_FILE.with_suffix(
        ".tmp"
    )

    with tmp_state.open(
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


    # Atomic replacement
    tmp_state.replace(
        STATE_FILE
    )


    # Update current run state
    for row in chunk_rows:

        saved_rows.append(row)

        processed_ids.add(
            str(row["id"])
        )


    tqdm.write(
        f"Chunk {chunk_number}: "
        f"{len(chunk_rows)} samples | "
        f"{chunk_elapsed:.2f}s | "
        f"saved ✓"
    )


# ============================================================
# RELOAD COMPLETE CHECKPOINT
# ============================================================

rows = []

with CHECKPOINT_CSV.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


# Reconstruct complete references / hypotheses
final_references = [
    row["reference"]
    for row in rows
]

final_hypotheses = [
    row["hypothesis"]
    for row in rows
]


if len(final_references) != len(sample_ids):

    raise RuntimeError(
        "Benchmark incomplete: "
        f"{len(final_references)} predictions "
        f"for {len(sample_ids)} samples."
    )


# Use these for all final metrics
references = final_references
hypotheses = final_hypotheses

inference_time = cumulative_inference_time

# ============================================================
# GLOBAL ASR METRICS
# ============================================================

print()
print("[6/6] Calculating metrics...")


overall_wer = wer(
    references,
    hypotheses,
)

overall_cer = cer(
    references,
    hypotheses,
)

overall_mer = mer(
    references,
    hypotheses,
)

overall_wil = wil(
    references,
    hypotheses,
)

overall_wip = wip(
    references,
    hypotheses,
)


word_stats = process_words(
    references,
    hypotheses,
)


# ============================================================
# PER-SAMPLE METRICS
# ============================================================

rows = []

sample_wers = []
sample_cers = []

exact_matches = 0


for (
    sample_id,
    reference,
    hypothesis,
    duration,
) in tqdm(
    zip(
        sample_ids,
        references,
        hypotheses,
        durations,
    ),
    total=len(references),
    desc="Per-sample metrics",
    unit="sample",
):

    sample_wer = wer(
        reference,
        hypothesis,
    )

    sample_cer = cer(
        reference,
        hypothesis,
    )

    is_exact = (
        reference == hypothesis
    )

    if is_exact:
        exact_matches += 1

    sample_wers.append(sample_wer)
    sample_cers.append(sample_cer)

    rows.append({
        "id": sample_id,
        "duration_sec": duration,
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": sample_wer,
        "cer": sample_cer,
        "exact_match": int(is_exact),
    })


# ============================================================
# SPEED METRICS
# ============================================================

if total_audio_seconds > 0:

    rtf = (
        inference_time
        / total_audio_seconds
    )

    realtime_speed = 1.0 / rtf

else:

    rtf = None
    realtime_speed = None


exact_match_rate = (
    exact_matches
    / len(references)
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    peak_vram_bytes = (
        torch.cuda.max_memory_allocated()
    )

    peak_vram_gb = (
        peak_vram_bytes
        / 1024 ** 3
    )

else:

    peak_vram_gb = None


# ============================================================
# SAVE CSV
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "id",
            "duration_sec",
            "reference",
            "hypothesis",
            "wer",
            "cer",
            "exact_match",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


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
    "Dataset              : FLEURS kk_kz TEST"
)

print(
    f"Raw samples          : {len(dataset)}"
)

print(
    f"Unique samples       : {len(samples)}"
)

print(
    f"Duplicates removed   : {duplicate_count}"
)

print(
    f"Audio duration       : "
    f"{total_audio_seconds / 60:.2f} min"
)


print()
print("MODEL")
print("-" * 80)

print(
    f"Model                : {MODEL}"
)

print(
    f"Chunk size           : {CHUNK_SIZE}"
)

print(
    f"GPU batch size       : {MODEL_BATCH_SIZE}"
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
    f"{exact_matches}/{len(references)} "
    f"({exact_match_rate * 100:.2f}%)"
)


print()
print("WORD ERRORS")
print("-" * 80)

print(
    f"Hits                 : {word_stats.hits}"
)

print(
    f"Substitutions        : {word_stats.substitutions}"
)

print(
    f"Deletions            : {word_stats.deletions}"
)

print(
    f"Insertions           : {word_stats.insertions}"
)


print()
print("PER-UTTERANCE")
print("-" * 80)

print(
    f"Mean sample WER      : "
    f"{mean(sample_wers) * 100:.2f}%"
)

print(
    f"Median sample WER    : "
    f"{median(sample_wers) * 100:.2f}%"
)

print(
    f"Mean sample CER      : "
    f"{mean(sample_cers) * 100:.2f}%"
)

print(
    f"Median sample CER    : "
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
    f"{inference_time:.2f} sec"
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
print(
    f"Too long excluded    : {len(too_long_ids)}"
)
print()
print("OUTPUT")
print("-" * 80)

print(
    f"CSV saved to:\n{OUTPUT_CSV}"
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)