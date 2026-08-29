import csv
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Audio, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor


# ============================================================
# CONFIG
# ============================================================

MODEL_REPO = "kazega0/KazEGA-HuBERT"

BASE_MODEL = "facebook/hubert-base-ls960"

DATASET_REPO = "ai4kazakh/ISSAI_KazEmoTTS"


# Three KazEmoTTS speakers
SPEAKERS = [
    "akzhol",
    "madi",
    "marzhan",
]

SPLITS = [
    "test",
]


SAMPLE_RATE = 16_000

# Official KazEGA-HuBERT inference setup:
# maximum first 10 seconds
MAX_LENGTH = 160_000


# RTX 4060 Laptop 8 GB:
# start safely.
BATCH_SIZE = 4


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent


CHECKPOINT_CSV = (
    OUTPUT_DIR
    / "kazega_kazemotts_checkpoint.csv"
)


STATE_JSON = (
    OUTPUT_DIR
    / "kazega_kazemotts_state.json"
)


RESULTS_CSV = (
    OUTPUT_DIR
    / "kazega_kazemotts_results.csv"
)


SUMMARY_JSON = (
    OUTPUT_DIR
    / "kazega_kazemotts_summary.json"
)


PER_CLASS_CSV = (
    OUTPUT_DIR
    / "kazega_kazemotts_per_class.csv"
)


CONFUSION_CSV = (
    OUTPUT_DIR
    / "kazega_kazemotts_confusion_matrix.csv"
)


# ============================================================
# LABELS
# ============================================================

# KazEmoTTS labels:
#
# neutral
# angry
# happy
# sad
# scared
# surprised
#
# KazEGA-HuBERT calls scared -> fearful.

GT_MAPPING = {
    "neutral": "neutral",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",

    "scared": "fearful",
    "fear": "fearful",
    "fearful": "fearful",

    "surprised": "surprised",
    "surprise": "surprised",
}


# Ground-truth classes in KazEmoTTS
EVAL_CLASSES = [
    "angry",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


# KazEGA-HuBERT can additionally output disgusted.
MODEL_CLASSES = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


FIELDS = [
    "recording_idx",
    "speaker",
    "split",
    "ground_truth",
    "prediction",
    "confidence",
    "correct",
    "duration_sec",
    "used_duration_sec",
    "truncated",
]


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MultiTaskHubert(nn.Module):

    def __init__(
        self,
        num_emotions,
        num_genders,
        num_ages,
    ):

        super().__init__()


        # IMPORTANT:
        # checkpoint actually stores keys as backbone.*
        self.backbone = HubertModel.from_pretrained(
            BASE_MODEL,
            output_hidden_states=True,
        )


        hidden_size = (
            self.backbone.config.hidden_size
        )


        num_layers = (
            self.backbone.config.num_hidden_layers
            + 1
        )


        # ----------------------------------------------------
        # Learnable layer weights
        # ----------------------------------------------------

        self.emotion_weights = nn.Parameter(
            torch.ones(num_layers)
        )

        self.gender_weights = nn.Parameter(
            torch.ones(num_layers)
        )

        self.age_weights = nn.Parameter(
            torch.ones(num_layers)
        )


        # ----------------------------------------------------
        # Emotion head
        # ----------------------------------------------------

        self.emotion_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_emotions,
            ),
        )


        # ----------------------------------------------------
        # Gender head
        # ----------------------------------------------------

        self.gender_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                num_genders,
            ),
        )


        # ----------------------------------------------------
        # Age head
        # ----------------------------------------------------

        self.age_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                256,
                num_ages,
            ),
        )


    def forward(
        self,
        input_values,
        input_lengths,
    ):

        outputs = self.backbone(
            input_values,
            output_hidden_states=True,
        )


        # ----------------------------------------------------
        # Stack all HuBERT layers
        #
        # [layers, batch, time, hidden]
        # ----------------------------------------------------

        hidden = torch.stack(
            outputs.hidden_states,
            dim=0,
        )


        # Convert raw audio length
        # -> HuBERT feature-frame length
        feature_lengths = (
            self.backbone
            ._get_feat_extract_output_lengths(
                input_lengths
            )
        )


        # ----------------------------------------------------
        # Weighted layer pooling
        # ----------------------------------------------------

        def pool(
            layer_weights,
        ):

            weights = torch.softmax(
                layer_weights,
                dim=0,
            )


            # weighted sum across layers
            features = (

                weights[
                    :,
                    None,
                    None,
                    None,
                ]

                * hidden

            ).sum(dim=0)


            # features:
            # [batch, time, hidden]

            pooled = []


            for batch_idx in range(
                features.shape[0]
            ):

                length = int(
                    feature_lengths[
                        batch_idx
                    ].item()
                )


                valid = features[
                    batch_idx,
                    :length,
                    :
                ]


                pooled.append(
                    valid.mean(dim=0)
                )


            return torch.stack(
                pooled,
                dim=0,
            )


        emotion_features = pool(
            self.emotion_weights
        )


        gender_features = pool(
            self.gender_weights
        )


        age_features = pool(
            self.age_weights
        )


        return (

            self.emotion_head(
                emotion_features
            ),

            self.gender_head(
                gender_features
            ),

            self.age_head(
                age_features
            ),
        )


# ============================================================
# AUDIO LOADER
# ============================================================

def load_audio(sample):

    audio = sample["audio"]


    audio_bytes = audio.get(
        "bytes"
    )

    audio_path = audio.get(
        "path"
    )


    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    if audio_bytes is not None:

        waveform, sr = sf.read(

            io.BytesIO(
                audio_bytes
            ),

            dtype="float32",
        )


    elif audio_path is not None:

        waveform, sr = sf.read(
            audio_path,
            dtype="float32",
        )


    else:

        raise RuntimeError(
            "Audio has no bytes/path"
        )


    # --------------------------------------------------------
    # Stereo -> mono
    # --------------------------------------------------------

    if waveform.ndim > 1:

        waveform = waveform.mean(
            axis=1
        )


    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Resample -> 16 kHz
    # --------------------------------------------------------

    if sr != SAMPLE_RATE:

        waveform = librosa.resample(

            waveform,

            orig_sr=sr,

            target_sr=SAMPLE_RATE,
        )


    # --------------------------------------------------------
    # Original duration
    # --------------------------------------------------------

    duration_sec = (
        len(waveform)
        / SAMPLE_RATE
    )


    # --------------------------------------------------------
    # KazEGA inference:
    # use maximum first 10 sec
    # --------------------------------------------------------

    truncated = (
        len(waveform)
        > MAX_LENGTH
    )


    waveform = waveform[
        :MAX_LENGTH
    ]


    true_length = len(
        waveform
    )


    used_duration_sec = (
        true_length
        / SAMPLE_RATE
    )


    # --------------------------------------------------------
    # Pad to exactly 10 sec
    # --------------------------------------------------------

    if true_length < MAX_LENGTH:

        waveform = np.pad(

            waveform,

            (
                0,
                MAX_LENGTH
                - true_length,
            ),
        )


    return (
        waveform,
        true_length,
        duration_sec,
        used_duration_sec,
        truncated,
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def append_checkpoint(
    rows,
):

    with CHECKPOINT_CSV.open(

        "a",

        encoding="utf-8",

        newline="",

    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )


        writer.writerows(
            rows
        )


        file.flush()

        os.fsync(
            file.fileno()
        )


def save_state(
    processed_samples,
    cumulative_inference_time,
):

    state = {

        "model":
            MODEL_REPO,

        "processed_samples":
            processed_samples,

        "cumulative_inference_time":
            cumulative_inference_time,
    }


    temp_file = (
        STATE_JSON.with_suffix(
            ".tmp"
        )
    )


    with temp_file.open(

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            state,

            file,

            ensure_ascii=False,

            indent=2,
        )


        file.flush()

        os.fsync(
            file.fileno()
        )


    temp_file.replace(
        STATE_JSON
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)

print(
    "KazEGA-HuBERT — FULL KazEmoTTS BENCHMARK"
)

print("=" * 80)


print(
    f"Device: {DEVICE}"
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# 1. LOAD DATASET
# ============================================================

print()
print(
    "[1/6] Loading ALL KazEmoTTS..."
)


datasets = []


for speaker in SPEAKERS:

    for split in SPLITS:

        print(
            f"Loading {speaker}/{split}..."
        )


        part = load_dataset(

            DATASET_REPO,

            speaker,

            split=split,
        )


        # Do not decode with TorchCodec.
        part = part.cast_column(

            "audio",

            Audio(
                decode=False
            ),
        )


        # Add source metadata
        part = part.add_column(

            "speaker_name",

            [speaker]
            * len(part),
        )


        part = part.add_column(

            "split_name",

            [split]
            * len(part),
        )


        datasets.append(
            part
        )


dataset = concatenate_datasets(
    datasets
)


print()
print(
    f"TOTAL RECORDINGS: "
    f"{len(dataset):,}"
)


if len(dataset) != 54760:

    print(
        "WARNING: expected 54,760 "
        "recordings."
    )


# ============================================================
# LABEL CHECK
# ============================================================

raw_distribution = Counter(

    str(label).lower()

    for label in dataset[
        "emotion"
    ]
)


print()
print(
    "KazEmoTTS labels:"
)


for label, count in sorted(
    raw_distribution.items()
):

    print(
        f"  {label:12s}: "
        f"{count:,}"
    )


unknown_labels = [

    label

    for label in raw_distribution

    if label
    not in GT_MAPPING
]


if unknown_labels:

    raise RuntimeError(

        "Unknown emotion labels: "
        f"{unknown_labels}"
    )


# ============================================================
# 2. FEATURE EXTRACTOR
# ============================================================

print()
print(
    "[2/6] Loading "
    "Wav2Vec2FeatureExtractor..."
)


processor = (
    Wav2Vec2FeatureExtractor
    .from_pretrained(
        BASE_MODEL
    )
)


print(
    "Feature extractor OK"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print()
print(
    "[3/6] Loading KazEGA-HuBERT..."
)


model_path = hf_hub_download(

    repo_id=MODEL_REPO,

    filename="model.pt",
)


labels_path = hf_hub_download(

    repo_id=MODEL_REPO,

    filename="label_encoders.json",
)


# ------------------------------------------------------------
# Label encoders
# ------------------------------------------------------------

with open(

    labels_path,

    "r",

    encoding="utf-8",

) as file:

    encoders = json.load(
        file
    )


id2label = {

    task: {

        idx: label

        for label, idx
        in mapping.items()
    }

    for task, mapping
    in encoders.items()
}


print(
    "Emotion labels:",
    id2label["emotion"]
)


# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(

    model_path,

    map_location="cpu",

    weights_only=False,
)


model_load_start = (
    time.perf_counter()
)


model = MultiTaskHubert(

    checkpoint[
        "num_emotions"
    ],

    checkpoint[
        "num_genders"
    ],

    checkpoint[
        "num_ages"
    ],
)


# Since our class is now named backbone,
# checkpoint keys match directly.

model.load_state_dict(

    checkpoint[
        "model_state_dict"
    ],

    strict=True,
)


model.to(
    DEVICE
)


model.eval()


model_load_time = (

    time.perf_counter()
    - model_load_start
)


print(
    f"Model loaded successfully "
    f"in {model_load_time:.2f} sec"
)


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


# ============================================================
# 4. RESTORE CHECKPOINT
# ============================================================

print()
print(
    "[4/6] Restoring benchmark checkpoint..."
)


processed = set()

cumulative_inference_time = 0.0


if CHECKPOINT_CSV.exists():

    with CHECKPOINT_CSV.open(

        "r",

        encoding="utf-8",

        newline="",

    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            processed.add(

                int(
                    row[
                        "recording_idx"
                    ]
                )
            )


else:

    with CHECKPOINT_CSV.open(

        "w",

        encoding="utf-8",

        newline="",

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=FIELDS,
        )


        writer.writeheader()


# ------------------------------------------------------------
# Restore cumulative inference time
# ------------------------------------------------------------

if STATE_JSON.exists():

    with STATE_JSON.open(

        "r",

        encoding="utf-8",

    ) as file:

        state = json.load(
            file
        )


    cumulative_inference_time = float(

        state.get(

            "cumulative_inference_time",

            0.0,
        )
    )


remaining = [

    idx

    for idx in range(
        len(dataset)
    )

    if idx
    not in processed
]


print(
    f"Already processed : "
    f"{len(processed):,}"
)

print(
    f"Remaining         : "
    f"{len(remaining):,}"
)


# ============================================================
# 5. INFERENCE
# ============================================================

print()
print(
    "[5/6] Running inference..."
)

print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)


number_of_batches = math.ceil(

    len(remaining)
    / BATCH_SIZE
)


for start in tqdm(

    range(
        0,
        len(remaining),
        BATCH_SIZE,
    ),

    total=number_of_batches,

    desc="KazEGA-HuBERT",

    unit="batch",

):


    batch_indices = remaining[

        start:
        start + BATCH_SIZE
    ]


    waveforms = []

    true_lengths = []

    metadata = []


    # --------------------------------------------------------
    # Prepare samples
    # --------------------------------------------------------

    for recording_idx in (
        batch_indices
    ):

        sample = dataset[
            recording_idx
        ]


        (
            waveform,
            true_length,
            duration_sec,
            used_duration_sec,
            truncated,
        ) = load_audio(
            sample
        )


        raw_label = str(
            sample["emotion"]
        ).lower()


        ground_truth = (
            GT_MAPPING[
                raw_label
            ]
        )


        waveforms.append(
            waveform
        )


        true_lengths.append(
            true_length
        )


        metadata.append({

            "recording_idx":
                recording_idx,

            "speaker":
                sample[
                    "speaker_name"
                ],

            "split":
                sample[
                    "split_name"
                ],

            "ground_truth":
                ground_truth,

            "duration_sec":
                duration_sec,

            "used_duration_sec":
                used_duration_sec,

            "truncated":
                int(truncated),
        })


    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(

        waveforms,

        sampling_rate=SAMPLE_RATE,

        return_tensors="pt",
    )


    input_values = (
        inputs
        .input_values
        .to(DEVICE)
    )


    lengths_tensor = torch.tensor(

        true_lengths,

        dtype=torch.long,

        device=DEVICE,
    )


    # --------------------------------------------------------
    # GPU timing
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()


    batch_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.inference_mode():

        (
            emotion_logits,
            _,
            _,
        ) = model(

            input_values,

            lengths_tensor,
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


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = F.softmax(

        emotion_logits,

        dim=-1,
    )


    batch_rows = []


    for local_idx, info in enumerate(
        metadata
    ):

        class_idx = int(

            probabilities[
                local_idx
            ].argmax()
        )


        prediction = (
            id2label[
                "emotion"
            ][
                class_idx
            ]
        )


        confidence = float(

            probabilities[
                local_idx,
                class_idx,
            ].item()
        )


        correct = int(

            prediction
            == info[
                "ground_truth"
            ]
        )


        batch_rows.append({

            "recording_idx":
                info[
                    "recording_idx"
                ],

            "speaker":
                info[
                    "speaker"
                ],

            "split":
                info[
                    "split"
                ],

            "ground_truth":
                info[
                    "ground_truth"
                ],

            "prediction":
                prediction,

            "confidence":
                confidence,

            "correct":
                correct,

            "duration_sec":
                info[
                    "duration_sec"
                ],

            "used_duration_sec":
                info[
                    "used_duration_sec"
                ],

            "truncated":
                info[
                    "truncated"
                ],
        })


    # --------------------------------------------------------
    # SAVE AFTER EVERY GPU BATCH
    # --------------------------------------------------------

    append_checkpoint(
        batch_rows
    )


    for info in metadata:

        processed.add(

            info[
                "recording_idx"
            ]
        )


    save_state(

        processed_samples=len(
            processed
        ),

        cumulative_inference_time=(
            cumulative_inference_time
        ),
    )


# ============================================================
# 6. METRICS
# ============================================================

print()
print(
    "[6/6] Calculating metrics..."
)


rows = []


with CHECKPOINT_CSV.open(

    "r",

    encoding="utf-8",

    newline="",

) as file:

    rows = list(
        csv.DictReader(file)
    )


# ------------------------------------------------------------
# Duplicate safety
# ------------------------------------------------------------

recording_ids = [

    int(
        row[
            "recording_idx"
        ]
    )

    for row in rows
]


if len(recording_ids) != len(
    set(recording_ids)
):

    raise RuntimeError(
        "Duplicate recording_idx "
        "inside checkpoint!"
    )


# ------------------------------------------------------------
# Completeness
# ------------------------------------------------------------

if len(rows) != len(dataset):

    raise RuntimeError(

        "Benchmark incomplete: "
        f"{len(rows):,} / "
        f"{len(dataset):,}"
    )


rows.sort(

    key=lambda row:

        int(
            row[
                "recording_idx"
            ]
        )
)


# ============================================================
# ARRAYS
# ============================================================

y_true = [

    row[
        "ground_truth"
    ]

    for row in rows
]


y_pred = [

    row[
        "prediction"
    ]

    for row in rows
]


# ============================================================
# GLOBAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)


balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred,
    )
)


# ------------------------------------------------------------
# Main Macro F1:
# six actual KazEmoTTS classes
# ------------------------------------------------------------

(
    macro_precision,
    macro_recall,
    macro_f1,
    _,
) = precision_recall_fscore_support(

    y_true,

    y_pred,

    labels=EVAL_CLASSES,

    average="macro",

    zero_division=0,
)


# ------------------------------------------------------------
# Weighted metrics
# ------------------------------------------------------------

(
    weighted_precision,
    weighted_recall,
    weighted_f1,
    _,
) = precision_recall_fscore_support(

    y_true,

    y_pred,

    labels=EVAL_CLASSES,

    average="weighted",

    zero_division=0,
)


# ============================================================
# PER-CLASS
# ============================================================

(
    class_precision,
    class_recall,
    class_f1,
    class_support,
) = precision_recall_fscore_support(

    y_true,

    y_pred,

    labels=EVAL_CLASSES,

    zero_division=0,
)


per_class_rows = []


for idx, label in enumerate(
    EVAL_CLASSES
):

    per_class_rows.append({

        "class":
            label,

        "precision":
            float(
                class_precision[idx]
            ),

        "recall":
            float(
                class_recall[idx]
            ),

        "f1":
            float(
                class_f1[idx]
            ),

        "support":
            int(
                class_support[idx]
            ),
    })


with PER_CLASS_CSV.open(

    "w",

    encoding="utf-8",

    newline="",

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=[
            "class",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )


    writer.writeheader()

    writer.writerows(
        per_class_rows
    )


# ============================================================
# CONFUSION MATRIX 6 x 7
# ============================================================

confusion = {

    true_label: {

        pred_label: 0

        for pred_label
        in MODEL_CLASSES
    }

    for true_label
    in EVAL_CLASSES
}


for true_label, pred_label in zip(
    y_true,
    y_pred,
):

    confusion[
        true_label
    ][
        pred_label
    ] += 1


with CONFUSION_CSV.open(

    "w",

    encoding="utf-8",

    newline="",

) as file:

    writer = csv.writer(
        file
    )


    writer.writerow([

        "true\\pred",

        *MODEL_CLASSES,
    ])


    for true_label in (
        EVAL_CLASSES
    ):

        writer.writerow([

            true_label,

            *[
                confusion[
                    true_label
                ][
                    pred_label
                ]

                for pred_label
                in MODEL_CLASSES
            ],
        ])


# ============================================================
# DISTRIBUTIONS
# ============================================================

gt_distribution = Counter(
    y_true
)


pred_distribution = Counter(
    y_pred
)


disgusted_count = (
    pred_distribution[
        "disgusted"
    ]
)


disgusted_rate = (

    disgusted_count
    / len(rows)
)


mean_confidence = float(

    np.mean([

        float(
            row[
                "confidence"
            ]
        )

        for row in rows
    ])
)


# ============================================================
# TRUNCATION
# ============================================================

truncated_count = sum(

    int(
        row[
            "truncated"
        ]
    )

    for row in rows
)


# ============================================================
# PER-SPEAKER
# ============================================================

speaker_results = {}


for speaker in SPEAKERS:

    speaker_rows = [

        row

        for row in rows

        if row[
            "speaker"
        ] == speaker
    ]


    speaker_true = [

        row[
            "ground_truth"
        ]

        for row in speaker_rows
    ]


    speaker_pred = [

        row[
            "prediction"
        ]

        for row in speaker_rows
    ]


    speaker_accuracy = (
        accuracy_score(

            speaker_true,

            speaker_pred,
        )
    )


    (
        _,
        _,
        speaker_macro_f1,
        _,
    ) = precision_recall_fscore_support(

        speaker_true,

        speaker_pred,

        labels=EVAL_CLASSES,

        average="macro",

        zero_division=0,
    )


    speaker_results[
        speaker
    ] = {

        "samples":
            len(
                speaker_rows
            ),

        "accuracy":
            float(
                speaker_accuracy
            ),

        "macro_f1":
            float(
                speaker_macro_f1
            ),
    }


# ============================================================
# PERFORMANCE
# ============================================================

total_audio_seconds = sum(

    float(
        row[
            "used_duration_sec"
        ]
    )

    for row in rows
)


rtf = (

    cumulative_inference_time
    / total_audio_seconds
)


realtime_speed = (
    1.0 / rtf
)


if torch.cuda.is_available():

    peak_vram_gb = (

        torch.cuda
        .max_memory_allocated()

        / (1024 ** 3)
    )


else:

    peak_vram_gb = None


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

with RESULTS_CSV.open(

    "w",

    encoding="utf-8",

    newline="",

) as file:

    writer = csv.DictWriter(

        file,

        fieldnames=FIELDS,
    )


    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "model":
        MODEL_REPO,

    "dataset":
        DATASET_REPO,

    "samples":
        len(rows),

    "used_audio_hours":
        total_audio_seconds
        / 3600,

    "truncated_over_10s":
        truncated_count,

    "accuracy":
        float(
            accuracy
        ),

    "balanced_accuracy":
        float(
            balanced_accuracy
        ),

    "macro_precision_6class":
        float(
            macro_precision
        ),

    "macro_recall_6class":
        float(
            macro_recall
        ),

    "macro_f1_6class":
        float(
            macro_f1
        ),

    "weighted_f1":
        float(
            weighted_f1
        ),

    "mean_confidence":
        mean_confidence,

    "disgusted_predictions":
        disgusted_count,

    "disgusted_prediction_rate":
        disgusted_rate,

    "ground_truth_distribution":
        dict(
            gt_distribution
        ),

    "prediction_distribution":
        dict(
            pred_distribution
        ),

    "per_speaker":
        speaker_results,

    "model_load_seconds":
        model_load_time,

    "inference_seconds":
        cumulative_inference_time,

    "rtf":
        rtf,

    "realtime_speed":
        realtime_speed,

    "peak_vram_gb":
        peak_vram_gb,
}


with SUMMARY_JSON.open(

    "w",

    encoding="utf-8",

) as file:

    json.dump(

        summary,

        file,

        ensure_ascii=False,

        indent=2,
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
    f"Recordings           : "
    f"{len(rows):,}"
)

print(
    f"Used audio           : "
    f"{total_audio_seconds / 3600:.2f} h"
)

print(
    f"Truncated >10 sec    : "
    f"{truncated_count:,}"
)


print()
print("GLOBAL EMOTION METRICS")
print("-" * 80)


print(
    f"Accuracy ↑           : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Balanced Accuracy ↑  : "
    f"{balanced_accuracy * 100:.2f}%"
)


print(
    f"Macro Precision ↑    : "
    f"{macro_precision * 100:.2f}%"
)


print(
    f"Macro Recall ↑       : "
    f"{macro_recall * 100:.2f}%"
)


print(
    f"Macro F1 ↑           : "
    f"{macro_f1 * 100:.2f}%"
)


print(
    f"Weighted F1 ↑        : "
    f"{weighted_f1 * 100:.2f}%"
)


print(
    f"Mean confidence      : "
    f"{mean_confidence * 100:.2f}%"
)


print(
    f"Disgusted predictions: "
    f"{disgusted_count:,} "
    f"({disgusted_rate * 100:.2f}%)"
)


# ============================================================
# PER CLASS
# ============================================================

print()
print("PER CLASS")
print("-" * 80)


for row in per_class_rows:

    print(

        f"{row['class']:12s} | "

        f"P="
        f"{row['precision'] * 100:6.2f}% | "

        f"R="
        f"{row['recall'] * 100:6.2f}% | "

        f"F1="
        f"{row['f1'] * 100:6.2f}% | "

        f"N="
        f"{row['support']:,}"
    )


# ============================================================
# PER SPEAKER
# ============================================================

print()
print("PER SPEAKER")
print("-" * 80)


for speaker, values in (
    speaker_results.items()
):

    print(

        f"{speaker:10s} | "

        f"N="
        f"{values['samples']:6,d} | "

        f"Accuracy="
        f"{values['accuracy'] * 100:6.2f}% | "

        f"Macro-F1="
        f"{values['macro_f1'] * 100:6.2f}%"
    )


# ============================================================
# PERFORMANCE
# ============================================================

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


# ============================================================
# OUTPUT
# ============================================================

print()
print("OUTPUT")
print("-" * 80)


print(
    f"Results:\n"
    f"{RESULTS_CSV}"
)

print()

print(
    f"Summary:\n"
    f"{SUMMARY_JSON}"
)

print()

print(
    f"Per-class:\n"
    f"{PER_CLASS_CSV}"
)

print()

print(
    f"Confusion matrix:\n"
    f"{CONFUSION_CSV}"
)


print()
print("=" * 80)
print("DONE")
print("=" * 80)