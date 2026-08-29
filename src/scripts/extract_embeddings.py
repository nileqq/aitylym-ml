from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor


ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = Path("data/metadata/metadata.csv")
OUTPUT_DIR = Path("data/features/wavlm_base_plus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "microsoft/wavlm-base-plus"
TARGET_SR = 16_000

# Сначала тестируем на 200 сэмплах.
# Потом поставим None.
LIMIT = 6610


# -------------------------
# Device
# -------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# -------------------------
# Metadata
# -------------------------

metadata = pd.read_csv(METADATA_PATH)

metadata["speaker_key"] = (
    metadata["video_id"].astype(str)
    + "__"
    + metadata["speaker_id"].astype(str)
)

work = metadata.sample(n=LIMIT, random_state=42).reset_index(drop=True)

print("Samples:", len(work))


# -------------------------
# Hugging Face model
# -------------------------

print(f"Loading {MODEL_ID}...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)

model.to(device)
model.eval()

print("Model loaded")
print("Hidden size:", model.config.hidden_size)


# -------------------------
# Embedding extraction
# -------------------------

embeddings = []
valid_rows = []


for _, row in tqdm(
    work.iterrows(),
    total=len(work),
    desc="Extracting WavLM embeddings",
):
    audio_path = Path(row["audio_path"])

    if not audio_path.is_absolute():
        audio_path = ROOT / audio_path

    if not audio_path.exists():
        print(f"\n[SKIP] Missing: {audio_path}")
        continue

    try:
        # mono + automatic resampling to 16 kHz
        waveform, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
        )

        inputs = feature_extractor(
            waveform,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        # [1, time, 768]
        hidden = outputs.last_hidden_state

        # time pooling:
        # [1, time, 768] -> [768]
        embedding = (
            hidden
            .mean(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        embeddings.append(embedding)
        valid_rows.append(row)

    except Exception as e:
        print(f"\n[ERROR] {audio_path}")
        print(e)


# -------------------------
# Save
# -------------------------

if not embeddings:
    raise RuntimeError("No embeddings were extracted")


X = np.stack(embeddings)

index = pd.DataFrame(valid_rows).reset_index(drop=True)
index.insert(
    0,
    "embedding_row",
    np.arange(len(index))
)


EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
INDEX_PATH = OUTPUT_DIR / "index.csv"

np.save(
    EMBEDDINGS_PATH,
    X,
)

index.to_csv(
    INDEX_PATH,
    index=False,
)


# -------------------------
# Report
# -------------------------

print()
print("=== DONE ===")

print("Embeddings shape:", X.shape)
print("dtype:", X.dtype)

print()
print("Embeddings:")
print(EMBEDDINGS_PATH)

print("Index:")
print(INDEX_PATH)