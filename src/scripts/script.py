from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import os

AUDIO_DIR = Path("data/audio")
META_DIR = Path("data/metadata")
PARQUET_DIR = Path("data/parquet")
def run(PARQUET: Path) -> int:
    x = 0
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(PARQUET)

    print(table.schema)
    print(f"Rows: {table.num_rows}")

    rows = table.to_pylist()
    metadata = []

    for i, row in enumerate(rows):
        audio = row["audio"]

        if isinstance(audio, dict):
            audio_bytes = audio.get("bytes")
            original_path = audio.get("path")
        else:
            audio_bytes = audio
            original_path = None

        if not audio_bytes:
            print(f"[SKIP] row {i}: no audio bytes")
            continue

        if original_path:
            suffix = Path(original_path).suffix or ".bin"
        else:
            suffix = ".bin"

        sample_id = f"{PARQUET.stem}_{i:06d}"
        audio_path = AUDIO_DIR / f"{sample_id}{suffix}"

        audio_path.write_bytes(audio_bytes)

        metadata.append({
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "text": row.get("text"),
            "video_id": row.get("video_id"),
            "gender": row.get("gender"),
            "age": row.get("age"),
            "speaker_id": row.get("speaker_id"),
            "original_path": original_path,
        })

    df = pd.DataFrame(metadata)

    metadata_path = META_DIR / f"{PARQUET.stem}.csv"
    df.to_csv(metadata_path, index=False)

    print()
    x += len(df)
    print(f"Recovered: {len(df)} audio files")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Metadata: {metadata_path}")

    return x

if os.path.exists(PARQUET_DIR):
    print("Exists!")

x = 0
for root, dirs, files in os.walk(PARQUET_DIR):
    for filename in files:
        x += run(Path(os.path.join(root, filename)))

print()
print(f"Overall recovered: {x} files")
