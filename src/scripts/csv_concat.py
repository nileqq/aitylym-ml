from pathlib import Path
import pandas as pd

META_DIR = Path("data/metadata")

files = sorted(META_DIR.glob("train-*.csv"))

df = pd.concat(
    [pd.read_csv(file) for file in files],
    ignore_index=True
)

df.to_csv(META_DIR / "metadata.csv", index=False)

print(f"Files merged: {len(files)}")
print(f"Total samples: {len(df)}")
print(f"Unique speakers: {df['speaker_id'].nunique()}")