from pathlib import Path
import wave
import pandas as pd
from tqdm import tqdm

NOISE_DIR = Path("data/audio_hdbscan_noise")

rows = []

for path in tqdm(NOISE_DIR.glob("*.wav")):
    with wave.open(str(path), "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / rate

    rows.append({
        "file": path.name,
        "duration": duration
    })

df = pd.DataFrame(rows)

print("Total:", len(df))
print("< 1 sec:", (df["duration"] < 1).sum())
print("< 2 sec:", (df["duration"] < 2).sum())

i = 2
while (df["duration"] >= i).sum() > 0:
    count = (df["duration"] >= i).sum()

    print(f">= {i} sec: {count}")

    if i >= 5:
        selected = df[df["duration"] >= i]

        for _, row in selected.iterrows():
            print(
                row["file"],
                f'{row["duration"]:.2f}s'
            )

    i += 1

print()
print(df["duration"].describe())