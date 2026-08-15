import pandas as pd
from pathlib import Path

META_DIR = Path("data/metadata")
METADATA_PATH = META_DIR / "metadata.csv"

metadata = pd.read_csv(METADATA_PATH)

metadata["speaker_key"] = metadata["video_id"].astype(str) + "__" + metadata["speaker_id"].astype(str)

print("=== BASIC INFO ===")
print(f"Samples: {len(metadata)}")
print(f"Columns: {list(metadata.columns)}")
print()


# -------------------------
# Missing values
# -------------------------

print("=== MISSING VALUES ===")
print(metadata.isna().sum())
print()


# -------------------------
# Gender
# -------------------------

print("=== GENDER ===")

gender_stats = metadata["gender"].value_counts(dropna=False)

print(gender_stats)
print()

print(
    (metadata["gender"]
     .value_counts(normalize=True, dropna=False) * 100)
     .round(2)
)
print()


# -------------------------
# Age
# -------------------------

print("=== AGE ===")

age_stats = metadata["age"].value_counts(dropna=False)

print(age_stats)
print()

print(
    (metadata["age"]
     .value_counts(normalize=True, dropna=False) * 100)
     .round(2)
)
print()


# -------------------------
# Speakers
# -------------------------

print("=== SPEAKERS ===")

speaker_counts = (
    metadata.groupby("speaker_key")
    .size()
)

mean = speaker_counts.mean()
median = speaker_counts.median()

# Population variance/std — мы анализируем весь имеющийся набор speakers
variance = speaker_counts.var(ddof=0)
std = speaker_counts.std(ddof=0)

q1 = speaker_counts.quantile(0.25)
q3 = speaker_counts.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

cv = std / mean

print("=== SAMPLES / SPEAKER ===")
print(f"Speakers: {len(speaker_counts)}")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Std: {std:.2f}")

print()
print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")

print()
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print(f"Coefficient of variation: {cv:.2f}")

print()
print("Large-speaker outliers:")
print(
    speaker_counts[
        speaker_counts > upper_bound
    ].sort_values(ascending=False)
)

# -------------------------
# Videos / sources
# -------------------------

print("=== VIDEOS ===")

print(f"Unique videos: {metadata['video_id'].nunique()}")

video_counts = (
    metadata.groupby("video_id")
    .size()
    .sort_values(ascending=False)
)

print()
print("Top 20 videos:")
print(video_counts.head(20))
print()


# -------------------------
# Speaker demographics
# -------------------------

print("=== SPEAKER DEMOGRAPHICS ===")

print("Speaker segments:", metadata["speaker_key"].nunique())

speaker_stats = (
    metadata.groupby("speaker_key")
    .agg(
        samples=("sample_id", "count"),
        age=("age", "first"),
        gender=("gender", "first"),
    )
)

print(speaker_stats["samples"].describe())
print(speaker_stats["age"].value_counts())
print(speaker_stats["gender"].value_counts())


# -------------------------
# Check consistency
# -------------------------

print("=== CONSISTENCY CHECK ===")

gender_per_speaker = metadata.groupby("speaker_id")["gender"].nunique()
age_per_speaker = metadata.groupby("speaker_id")["age"].nunique()

print(
    "Speakers with multiple gender labels:",
    (gender_per_speaker > 1).sum()
)

print(
    "Speakers with multiple age labels:",
    (age_per_speaker > 1).sum()
)

print()


# -------------------------
# Duplicates
# -------------------------

print("=== DUPLICATES ===")

print(
    "Duplicate sample_id:",
    metadata["sample_id"].duplicated().sum()
)

print(
    "Duplicate audio_path:",
    metadata["audio_path"].duplicated().sum()
)

print()


# -------------------------
# Save speaker summary
# -------------------------

speaker_stats.to_csv(
    META_DIR / "speaker_summary.csv"
)

print("Saved:")
print(META_DIR / "speaker_summary.csv")