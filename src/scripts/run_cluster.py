from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


FEATURE_DIR = Path("data/features/wavlm_base_plus")
PLOT_DIR = FEATURE_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


X = np.load(FEATURE_DIR / "embeddings.npy")
metadata = pd.read_csv(FEATURE_DIR / "index.csv")


# -------------------------
# Speaker centering
# -------------------------

X_centered = X.copy()

for speaker in metadata["speaker_key"].unique():
    idx = metadata.index[
        metadata["speaker_key"] == speaker
    ].to_numpy()

    centroid = X[idx].mean(axis=0)
    X_centered[idx] -= centroid


# -------------------------
# PCA for clustering
# -------------------------

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_centered)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)


# -------------------------
# HDBSCAN
# -------------------------

clusterer = HDBSCAN(
    min_cluster_size=10,
    min_samples=5
)

labels = clusterer.fit_predict(X_pca)

metadata["cluster_id"] = labels


# -------------------------
# Cluster statistics
# -------------------------

print()
print("=== CLUSTERS ===")
print(metadata["cluster_id"].value_counts().sort_index())

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

noise = np.sum(labels == -1)

print()
print("Clusters:", n_clusters)
print("Noise:", noise)
print(f"Noise %: {noise / len(labels) * 100:.2f}%")


# -------------------------
# Clustering metrics
# -------------------------

mask = labels != -1

X_eval = X_pca[mask]
labels_eval = labels[mask]

if len(set(labels_eval)) > 1:
    print()
    print("=== CLUSTERING QUALITY ===")

    print(
        "Silhouette:",
        silhouette_score(X_eval, labels_eval)
    )

    print(
        "Davies-Bouldin:",
        davies_bouldin_score(X_eval, labels_eval)
    )

    print(
        "Calinski-Harabasz:",
        calinski_harabasz_score(X_eval, labels_eval)
    )


# -------------------------
# PCA 2D for visualization
# -------------------------

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_centered)


def save_scatter(colors, title, filename):
    plt.figure(figsize=(10, 7))

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=colors,
        s=18,
        alpha=0.75
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# -------------------------
# HDBSCAN clusters
# -------------------------

save_scatter(
    labels,
    "WavLM PCA — HDBSCAN clusters",
    "clusters.png"
)


# -------------------------
# Age
# -------------------------

age_codes = (
    metadata["age"]
    .astype("category")
    .cat.codes
)

save_scatter(
    age_codes,
    "WavLM PCA — Age",
    "age.png"
)


# -------------------------
# Gender
# -------------------------

gender_codes = (
    metadata["gender"]
    .astype("category")
    .cat.codes
)

save_scatter(
    gender_codes,
    "WavLM PCA — Gender",
    "gender.png"
)


# -------------------------
# Speaker
# -------------------------

speaker_codes = (
    metadata["speaker_key"]
    .astype("category")
    .cat.codes
)

save_scatter(
    speaker_codes,
    "WavLM PCA — Speaker",
    "speakers.png"
)


# -------------------------
# PCA explained variance
# -------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(
        1,
        len(pca.explained_variance_ratio_) + 1
    ),
    np.cumsum(
        pca.explained_variance_ratio_
    )
)

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance")

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "pca_explained_variance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# -------------------------
# Save
# -------------------------

metadata.to_csv(
    FEATURE_DIR / "clusters.csv",
    index=False
)

print()
print(f"Plots saved to: {PLOT_DIR}")
print(f"Clusters saved to: {FEATURE_DIR / 'clusters.csv'}")