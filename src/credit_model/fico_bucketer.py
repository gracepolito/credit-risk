
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class BucketResult:
    boundaries: List[float]      # right-closed ascending boundaries, last = +inf
    ratings: np.ndarray          # rating per record (1..K), 1 = best
    counts: List[int]            # records per rating
    pd_per_rating: List[float]   # smoothed PD per rating

def _kmeans_1d_boundaries(fico: np.ndarray, K: int, max_iter: int = 100) -> List[float]:
    x = np.sort(fico.astype(float))
    qs = np.linspace(0, 1, K + 2)[1:-1]
    centers = np.quantile(x, qs)
    for _ in range(max_iter):
        mids = (centers[:-1] + centers[1:]) / 2.0 if K > 1 else np.array([])
        bounds = list(mids) + [float("inf")]
        labels = np.zeros_like(x, dtype=int)
        j = 0
        for i, val in enumerate(x):
            while j < K - 1 and val > bounds[j]:
                j += 1
            labels[i] = j
        new_centers = np.array([x[labels == k].mean() if np.any(labels == k) else centers[k] for k in range(K)])
        if np.allclose(new_centers, centers, atol=1e-6, rtol=0):
            break
        centers = new_centers
    if K == 1:
        return [float("inf")]
    mids = (centers[:-1] + centers[1:]) / 2.0
    return list(mids) + [float("inf")]

def _assign_ratings(fico: np.ndarray, boundaries: List[float]) -> np.ndarray:
    K = len(boundaries)
    idx = np.zeros_like(fico, dtype=int)
    for i, val in enumerate(fico):
        j = 0
        while j < K - 1 and val > boundaries[j]:
            j += 1
        idx[i] = j
    return (K - idx).astype(int)  # 1 = best

def fit_fico_buckets_mse(df: pd.DataFrame, fico_col: str, default_col: str, K: int) -> BucketResult:
    fico = df[fico_col].values
    dflt = df[default_col].values.astype(int)
    boundaries = _kmeans_1d_boundaries(fico, K)
    ratings = _assign_ratings(fico, boundaries)
    counts, pd_per_rating = [], []
    for r in range(1, K+1):
        mask = ratings == r
        n = int(mask.sum())
        k = int(dflt[mask].sum()) if n > 0 else 0
        pd_hat = (k + 0.5) / (n + 1.0) if n > 0 else 0.0
        counts.append(n)
        pd_per_rating.append(pd_hat)
    return BucketResult(boundaries, ratings, counts, pd_per_rating)

def save_boundaries(boundaries: List[float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(boundaries, f)

def load_boundaries(path: str) -> List[float]:
    with open(path, "r") as f:
        return json.load(f)

def apply_boundaries(fico_series: pd.Series, boundaries: List[float]) -> np.ndarray:
    K = len(boundaries)
    out = np.empty(len(fico_series), dtype=int)
    for i, val in enumerate(fico_series.values):
        j = 0
        while j < K - 1 and val > boundaries[j]:
            j += 1
        out[i] = K - j
    return out
