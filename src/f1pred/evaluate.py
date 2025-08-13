"""
Evaluation utilities for ranking/regression of race results.

We compute metrics per race (group) and then macro-average across races to avoid
large fields dominating small ones.
"""
from __future__ import annotations

from typing import Dict, Iterable, List
import numpy as np
import pandas as pd


def _group_indices(groups: Iterable) -> List[np.ndarray]:
    """Return index arrays for each unique group in order of appearance."""
    groups = pd.Series(groups)
    ids = []
    for _, idx in groups.groupby(groups).groups.items():
        ids.append(np.asarray(list(idx)))
    return ids


def _predicted_positions(scores: np.ndarray) -> np.ndarray:
    """Convert model scores to predicted ranks (1 = best). Higher score is better."""
    order = np.argsort(-scores, kind="mergesort")  # stable sort
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _ndcg_at_k(y_true_pos: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Compute NDCG@k for one race.

    Relevance from position: rel = (max_pos + 1 - position) so P1 has highest gain.
    """
    n = len(y_true_pos)
    if n == 0:
        return np.nan
    k = min(k, n)
    rel = (n + 1 - y_true_pos)

    # DCG
    order = np.argsort(-y_scores, kind="mergesort")[:k]
    gains = rel[order] / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()

    # Ideal DCG
    ideal = np.sort(rel)[::-1][:k] / np.log2(np.arange(2, k + 2))
    idcg = ideal.sum() or 1.0
    return float(dcg / idcg)


def _map_at_k(y_true_pos: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """MAP@k by treating the top-k finishers as relevant."""
    n = len(y_true_pos)
    k = min(k, n)
    relevant = set(np.flatnonzero(y_true_pos <= k))
    if not relevant:
        return np.nan

    order = np.argsort(-y_scores, kind="mergesort")[:k]
    hits = 0
    precisions = []
    for i, idx in enumerate(order, start=1):
        if idx in relevant:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def _spearman(y_true_pos: np.ndarray, y_scores: np.ndarray) -> float:
    """Spearman rank correlation between true and predicted ranks (per race)."""
    pred_ranks = _predicted_positions(y_scores)
    true_ranks = y_true_pos.astype(float)

    def _pearson(a, b):
        a = a.astype(float)
        b = b.astype(float)
        a = a - a.mean()
        b = b - b.mean()
        denom = (a.std(ddof=0) * b.std(ddof=0)) or 1.0
        return float((a * b).mean() / denom)

    return _pearson(pd.Series(true_ranks).rank().to_numpy(), pd.Series(pred_ranks).rank().to_numpy())


def _rmse_positions(y_true_pos: np.ndarray, y_scores: np.ndarray) -> float:
    pred_pos = _predicted_positions(y_scores).astype(float)
    true_pos = y_true_pos.astype(float)
    return float(np.sqrt(np.mean((pred_pos - true_pos) ** 2)))


def evaluate_ranking(
    y_true: Iterable[float],
    y_scores: Iterable[float],
    groups: Iterable,
    *,
    metrics: Iterable[str] = ("ndcg@10", "map@10", "spearman", "rmse"),
    top_k: Iterable[int] = (1, 3, 5, 10),
    bootstrap: Dict = None,
) -> Dict[str, float]:
    """Evaluate predictions with group-wise ranking metrics."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_scores = np.asarray(list(y_scores), dtype=float)
    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")

    group_ids = _group_indices(groups)

    def _compute() -> Dict[str, float]:
        out: Dict[str, float] = {}
        ndcgs = {k: [] for k in top_k}
        maps = {k: [] for k in top_k}
        spears = []
        rmses = []
        for idx in group_ids:
            yt = y_true[idx]
            ys = y_scores[idx]
            if len(yt) == 0:
                continue
            for k in top_k:
                ndcgs[k].append(_ndcg_at_k(yt, ys, k))
                maps[k].append(_map_at_k(yt, ys, k))
            spears.append(_spearman(yt, ys))
            rmses.append(_rmse_positions(yt, ys))
        for k in top_k:
            out[f"ndcg@{k}"] = float(np.nanmean(ndcgs[k]))
            out[f"map@{k}"] = float(np.nanmean(maps[k]))
        out["spearman"] = float(np.nanmean(spears))
        out["rmse"] = float(np.nanmean(rmses))
        return out

    if not bootstrap or not bootstrap.get("enabled", False):
        return _compute()

    # Bootstrap over races (groups)
    rng = np.random.default_rng(int(bootstrap.get("seed", 42)))
    n = len(group_ids)
    b = int(bootstrap.get("n_samples", 1000))

    samples = []
    for _ in range(b):
        # Sample group indices with replacement
        sample_ids = [group_ids[i] for i in rng.integers(0, n, size=n)]
        idxs = np.concatenate(sample_ids)
        # Build pseudo group labels to keep macro-avg semantics
        pseudo_groups = np.concatenate([np.repeat(j, len(g)) for j, g in enumerate(sample_ids)])
        metrics_sample = evaluate_ranking(
            y_true[idxs], y_scores[idxs], groups=pseudo_groups, metrics=metrics, top_k=top_k, bootstrap=None
        )
        samples.append(metrics_sample)

    keys = samples[0].keys() if samples else []
    agg: Dict[str, float] = _compute()
    for k in keys:
        arr = np.array([s[k] for s in samples], dtype=float)
        agg[f"{k}_mean"] = float(np.nanmean(arr))
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        agg[f"{k}_lo"] = float(lo)
        agg[f"{k}_hi"] = float(hi)
    return agg