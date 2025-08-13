from __future__ import annotations
import json, logging, random, time, pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import yaml

# Optional deps
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # some utils will guard for this

# ---------------- Repro & Logging ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        pass

def get_logger(name: str = "f1pred", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(h)
    return logger

@contextmanager
def timer(msg: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{msg}: {dt:.3f}s")

def timeit(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        print(f"{fn.__name__} took {dt:.3f}s")
        return out
    return wrapper

# ---------------- CLI & Config ----------------

def add_common_args(ap) -> None:
    """Attach common training/eval args to an argparse.ArgumentParser."""
    ap.add_argument("-c", "--config", required=False, help="Path to YAML config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_root", type=str, default="runs")
    ap.add_argument("--experiment_name", type=str, default="exp")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=0)

def load_config(path_or_none) -> dict:
    """Load a YAML config into a dict (or {} if None)."""
    if not path_or_none:
        return {}
    p = Path(path_or_none)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping.")
    return data

def output_dir_from_cfg(cfg: dict, create: bool = True) -> Path:
    root = Path(cfg.get("output_root", "runs"))
    exp  = cfg.get("experiment_name", "exp")
    ts   = time.strftime("%Y%m%d-%H%M%S")
    out = root / exp / ts
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out

def save_artifacts(cfg: dict, artifacts: dict, outdir: Path) -> None:
    """Save config, metrics, and any simple artifacts to disk."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # config
    with (outdir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # metrics
    metrics = artifacts.get("metrics")
    if metrics is not None:
        with (outdir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    # models or arrays
    if "models" in artifacts and isinstance(artifacts["models"], dict):
        for name, obj in artifacts["models"].items():
            save_joblib(obj, outdir / f"{name}.joblib")

# ---------------- Atomic IO ----------------

def save_joblib(obj: Any, path: Path | str) -> None:
    path = Path(path)
    try:
        import joblib  # type: ignore
        joblib.dump(obj, path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(obj, f)

def load_joblib(path: Path | str) -> Any:
    path = Path(path)
    try:
        import joblib  # type: ignore
        return joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)

def save_json(data: Any, path: Path | str) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(df, path: Path | str, index: bool = False) -> None:
    if pd is None:
        raise RuntimeError("pandas not available for save_csv")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def read_csv(path: Path | str):
    if pd is None:
        raise RuntimeError("pandas not available for read_csv")
    return pd.read_csv(path)

# ---------------- CV & Scoring ----------------

def make_group_kfold(n_splits: int, groups, shuffle: bool = False, random_state: int | None = None):
    """Simple leakage-safe group CV splitter; yields (train_idx, val_idx)."""
    groups = np.asarray(groups)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if groups.size == 0:
        raise ValueError("groups must be non-empty")
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.arange(len(uniq))
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(order)
    fold_of_group = np.empty(len(uniq), dtype=int)
    for i, gidx in enumerate(order):
        fold_of_group[gidx] = i % n_splits
    for f in range(n_splits):
        val_groups = uniq[fold_of_group == f]
        val_mask = np.isin(groups, val_groups)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        yield train_idx, val_idx

def scoring_classification(y_true: Sequence[int], y_pred: Sequence[int], y_proba: Sequence[float] | None = None) -> Dict[str, float]:
    """Return dict of common binary classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, float] = {}
    # Try sklearn
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # type: ignore
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        if y_proba is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_proba)))
        return out
    except Exception:
        # Fallback implementations
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        out["accuracy"] = float((tp + tn) / max(1, tp + fp + fn + tn))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        out["precision"] = float(prec)
        out["recall"] = float(rec)
        out["f1"] = float(0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        # roc_auc omitted in fallback
        return out

def group_sizes_from_labels(labels: Sequence[Any]) -> Dict[Any, int]:
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    return {k: int(v) for k, v in zip(uniq, counts)}

def summarize_binary_classification(y_true: Sequence[int], y_pred: Sequence[int], y_proba: Sequence[float] | None = None) -> Dict[str, float]:
    """Thin wrapper around scoring_classification for convenience."""
    return scoring_classification(y_true, y_pred, y_proba)

# ---------------- DataFrame helpers (safe & generic) ----------------

def validate_columns(df, required: Iterable[str]) -> None:
    if pd is None:
        raise RuntimeError("pandas not available for validate_columns")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def reduce_mem_usage(df):
    """Downcast numeric columns to save memory without changing values."""
    if pd is None:
        raise RuntimeError("pandas not available for reduce_mem_usage")
    for col in df.select_dtypes(include=["int", "int64", "uint64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df

def make_race_key(df, cols: List[str] | None = None, new_col: str = "race_key"):
    """Create a stable race key based on common F1 columns (season+round if present)."""
    if pd is None:
        raise RuntimeError("pandas not available for make_race_key")
    if cols is None:
        if all(c in df.columns for c in ["season", "round"]):
            cols = ["season", "round"]
        elif "raceId" in df.columns:
            cols = ["raceId"]
        elif all(c in df.columns for c in ["date", "circuitId"]):
            cols = ["date", "circuitId"]
        else:
            raise KeyError("Cannot infer race key; provide cols.")
    df[new_col] = df[cols].astype(str).agg("-".join, axis=1)
    return df

def add_prev_wins(df, group_col: str = "driverId", pos_col: str = "position", new_col: str = "prev_wins"):
    """Add cumulative wins up to previous race per driver."""
    if pd is None:
        raise RuntimeError("pandas not available for add_prev_wins")
    if group_col not in df.columns or pos_col not in df.columns:
        raise KeyError(f"Required columns not found: {group_col}, {pos_col}")
    is_win = (df[pos_col].astype(str) == "1").astype(int)
    df[new_col] = (
        is_win.groupby(df[group_col]).cumsum().shift(fill_value=0).astype(int)
    )
    return df

def chronological_sort(df):
    """Sort by date if available, else by (season, round), else no-op."""
    if pd is None:
        raise RuntimeError("pandas not available for chronological_sort")
    if "date" in df.columns:
        return df.sort_values("date")
    if all(c in df.columns for c in ["season", "round"]):
        return df.sort_values(["season", "round"])
    return df