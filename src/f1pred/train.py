"""Training entry-point for the F1 prediction project."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .utils import (
    add_common_args,
    get_logger,
    load_config,
    output_dir_from_cfg,
    set_seed,
    save_artifacts,
    make_group_kfold,
    group_sizes_from_labels,
    timer,
)
from .data_ingest import load_tables
from .build_features import build_feature_matrix
from .evaluate import evaluate_ranking


# ---------------- Model factory -------------------

def _import_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception as e:
        raise RuntimeError(
            "xgboost is required for the default model. Please install it: pip install xgboost"
        ) from e


def _make_model(cfg: Dict[str, Any]):
    xgb = _import_xgb()
    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "xgboost").lower()
    mtype = mcfg.get("type", "ranking").lower()
    params = mcfg.get("params", {}).copy()

    # XGBoost family
    if name == "xgboost":
        if mtype == "ranking":
            Model = xgb.sklearn.XGBRanker
        elif mtype == "regression":
            Model = xgb.sklearn.XGBRegressor
        elif mtype == "classification":
            Model = xgb.sklearn.XGBClassifier
        else:
            raise ValueError(f"Unsupported model type for xgboost: {mtype}")
        return Model(**params)

    # ---- scikit-learn models ----
    if name in {"gradient_boosting", "gbr", "gb"}:
        if mtype == "regression":
            from sklearn.ensemble import GradientBoostingRegressor as Model  # type: ignore
        elif mtype == "classification":
            from sklearn.ensemble import GradientBoostingClassifier as Model  # type: ignore
        else:
            raise ValueError("gradient_boosting supports regression or classification")
        return Model(**params)

    if name in {"linear_regression", "linreg"}:
        if mtype != "regression":
            raise ValueError("linear_regression requires model.type = regression")
        from sklearn.linear_model import LinearRegression as Model  # type: ignore
        return Model(**params)

    if name in {"logistic_regression", "logreg"}:
        if mtype != "classification":
            raise ValueError("logistic_regression requires model.type = classification")
        from sklearn.linear_model import LogisticRegression as Model  # type: ignore
        return Model(**params)

    raise ValueError(f"Unsupported model name: {name}")



def _present_predictions(tables: Dict[str, Any], groups: np.ndarray, y_true: np.ndarray, scores: np.ndarray, out_dir: Path) -> tuple[str, str] | None:
    """
    Build readable per-race leaderboards and save them to CSV + Markdown.

    Assumptions:
    - The training rows align with `tables["results"]` row order. If some rows were filtered,
      we defensively trim to the smallest shared length.
    - `groups` identifies a race-like grouping key returned by build_feature_matrix.
    """
    try:
        import pandas as pd
    except Exception:
        return None

    if "results" not in tables:
        return None

    # Start from results; align lengths defensively
    res = tables["results"].reset_index(drop=True).copy()
    n = min(len(res), len(y_true), len(scores), len(groups))
    if n == 0:
        return None
    res = res.iloc[:n].copy()

    # Attach evaluation columns
    res["actual_pos"] = np.asarray(y_true)[:n]
    res["score"] = np.asarray(scores)[:n]
    res["race_group"] = np.asarray(groups)[:n]

    # Rank within each race/group by score (higher score = better predicted finish)
    res["pred_rank"] = res.groupby("race_group")["score"].rank(ascending=False, method="first").astype(int)
    res["delta"] = res["actual_pos"] - res["pred_rank"]

    # Optional enrich: race name/year, driver name, team
    df = res
    if "races" in tables and "raceId" in df.columns and "raceId" in tables["races"].columns:
        races = tables["races"].copy()
        keep = [c for c in ["raceId", "year", "name", "round"] if c in races.columns]
        races = races[keep].rename(columns={"name": "race_name"})
        df = df.merge(races, on="raceId", how="left")

    if "drivers" in tables and "driverId" in df.columns and "driverId" in tables["drivers"].columns:
        drv = tables["drivers"].copy()
        if "forename" not in drv.columns:
            drv["forename"] = ""
        if "surname" not in drv.columns:
            drv["surname"] = ""
        drv["driver_name"] = (drv["forename"].fillna("") + " " + drv["surname"].fillna("")).str.strip()
        df = df.merge(drv[["driverId", "driver_name"]], on="driverId", how="left")

    if "constructors" in tables and "constructorId" in df.columns and "constructorId" in tables["constructors"].columns:
        cons = tables["constructors"].copy()
        keep = [c for c in ["constructorId", "name"] if c in cons.columns]
        cons = cons[keep].rename(columns={"name": "team"})
        df = df.merge(cons, on="constructorId", how="left")

    # Choose readable columns if they exist
    preferred_cols = [
        "year", "round", "race_name",
        "driver_name", "team",
        "grid", "actual_pos", "pred_rank", "delta", "score",
        "raceId", "driverId", "constructorId"
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        cols = ["race_group", "actual_pos", "pred_rank", "delta", "score"]
    df_out = df[cols].copy()

    # Sort nicely: by season/round then predicted rank if available
    sort_cols = [c for c in ["year", "round", "pred_rank"] if c in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols, ascending=[True] * len(sort_cols))

    # Write to disk
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    md_path = out_dir / "report.md"
    df_out.to_csv(csv_path, index=False)

    # Build a short Markdown leaderboard for the last ~5 races
    lines = ["# Prediction Leaderboards\n"]
    # Figure out grouping key for a readable section header
    if {"year", "round", "race_name"}.issubset(df_out.columns):
        group_keys = ["year", "round", "race_name"]
    elif {"year", "round"}.issubset(df_out.columns):
        group_keys = ["year", "round"]
    elif "raceId" in df_out.columns:
        group_keys = ["raceId"]
    else:
        group_keys = ["race_group"]

    for key, sub in df_out.groupby(group_keys, sort=True):
        # Only keep top-10 by predicted rank if present, else by score desc
        if "pred_rank" in sub.columns:
            sub = sub.sort_values("pred_rank").head(10)
        else:
            sub = sub.sort_values("score", ascending=False).head(10)

        title = f"## Race {key}\n"
        lines.append(title)

        header = ["Pred", "Actual", "Δ", "Driver", "Team"]
        # Fall back safely if columns don't exist
        def getcol(row, name, default=""):
            return row[name] if name in sub.columns else default

        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for _, r in sub.iterrows():
            pred = int(getcol(r, "pred_rank", 0)) if "pred_rank" in sub.columns else ""
            actual = int(getcol(r, "actual_pos", 0)) if "actual_pos" in sub.columns else ""
            delta = int(getcol(r, "delta", 0)) if "delta" in sub.columns else ""
            driver = getcol(r, "driver_name")
            team = getcol(r, "team")
            lines.append(f"| {pred} | {actual} | {delta} | {driver} | {team} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Optional console preview using rich if available
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        console.print(f"[bold green]Saved prediction tables to {csv_path} and {md_path}[/bold green]")

        # Show a quick preview of the most recent race/group
        # Try to get the last group by lexical sort
        last_group = sorted(list(df_out.groupby(group_keys).groups.keys()))[-1]
        preview = df_out.groupby(group_keys).get_group(last_group)
        if "pred_rank" in preview.columns:
            preview = preview.sort_values("pred_rank").head(10)
        else:
            preview = preview.sort_values("score", ascending=False).head(10)

        table = Table(title=f"Top-10 Predictions — {last_group}")
        for h in ["Pred", "Actual", "Δ", "Driver", "Team"]:
            table.add_column(h)
        for _, r in preview.iterrows():
            table.add_row(
                str(int(r["pred_rank"])) if "pred_rank" in preview.columns else "",
                str(int(r["actual_pos"])) if "actual_pos" in preview.columns else "",
                str(int(r["delta"])) if "delta" in preview.columns else "",
                r.get("driver_name", ""),
                r.get("team", "")
            )
        console.print(table)
    except Exception:
        pass

    return str(csv_path), str(md_path)

# ---------------- Training loop -------------------

def train_main(argv: List[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train F1 model")
    add_common_args(parser)
    parser.add_argument("--no-save", action="store_true", help="Do not save model artifacts")
    args = parser.parse_args(argv)

    # load config (single path) and merge CLI overrides expected by utils
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.experiment_name:
        cfg["experiment_name"] = args.experiment_name

    log = get_logger(level=cfg.get("logging", {}).get("level", "INFO"))
    set_seed(cfg.get("seed", 42))

    out_dir = output_dir_from_cfg(cfg)
    log.info(f"Output dir: {out_dir}")

    # 1) Load data
    with timer("load tables"):
        tables = load_tables(cfg)
    for k, v in tables.items():
        log.info(f"Loaded table '{k}': shape={v.shape}")

    # 2) Build features
    with timer("build features"):
        X, y, groups, feat_names, state = build_feature_matrix(tables, cfg, fit=True)
    # Keep original finish positions for evaluation; transform to relevance for ranking training
    y_true = y.copy()
    mtype = cfg.get("model", {}).get("type", "ranking").lower()
    if mtype == "ranking":
        # Convert finish position (1=best) -> relevance (31..0)
        y = np.maximum(0, 32 - y).astype(int)
    elif mtype == "classification":
        target_label = cfg.get("target", {}).get("label", "").lower()
        if target_label in {"winner", "win"}:
            y = (y == 1).astype(int)
    log.info(f"Feature matrix: X={X.shape}, y={y.shape}, groups={len(groups)} features={len(feat_names)}")

    # 3) Cross-validated training
    split_cfg = cfg.get("split", {})
    n_splits = int(split_cfg.get("n_splits", 5))

    early = cfg.get("model", {}).get("early_stopping", {})
    es_rounds = int(early.get("rounds", 50))
    es_metric = early.get("metric", "ndcg@10")

    _ = _import_xgb()  # ensure import works

    # Prepare EarlyStopping callback (works across xgboost versions)
    xgb = _import_xgb()
    callbacks = None
    try:
        # Newer API supports metric_name and save_best
        callbacks = [xgb.callback.EarlyStopping(rounds=es_rounds, metric_name=es_metric, save_best=True)]
    except Exception:
        try:
            # Older API
            callbacks = [xgb.callback.EarlyStopping(rounds=es_rounds)]
        except Exception:
            callbacks = None

    all_preds = np.zeros_like(y, dtype=float)
    fold_metrics: List[Dict[str, float]] = []
    best_model = None
    best_metric = -np.inf

    for fold, (tr_idx, va_idx) in enumerate(make_group_kfold(n_splits, np.asarray(groups))):
        X_tr, y_tr, g_tr = X[tr_idx], y[tr_idx], np.asarray(groups)[tr_idx]
        X_va, y_va, g_va = X[va_idx], y[va_idx], np.asarray(groups)[va_idx]
        # For XGBRanker, samples must be grouped by query with `group` = counts per contiguous group.
        if mtype == "ranking":
            # Stable sort to keep within-group order deterministic
            order_tr = np.argsort(g_tr, kind="mergesort")
            X_tr, y_tr, g_tr = X_tr[order_tr], y_tr[order_tr], g_tr[order_tr]
            order_va = np.argsort(g_va, kind="mergesort")
            X_va, y_va, g_va = X_va[order_va], y_va[order_va], g_va[order_va]
            # Counts per contiguous group in the now-sorted arrays
            group_sizes_tr = np.unique(g_tr, return_counts=True)[1].astype(np.uint32)
            group_sizes_va = np.unique(g_va, return_counts=True)[1].astype(np.uint32)

        model = _make_model(cfg)
        if mtype == "ranking":
            model.set_params(eval_metric=es_metric)
            try:
                model.fit(
                    X_tr,
                    y_tr,
                    group=group_sizes_tr,
                    eval_set=[(X_va, y_va)],
                    eval_group=[group_sizes_va],
                    verbose=False,
                    callbacks=callbacks,
                )
            except TypeError:
                try:
                    model.fit(
                        X_tr,
                        y_tr,
                        group=group_sizes_tr,
                        eval_set=[(X_va, y_va)],
                        eval_group=[group_sizes_va],
                        verbose=False,
                        early_stopping_rounds=es_rounds,
                    )
                except TypeError:
                    # Very old API: neither callbacks nor early_stopping_rounds is supported
                    model.fit(
                        X_tr,
                        y_tr,
                        group=group_sizes_tr,
                        eval_set=[(X_va, y_va)],
                        eval_group=[group_sizes_va],
                        verbose=False,
                    )
            preds = model.predict(X_va)
        elif mtype in {"regression", "classification"}:
            # Detect whether model.fit supports xgboost-style eval_set
            fit_sig = inspect.signature(model.fit)
            has_eval_set = "eval_set" in fit_sig.parameters

            if has_eval_set:
                # Likely an xgboost regressor/classifier
                try:
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                    )
                except TypeError:
                    model.fit(X_tr, y_tr)
            else:
                # Plain scikit-learn estimators
                model.fit(X_tr, y_tr)

            # Produce ranking-friendly scores
            if mtype == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X_va)[:, 1]
                elif hasattr(model, "decision_function"):
                    preds = model.decision_function(X_va)
                else:
                    preds = model.predict(X_va)
            else:
                preds = -model.predict(X_va)
        else:
            raise ValueError(f"Unsupported model type: {mtype}")

        all_preds[va_idx] = preds
        with timer("evaluate fold"):
            report = evaluate_ranking(
                y_true[va_idx],
                preds,
                g_va,
                top_k=tuple(cfg.get("evaluation", {}).get("top_k", [1, 3, 5, 10])),
            )
        fold_metrics.append(report)
        log.info(f"Fold {fold}: {report}")

        score = report.get("ndcg@10", report.get("spearman", -np.inf))
        if score > best_metric:
            best_metric = score
            best_model = model

    # 4) Aggregate CV metrics
    metrics_mean = (
        {k: float(np.mean([fm[k] for fm in fold_metrics if k in fm])) for k in fold_metrics[0].keys()}
        if fold_metrics
        else {}
    )
    log.info(f"CV mean metrics: {metrics_mean}")

    # 5) Refit on all data with best n_estimators if available
    final_model = _make_model(cfg)
    if best_model is not None and hasattr(best_model, "best_iteration") and best_model.best_iteration is not None:
        n_est = int(best_model.best_iteration) + 1
        if n_est > 0:
            final_model.set_params(n_estimators=n_est)
    # Only set eval_metric if supported by the model (e.g., XGBoost, not scikit-learn)
    try:
        params = final_model.get_params()  # type: ignore[attr-defined]
    except Exception:
        params = {}
    if isinstance(params, dict) and "eval_metric" in params:
        final_model.set_params(eval_metric=es_metric)

    if mtype == "ranking":
        # Sort the full dataset by group so sizes are valid
        order_full = np.argsort(groups, kind="mergesort")
        inv_order = np.empty_like(order_full)
        inv_order[order_full] = np.arange(len(order_full))
        X_ord, y_ord, g_ord = X[order_full], y[order_full], groups[order_full]
        full_sizes = np.unique(g_ord, return_counts=True)[1].astype(np.uint32)
        with timer("final fit (ranking)"):
            final_model.fit(X_ord, y_ord, group=full_sizes, verbose=False)
        with timer("final predict (ranking)"):
            final_preds_ord = final_model.predict(X_ord)
        # Map predictions back to original row order
        final_preds = np.empty_like(final_preds_ord)
        final_preds[order_full] = final_preds_ord
    else:
        with timer("final fit"):
            fit_sig = inspect.signature(final_model.fit)
            fit_kwargs = {}
            if "verbose" in fit_sig.parameters:
                fit_kwargs["verbose"] = False
            final_model.fit(X, y, **fit_kwargs)
        with timer("final predict"):
            if mtype == "classification":
                if hasattr(final_model, "predict_proba"):
                    final_preds = final_model.predict_proba(X)[:, 1]
                elif hasattr(final_model, "decision_function"):
                    final_preds = final_model.decision_function(X)
                else:
                    final_preds = final_model.predict(X)
            else:
                final_preds = -final_model.predict(X)

    with timer("evaluate full fit"):
        overall = evaluate_ranking(
            y_true,
            final_preds,
            groups,
            top_k=tuple(cfg.get("evaluation", {}).get("top_k", [1, 3, 5, 10])),
            bootstrap=cfg.get("evaluation", {}).get("bootstrap", {}),
        )
    log.info(f"Full-fit metrics: {overall}")

    # 6) Build human-friendly prediction outputs
    try:
        _present_predictions(tables, np.asarray(groups), np.asarray(y_true), np.asarray(final_preds), out_dir)
    except Exception as e:
        log.warning(f"Could not generate readable predictions: {e}")

    # 7) Save artifacts
    if not args.no_save:
        save_dir = out_dir
        with timer("save artifacts"):
            save_artifacts(cfg, {"models": {"final_model": final_model}, "metrics": overall}, save_dir)
            with open(Path(save_dir) / "cv_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"folds": fold_metrics, "cv_mean": metrics_mean, "full_fit": overall}, f, indent=2)
        log.info(f"Saved model to {save_dir}")

    return {"cv_mean": metrics_mean, "full_fit": overall, "output_dir": str(out_dir)}


if __name__ == "__main__":
    train_main()