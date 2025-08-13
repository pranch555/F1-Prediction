# F1 Race Result Ranking & Prediction

Predict full race finishing order for Formula 1 using a **learning-to-rank** approach with strict **leakage-safe** feature engineering and **grouped cross-validation**. The project turns raw Ergast-style CSVs into features, trains rankers, evaluates per-race metrics (NDCG@k, Spearman, MAE vs **grid baseline**), and exports ready-to-use predictions.

---

## 🔥 Why this project is interesting (for reviewers/recruiters)

- **End-to-end ML system**: ingestion → feature store → CV → training → evaluation → artifacts.
- **Learning-to-Rank**: models optimize ranking quality per race (e.g., LightGBM `lambdarank`).
- **No data leakage**: driver/constructor rolling stats are shifted; CV grouped by race/season.
- **Baselines that matter**: compare against “grid = finish” and season-naive baselines.
- **Reproducible runs**: every run logs config + metrics under `runs/exp/<timestamp>`.
- **Modular code**: clean `src/f1pred/*` with tests and configs that can plug in new features.

---

## 🧱 Repository layout

├─ configs/
│ ├─ base.yml # shared defaults (paths, features, CV, common params)
│ ├─ rexp_after_quali.yml # run config when qualifying grid is known
│ └─ rexp_before_quali.yml # run config when only practice/priors are known
├─ data/
│ ├─ raw/ # drop CSVs here (see “Data” section)
│ ├─ interim/ # cached/intermediate parquet files
│ └─ processed/ # model-ready features per race
├─ runs/
│ └─ exp// # config.yaml, metrics.json, cv_metrics.json, final_model.joblib
├─ src/
│ └─ f1pred/
│ ├─ init.py
│ ├─ data_ingest.py
│ ├─ build_features.py
│ ├─ train.py # python -m f1pred.train -c configs/base.yml
│ ├─ predict.py # python -m f1pred.predict -c ...
│ ├─ evaluate.py
│ └─ utils.py
├─ tests/
│ ├─ test_no_leakage.py
│ └─ test_schemas.py
└─ README.md

📁 Data

Place Ergast-style CSVs into data/raw/. Files commonly used here:
• circuits.csv
• constructors.csv
• constructor_results.csv
• constructor_standings.csv
• drivers.csv
• driver_standings.csv
• lap_times.csv
• pit_stops.csv
• qualifying.csv
• races.csv
• results.csv

▶️ Train

# Example: default experiment

python -m f1pred.train -c configs/base.yml

# After-quali setup (grid included)

python -m f1pred.train -c configs/rexp_after_quali.yml

# Before-quali setup (no grid features)

python -m f1pred.train -c configs/rexp_before_quali.yml

Outputs (per run) in runs/exp/<timestamp>/:
• config.yaml (frozen copy)
• final_model.joblib
• metrics.json (global metrics)
• cv_metrics.json (per fold)
• optional: feature_importance.csv, oof_predictions.parquet

✅ Tests
pytest -q
Typical guards included:
• No leakage in rolling features (first race per driver has NaNs on rolling stats).
• Per-race ranking sanity (each race should have multiple unique pred_rank values).
• Schema checks for required columns & dtypes.
