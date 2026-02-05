# Core Python Files to Share (6 files)

These are the main files: **2 for the model**, **3 for initial EDA**, **1 Streamlit app**.

---

## 2 – Model

| File | One-line summary |
|------|------------------|
| `train_gan_all_features_t23.py` | Trains the GAN on all features up to T23; saves generator, discriminator, and scalers in `gan_all_features_t23_models/`. |
| `test_gan_t24.py` | Loads the trained GAN, scores entities on T24, computes anomaly flags (threshold, Z-score, spike); outputs the CSV used by the app (e.g. Anomalies_List). |

---

## 3 – Initial EDA

| File | One-line summary |
|------|------------------|
| `build_ml_dataset_4_granularities.py` | Builds the ML dataset at Payer, PBM, State, Channel with T1–T25 columns, aggregates, and relationship flags (feeds the GAN). |
| `eda_comprehensive_report.py` | Runs comprehensive EDA and produces the EDA report. |
| `find_practical_cutoffs.py` | Finds practical volume/cutoff thresholds from the EDA (e.g. for filtering). |

*If your “3 EDA” files are different (e.g. step1_kpi_boxes, descriptive counting, etc.), we can swap these names.*

---

## 1 – Streamlit app

| File | One-line summary |
|------|------------------|
| `streamlit_anomaly_app.py` | Streamlit app: loads Anomalies_List.csv, filters by priority, select anomaly → LLM summary; password gate and editable prompt (Supabase). |

---

## Run order (for a new teammate)

1. **EDA:** `build_ml_dataset_4_granularities.py` (e.g. `--test` for full data) → then `eda_comprehensive_report.py` and `find_practical_cutoffs.py` as needed.
2. **Model:** `train_gan_all_features_t23.py` → then `test_gan_t24.py` to create the anomaly CSV.
3. **App:** `streamlit run streamlit_anomaly_app.py` (with Anomalies_List.csv in the same folder).
