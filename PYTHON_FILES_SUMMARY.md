# Python Files – One-Line Summary (Share with Teammate)

Python scripts in this project, grouped by area, with a single-line description. The repo has **165+** `.py` files; this doc covers the main ones. For a full list, run `dir /b *.py` (Windows) or `ls *.py` (Mac/Linux) in the project folder.

---

## Anomaly & GAN pipeline (core)

| File | Summary |
|------|--------|
| `streamlit_anomaly_app.py` | Streamlit dashboard: load Anomalies_List.csv, filter by priority, select anomaly, get LLM summary; password gate + editable prompt (Supabase). |
| `build_ml_dataset_4_granularities.py` | Builds ML datasets at Payer, PBM, State, Channel with T1–T25 columns, aggregates, volatility, relationship flags. |
| `train_gan_all_features_t23.py` | Trains GAN on all features up to T23 (no T24/T25); saves generator, discriminator, scalers in `gan_all_features_t23_models/`. |
| `test_gan_t24.py` | Loads trained GAN, scores entities on T24, computes extrapolation vs actual, anomaly flags (threshold, Z-score, spike); outputs Anomalies_List-style CSV. |
| `interpret_gan_t24_results.py` | Reads GAN T24 test CSVs and summarizes discriminator scores and error distributions for tuning. |
| `train_gan_t1_t23.py` | GAN training using T1–T23 time series from ML dataset (TRx, NBRx, NRx). |
| `train_gan_model.py` | Trains GAN on REXULTI dataset using T1–T24 features to predict T25_TRx. |
| `train_gan_model_rexulti.py` | Trains GAN on REXULTI dataset (loads GAN_TRAINING_DATASET_REXULTI.csv). |
| `train_gan_rexulti.py` | Trains GAN on REXULTI with T1–T24 features (all metrics). |
| `build_gan_dataset_rexulti.py` | Builds GAN training dataset for REXULTI only, 6 granularities, T1–T25 wide format. |
| `build_gan_training_dataset_comprehensive.py` | Builds comprehensive GAN training dataset (all entities/levels). |
| `predict_t25_all_data.py` | Predicts T25_TRx for all data (REXULTI + VRAYLAR) using trained GAN. |
| `create_final_anomaly_dataset.py` | Creates final anomaly dataset with TRx-focused rows and multiple anomaly flags/explanations. |
| `create_final_anomaly_dataset_all_entities.py` | Same as above for all entities. |
| `create_comprehensive_final_anomaly_dataset_all_entities.py` | Comprehensive final anomaly dataset across all entities. |
| `create_gan_ready_dataset.py` | Prepares dataset in GAN-ready format. |
| `create_comprehensive_gan_dataset.py` | Builds comprehensive GAN dataset. |
| `create_comprehensive_gan_dataset_all_entities.py` | Same for all entities. |
| `create_trx_focused_dataset.py` | Builds TRx-focused dataset for GAN/anomaly. |
| `create_all_entities_trx_dataset.py` | TRx dataset for all entities. |
| `train_context_aware_trx_gan.py` | Trains context-aware TRx GAN. |
| `train_trx_gan_with_context.py` | Trains TRx GAN with context. |
| `train_trx_gan_all_entities.py` | Trains TRx GAN for all entities. |
| `train_gan_models_by_granularity.py` | Trains separate GAN models per granularity. |
| `analyze_gan_anomaly_issues.py` | Analyzes GAN anomaly detection issues. |
| `analyze_context_aware_anomalies.py` | Analyzes context-aware anomalies. |
| `analyze_trx_anomalies.py` | Analyzes TRx anomalies. |
| `fix_gan_anomaly_detection.py` | Fixes/tunes GAN anomaly detection logic. |
| `setup_gan_environment.py` | Sets up GAN environment/dependencies. |

---

## Descriptive analytics & EDA

| File | Summary |
|------|--------|
| `build_descriptive_foundation.py` | Builds descriptive analytics foundation. |
| `build_descriptive_foundation_steps1_5.py` | Descriptive foundation steps 1–5. |
| `build_descriptive_counting_engine.py` | Counts records by all descriptive attributes and combinations (Phase 1). |
| `detailed_descriptive_analysis.py` | Detailed descriptive analysis and counting verification. |
| `prepare_descriptive_data.py` | Prepares data for descriptive analysis. |
| `prepare_descriptive_data_fixed.py` | Fixed version of descriptive data prep. |
| `build_complete_descriptive_analytics.py` | Complete descriptive analytics pipeline. |
| `descriptive_analysis_by_brand_with_cutoffs.py` | Descriptive analysis by brand (REXULTI vs VRAYLAR) with cutoff comparison. |
| `descriptive_analysis_full_dataset.py` | Descriptive analysis on full dataset. |
| `track_descriptive_loss_on_volume_filter.py` | Tracks descriptive “loss” when applying volume filter. |
| `build_geoff_descriptive_analytics.py` | Geoff’s descriptive analytics engine (observations, if-then, baselines). |
| `create_descriptive_html_report.py` | Generates HTML descriptive report. |
| `create_descriptive_html_report_steps1_5.py` | HTML report for descriptive steps 1–5. |
| `create_descriptive_insights_report.py` | Descriptive insights report (HTML). |
| `create_comprehensive_descriptive_report.py` | Comprehensive descriptive analysis report (what we count, how, why). |
| `create_ceo_descriptive_report.py` | CEO-friendly descriptive analytics HTML report. |
| `create_geoff_descriptive_html_report.py` | HTML report for Geoff’s descriptive analytics. |
| `create_ceo_variability_report.py` | CEO variability report (HTML). |
| `create_ceo_baselines_slide.py` | CEO baselines slide (HTML). |
| `eda_comprehensive_report.py` | EDA comprehensive report. |
| `eda_for_noise_filtering.py` | EDA for noise filtering. |
| `create_eda_elbow_detection.py` | EDA elbow-curve detection and slides. |
| `create_eda_executive_slides.py` | EDA executive slides. |
| `create_eda_visual_slides.py` | EDA visual slides. |
| `create_elbow_curves_html.py` | Elbow curves HTML. |
| `elbow_curve_noise_filtering.py` | Elbow-curve-based noise filtering. |
| `find_practical_cutoffs.py` | Finds practical cutoffs from EDA. |
| `column_variability_analysis.py` | Column-level variability analysis. |
| `understand_data_grain.py` | Identifies data grain and descriptive columns. |
| `identify_data_grain.py` | Identifies data grain (descriptive vs metrics). |
| `check_eda_approach.py` | Checks EDA approach. |
| `check_distribution_changes.py` | Checks distribution changes. |

---

## Step-based KPI & state analysis

| File | Summary |
|------|--------|
| `step1_kpi_boxes.py` | Step 1: KPI boxes (counts, baselines). |
| `step1_kpi_boxes_fixed.py` | Fixed Step 1 KPI boxes. |
| `step1_kpi_boxes_with_hcp.py` | Step 1 KPI boxes including HCP metrics. |
| `step2_avg_unique_values.py` | Step 2: Average unique values. |
| `step3_state_analysis_with_ratios.py` | Step 3: State-level analysis with ratios. |
| `step3_unique_metrics_by_state.py` | Step 3: Unique metrics by state. |
| `step4_baselines_by_state.py` | Step 4: Baselines by state. |
| `create_state_analysis_html_report.py` | State analysis HTML report. |
| `create_geo_charts_simple.py` | Simple geographic charts. |
| `create_geo_charts_ppt.py` | Geo charts for PowerPoint. |
| `create_geo_charts_with_ppt.py` | Geo charts with PPT export. |
| `create_geo_tables_html.py` | Geographic tables HTML report. |

---

## If–then rules & association rules

| File | Summary |
|------|--------|
| `build_if_then_rules.py` | Builds if–then rules from data. |
| `build_if_then_rules_enhanced.py` | Enhanced if–then rules. |
| `build_if_then_rules_fast.py` | Fast if–then rules build. |
| `build_if_then_rules_comprehensive.py` | Comprehensive if–then rules. |
| `phase3_association_rules.py` | Phase 3: Association rules (important combinations). |
| `phase3_metrics_association_rules.py` | Phase 3 metrics-based association rules. |
| `phase3_metrics_association_rules_fast.py` | Fast Phase 3 metrics association rules. |
| `create_phase3_html_report.py` | Phase 3 HTML report. |
| `create_top10_rules_html_report.py` | Top 10 rules HTML report. |
| `analyze_top_rules_with_real_data.py` | Analyzes top rules with real data. |
| `count_rules.py` | Counts discovered rules. |

---

## Pattern builder & selection

| File | Summary |
|------|--------|
| `pattern_builder_engine_complete.py` | Pattern builder: filters relationships + rules → top critical patterns. |
| `pattern_builder_engine_all_patterns.py` | Pattern builder over all patterns (158K relationships, 48K rules). |
| `pattern_builder_engine_test.py` | Quick test: first 1000 patterns only. |
| `statistical_pattern_selection.py` | Statistical pattern selection. |
| `analyze_pattern_selection_evidence.py` | Analyzes evidence for pattern selection. |
| `analyze_all_12_patterns_evidence.py` | Analyzes evidence for all 12 patterns. |
| `build_behavioral_patterns_rexulti.py` | Builds behavioral patterns for REXULTI. |
| `build_drill_down_analysis_rexulti.py` | Drill-down analysis for REXULTI. |
| `analyze_metrics_patterns_optimized.py` | Optimized metrics pattern analysis. |

---

## Data masking & prep

| File | Summary |
|------|--------|
| `mask_gapd_data.py` | Masks GAPD_Data.csv (payer, PBM, brand, etc.). |
| `mask_otsuka_data.py` | Masks OTSUKA_MASKED_DATA.xlsx (payer, PBM names, etc.). |
| `mask_otsuka_data_auto.py` | Auto-masks OTSUKA file without prompts. |
| `mask_all_grains_final.py` | Auto-masks ALL_GRAINS_FINAL.csv and creates mapping. |
| `create_masked_dataset.py` | Creates masked dataset. |
| `load_payer_pbm_mapping.py` | Loads payer–PBM mapping from Excel (both brands). |

---

## Cutoffs, baselines, comparisons

| File | Summary |
|------|--------|
| `build_baselines_for_ratios.py` | Builds baselines for ratio metrics. |
| `create_simple_cutoff_table.py` | Table: original vs after-cutoff numbers. |
| `create_brand_cutoff_comparison_report.py` | Visual report: descriptive by brand with cutoff comparison. |
| `explain_payer_pbm_calculation.py` | Explains how payer+PBM combinations are calculated. |
| `explain_payer_changes.py` | Explains payer-level changes. |
| `explain_stable_segment_spike.py` | Explains stable-segment spike. |

---

## Reports, slides, HTML

| File | Summary |
|------|--------|
| `create_next_steps_slide.py` | Next-steps slide after EDA cutoffs. |
| `create_trend_examples_html.py` | Trend examples HTML. |
| `root_cause_analyzer_complete.py` | Root cause analyzer (complete). |

---

## Checks, validation, exploration

| File | Summary |
|------|--------|
| `check_gapd_raw_data.py` | Checks GAPD raw data structure. |
| `check_entire_gapd_data.py` | Checks entire GAPD data. |
| `check_data_structure.py` | Checks data structure. |
| `check_data_sources.py` | Checks data sources. |
| `check_dataset.py` | Validates dataset. |
| `check_pbm_metrics.py` | Checks PBM metrics. |
| `check_unique_counts.py` | Checks unique counts. |
| `check_composite_relationships.py` | Checks composite relationships. |
| `check_probability_scores.py` | Checks probability scores. |
| `assess_data_format.py` | Assesses data format. |
| `validate_relationship_findings.py` | Validates relationship findings. |
| `comprehensive_relationship_analysis.py` | Comprehensive relationship analysis. |
| `comprehensive_relationship_analysis_incremental.py` | Incremental version. |
| `explore_exported_data.py` | Explores exported data. |
| `analyze_exported_data_with_hcp.py` | Analyzes exported data with HCP. |
| `analyze_grain_with_patient_hcp.py` | Analyzes grain with patient/HCP. |
| `clarify_patient_hcp_activity.py` | Clarifies patient/HCP activity. |

---

## Other / utilities

| File | Summary |
|------|--------|
| `calculate_total_combinations.py` | Calculates total combinations. |
| `apply_business_rules.py` | Applies business rules to data. |
| `parthenon_data_export.py` | Parthenon data export. |
| `trace_data_reduction.py` | Traces data reduction. |
| `investigate_data_values.py` | Investigates data values. |
| `test_data_preparation.py` | Tests data preparation. |
| `test_instability_detection.py` | Tests instability detection. |
| `test_instability_on_real_data.py` | Tests instability on real data. |
| `statistical_validation_gan_anomalies.py` | Statistical validation of GAN anomalies. |
| `streamlit_gapd_descriptive_app.py` | Streamlit app for GAPD descriptive analytics. |

---

## Quick reference – “What do I run?”

- **Anomaly dashboard (local):** `streamlit run streamlit_anomaly_app.py`
- **Build ML dataset (full):** `python build_ml_dataset_4_granularities.py --test`
- **Train GAN (all features, T23):** `python train_gan_all_features_t23.py` (uses ML dataset)
- **Test GAN on T24:** `python test_gan_t24.py` (produces Anomalies_List-style CSV for the app)
- **Mask data:** `mask_gapd_data.py` / `mask_otsuka_data.py` / `mask_all_grains_final.py` as needed

If you want this as a flat list (no categories) or as a CSV for a spreadsheet, say which format you prefer.
