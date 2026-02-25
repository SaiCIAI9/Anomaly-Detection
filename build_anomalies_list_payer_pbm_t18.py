"""
Build Anomalies_List_Payer_PBM_T18.csv for Streamlit anomaly app.
Joins GAN T18 results (all groups) with full ML dataset so LLM has all features.
Entity = Payer-PBM (PRCSN_PAYER_ENT_NM, PRCSN_PBM_VENDOR).
Target period = T18 (Oct 2024).
Runs add_sop_nvs_payer_pbm_t18.py after merge to add SOP 1-4 and NVS flags.
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAN_RESULTS = os.path.join(SCRIPT_DIR, 'gan_payer_pbm_t18_results', 'GAN_T18_RESULTS_ALL_GROUPS.csv')
ML_DATASET = os.path.join(SCRIPT_DIR, 'ML_DATASET_PAYER_PBM_wide_FULL.csv')
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'Anomalies_List_Payer_PBM_T18.csv')

def main():
    if not os.path.exists(GAN_RESULTS):
        print(f'ERROR: GAN results not found: {GAN_RESULTS}')
        print('Run: python gan_payer_pbm_t17_predict_t18.py --epochs 50')
        return 1
    if not os.path.exists(ML_DATASET):
        print(f'ERROR: ML dataset not found: {ML_DATASET}')
        return 1

    gan = pd.read_csv(GAN_RESULTS, low_memory=False)
    ml = pd.read_csv(ML_DATASET, low_memory=False)

    gan_cols = [c for c in gan.columns if c not in ml.columns or c in ['PRCSN_PAYER_ENT_NM', 'PRCSN_PBM_VENDOR']]
    merge_cols = ['PRCSN_PAYER_ENT_NM', 'PRCSN_PBM_VENDOR']
    gan_merge = gan[merge_cols + [c for c in gan_cols if c not in merge_cols]].copy()

    merged = ml.merge(gan_merge, on=merge_cols, how='inner', suffixes=('', '_gan'))
    dup = [c for c in merged.columns if c.endswith('_gan')]
    merged = merged.drop(columns=dup, errors='ignore')

    merged['GRANULARITY'] = merged['PRCSN_PAYER_ENT_NM'] + ' | ' + merged['PRCSN_PBM_VENDOR']
    merged['GRANULARITY_LEVEL'] = 'Payer-PBM'

    merged.to_csv(OUTPUT_CSV, index=False)
    n_anomalies = (merged['Anomaly_Flag'] == 1).sum() if 'Anomaly_Flag' in merged.columns else 0
    print(f'Saved: {OUTPUT_CSV}')
    print(f'  Rows: {len(merged)}, Anomalies (Anomaly_Flag=1): {n_anomalies}')

    # Add SOP and NVS flags (run add_sop_nvs_payer_pbm_t18.py)
    add_sop = os.path.join(SCRIPT_DIR, 'add_sop_nvs_payer_pbm_t18.py')
    if os.path.exists(add_sop):
        import subprocess
        r = subprocess.run([sys.executable, add_sop], cwd=SCRIPT_DIR, capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout)
        else:
            print(f'  Note: add_sop_nvs failed: {r.stderr or r.stdout}')
    return 0

if __name__ == '__main__':
    exit(main())
