"""
Add SOP 1-4 and NVS (Novartis) flags to payer-PBM T18 GAN results.
- SOP1: Same-granularity (NBRx/TRx ratio, volume change above DT)
- SOP2: Parent-child (entity vs national)
- SOP3: Sibling (similar directional movement)
- SOP4: Longitudinal (pattern recurrence, seasonal consistency)
- NVS: Trend Break (+/-3SD), Isolation Forest, Local Outlier Factor

Run after gan_payer_pbm_t17_predict_t18.py and build_anomalies_list_payer_pbm_t18.py.
Output: Anomalies_List_Payer_PBM_T18_WITH_SOP_NVS.csv
"""
import os
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANOMALIES_PATH = os.path.join(SCRIPT_DIR, 'Anomalies_List_Payer_PBM_T18.csv')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'Anomalies_List_Payer_PBM_T18_WITH_SOP_NVS.csv')
TARGET = 18
SD_MULTIPLIER = 3


def _ensure_float(df, col):
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors='coerce').fillna(0).values


def _window_mean(df, metric_prefix, start, length):
    cols = [f'{metric_prefix}_T{i}' for i in range(start, start + length) if f'{metric_prefix}_T{i}' in df.columns]
    if not cols:
        return None
    return df[cols].astype(float).mean(axis=1).values


def _dynamic_threshold(z_values, percentile=90):
    abs_z = np.abs(np.asarray(z_values, dtype=float))
    abs_z = abs_z[np.isfinite(abs_z)]
    if len(abs_z) == 0:
        return 2.0
    return np.percentile(abs_z, min(percentile, 99))


# --- NVS (Novartis) ---
def novartis_statistical_flag(df):
    """+/-3 SD: flag when TRx_T18 crosses Mean_12M +/- 3*Std_12M."""
    mean_col, std_col, actual_col = 'TRx_Mean_12M', 'TRx_Std_12M', f'TRx_T{TARGET}'
    if mean_col not in df.columns or std_col not in df.columns or actual_col not in df.columns:
        df['Novartis_Trend_Break_Flag'] = 0
        df['Novartis_Direction'] = 'None'
        return df
    mean_ = _ensure_float(df, mean_col)
    std_ = _ensure_float(df, std_col)
    std_ = np.where(std_ <= 0, np.nan, std_)
    actual = _ensure_float(df, actual_col)
    lower = mean_ - SD_MULTIPLIER * np.nan_to_num(std_, nan=1e-6)
    upper = mean_ + SD_MULTIPLIER * np.nan_to_num(std_, nan=1e-6)
    below = actual < lower
    above = actual > upper
    df['Novartis_Trend_Break_Flag'] = (below | above).astype(int)
    df['Novartis_Direction'] = 'None'
    df.loc[above, 'Novartis_Direction'] = 'High'
    df.loc[below, 'Novartis_Direction'] = 'Low'
    return df


def novartis_ml_flags(df):
    """IF and LOF on TRx T1..T18."""
    trx_cols = [f'TRx_T{i}' for i in range(1, TARGET + 1) if f'TRx_T{i}' in df.columns]
    if len(trx_cols) < 2:
        df['Novartis_IF_Flag'] = 0
        df['Novartis_LOF_Flag'] = 0
        return df
    X = df[trx_cols].astype(float).fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if not SKLEARN_AVAILABLE or len(X) < 4:
        df['Novartis_IF_Flag'] = 0
        df['Novartis_LOF_Flag'] = 0
        return df
    n = len(X)
    iso = IsolationForest(contamination=min(0.2, max(0.05, 5 / n)), random_state=42)
    df['Novartis_IF_Flag'] = (iso.fit_predict(X) == -1).astype(int)
    k = min(5, max(2, n // 2))
    lof = LocalOutlierFactor(n_neighbors=k, contamination=min(0.2, max(0.05, 5 / n)))
    df['Novartis_LOF_Flag'] = (lof.fit_predict(X) == -1).astype(int)
    return df


def novartis_reason(row):
    parts = []
    if row.get('Novartis_Trend_Break_Flag') == 1:
        parts.append(f"TrendBreak_{row.get('Novartis_Direction', '')}(+/-{SD_MULTIPLIER}SD)")
    if row.get('Novartis_IF_Flag') == 1:
        parts.append('IsolationForest')
    if row.get('Novartis_LOF_Flag') == 1:
        parts.append('LOF')
    return '; '.join(parts) if parts else 'None'


# --- SOPs ---
def sop1_simple(df):
    """SOP1: Same-granularity. Pass if >6 of (TRx,NBRx) x (T1-3,T1-6,T1-12,T1-17) exceed DT."""
    n = len(df)
    pass_any = np.zeros(n)
    std_fallback = _ensure_float(df, 'TRx_Std_12M')
    if std_fallback is None:
        std_fallback = np.ones(n) * 1e-6
    std_fallback = np.where(std_fallback > 0, std_fallback, 1e-6)
    for metric in ['TRx', 'NBRx']:
        v = _ensure_float(df, f'{metric}_T{TARGET}')
        if v is None:
            continue
        std = _ensure_float(df, f'{metric}_Std_12M')
        if std is None:
            std = std_fallback
        else:
            std = np.where(std > 0, std, 1e-6)
        for start, length in [(1, 3), (1, 6), (1, 12), (1, 17)]:
            base = _window_mean(df, metric, start, length)
            if base is None:
                continue
            z = (v - base) / std
            th = _dynamic_threshold(z)
            pass_any += (np.abs(z) >= th).astype(float)
    return (pass_any > 6).astype(int)


def sop2_hierarchical(df):
    """SOP2: Entity vs national. Pass if same directional movement."""
    n = len(df)
    trx = _ensure_float(df, f'TRx_T{TARGET}')
    base = _window_mean(df, 'TRx', 1, 3)
    if base is None or trx is None:
        return np.zeros(n)
    entity_move = (trx - base) / np.where(base > 0, base, 1e-6)
    national_trx = trx.sum()
    base_cols = [c for c in df.columns if c in ['TRx_T1', 'TRx_T2', 'TRx_T3']]
    national_base = df[base_cols].sum().sum() if base_cols else base.sum()
    national_move = (national_trx - national_base) / (national_base + 1e-6)
    return (np.sign(entity_move) == np.sign(national_move)).astype(int)


def sop3_sibling(df):
    """SOP3: Sibling. Pass if >=10% of siblings show similar directional movement."""
    n = len(df)
    trx = _ensure_float(df, f'TRx_T{TARGET}')
    base = _window_mean(df, 'TRx', 1, 3)
    if base is None or trx is None:
        return np.zeros(n)
    entity_move = (trx - base) / np.where(base > 0, base, 1e-6)
    entity_sign = np.sign(entity_move)
    pass_sop = np.zeros(n)
    for i in range(n):
        sibling_signs = np.delete(entity_sign, i)
        pct = np.sum(sibling_signs == entity_sign[i]) / len(sibling_signs) if len(sibling_signs) > 0 else 0
        pass_sop[i] = 1 if pct >= 0.10 else 0
    return pass_sop.astype(int)


def sop4_longitudinal(df):
    """SOP4: Pattern recurrence >=1, consistency ~80%. For T18: hist windows (17,15,2),(12,10,2),(6,4,2)."""
    n = len(df)
    trx_actual = _ensure_float(df, f'TRx_T{TARGET}')
    base_1_3 = _window_mean(df, 'TRx', 1, 3)
    if base_1_3 is None or trx_actual is None:
        return np.zeros(n)
    current_move = (trx_actual - base_1_3) / np.where(base_1_3 > 0, base_1_3, 1e-6)
    current_sign = np.sign(current_move)
    hist_windows = [(17, 15, 2), (12, 10, 2), (6, 4, 2)]
    similar_count = np.zeros(n)
    for (t_end, t_start, length) in hist_windows:
        if f'TRx_T{t_end}' not in df.columns:
            continue
        v_end = _ensure_float(df, f'TRx_T{t_end}')
        base_h = _window_mean(df, 'TRx', t_start, length)
        if base_h is None or v_end is None:
            continue
        move_h = (v_end - base_h) / np.where(base_h > 0, base_h, 1e-6)
        similar_count += (np.sign(move_h) == current_sign).astype(float)
    n_w = len(hist_windows)
    pattern_ok = (similar_count >= 1).astype(int)
    consistency_ok = (similar_count >= 0.8 * n_w).astype(int) if n_w > 0 else np.zeros(n)
    return (pattern_ok & consistency_ok).astype(int)


def run_sops_per_group(df_group):
    """Run SOPs on a group subset. Returns df with SOP columns."""
    df = df_group.copy()
    df['SOP1_Pass'] = sop1_simple(df)
    df['SOP2_Pass'] = sop2_hierarchical(df)
    df['SOP3_Pass'] = sop3_sibling(df)
    df['SOP4_Pass'] = sop4_longitudinal(df)
    df['SOP_Pass_Count'] = df['SOP1_Pass'] + df['SOP2_Pass'] + df['SOP3_Pass'] + df['SOP4_Pass']
    df['GAN_Anomaly_Confirmed_3of4'] = (df['SOP_Pass_Count'] >= 3).astype(int)
    df['GAN_Anomaly_Confirmed_4of4'] = (df['SOP_Pass_Count'] == 4).astype(int)
    return df


def main():
    if not os.path.exists(ANOMALIES_PATH):
        print(f'ERROR: {ANOMALIES_PATH} not found. Run build_anomalies_list_payer_pbm_t18.py first.')
        return 1

    df = pd.read_csv(ANOMALIES_PATH, low_memory=False)
    print(f'Loaded: {ANOMALIES_PATH} ({len(df)} rows)')

    # NVS (run on full - no group split needed for IF/LOF)
    df = novartis_statistical_flag(df)
    df = novartis_ml_flags(df)
    df['Novartis_Any_Flag'] = (
        (df['Novartis_Trend_Break_Flag'] == 1) |
        (df['Novartis_IF_Flag'] == 1) |
        (df['Novartis_LOF_Flag'] == 1)
    ).astype(int)
    df['Novartis_Reason'] = df.apply(novartis_reason, axis=1)

    # SOPs per Volume_Group (sibling comparison within group)
    if 'Volume_Group' not in df.columns:
        df['Volume_Group'] = 'GROUP1_High'
    out_parts = []
    for grp in df['Volume_Group'].dropna().unique():
        sub = df[df['Volume_Group'] == grp].copy()
        sub = run_sops_per_group(sub)
        out_parts.append(sub)
    df = pd.concat(out_parts, ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)
    n_anom = (df['Anomaly_Flag'] == 1).sum() if 'Anomaly_Flag' in df.columns else 0
    n_nvs = df['Novartis_Any_Flag'].sum()
    n_sop3 = (df['GAN_Anomaly_Confirmed_3of4'] == 1).sum()
    n_sop4 = (df['GAN_Anomaly_Confirmed_4of4'] == 1).sum()
    print(f'Saved: {OUTPUT_PATH}')
    print(f'  Anomaly_Flag=1: {n_anom}')
    print(f'  Novartis_Any_Flag=1: {n_nvs}')
    print(f'  GAN_Anomaly_Confirmed_3of4: {n_sop3}')
    print(f'  GAN_Anomaly_Confirmed_4of4: {n_sop4}')
    return 0


if __name__ == '__main__':
    exit(main())
