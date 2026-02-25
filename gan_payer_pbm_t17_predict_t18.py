"""
GAN for payer-PBM ML dataset: Train on T1-T17, predict T18.
Adapts Novartis approach: volumetric decile groups, separate GAN per group.
- GAN trains on features (T1-T17 only); discriminator scores pattern anomaly.
- T18 prediction = linear extrapolation: TRx_T17 + (TRx_T17 - TRx_T16).
- Anomaly flags: low discriminator, spike (PctError > threshold), Z-score, etc.
Data: ML_DATASET_PAYER_PBM_wide_FULL.csv (311 payer-PBM rows)
"""
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, MinMaxScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DATASET_PATH = os.path.join(PROJECT_ROOT, 'ML_DATASET_PAYER_PBM_wide_FULL.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'gan_payer_pbm_t17_models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'gan_payer_pbm_t18_results')

TRAIN_TILL = 17
PREDICT_T = 18
ID_COLUMNS = ['PRCSN_PAYER_ENT_NM', 'PRCSN_PBM_VENDOR']
VOLUME_COL = 'TRx_Mean_12M'
NOISE_DIM = 100
HIDDEN_DIM_G = 256
HIDDEN_DIM_D = 256
LR = 0.0002
BATCH_SIZE = 32
EPOCHS_DEFAULT = 100
BETAS = (0.5, 0.999)

GROUP_DEFS = [
    ('GROUP1_High', list(range(1, 4))),
    ('GROUP2_Mid', list(range(4, 7))),
    ('GROUP3_Low', list(range(7, 11))),
]


def assign_volumetric_decile(df, volume_col=VOLUME_COL):
    """Assign decile 1-10 by volume (descending). If n<10, use 3 tertiles."""
    df = df.copy()
    vol = pd.to_numeric(df[volume_col], errors='coerce').fillna(0)
    rank = vol.rank(ascending=False, method='first')
    n = len(df)
    if n < 10:
        df['Volume_Decile'] = pd.qcut(rank, min(3, n), labels=[1, 2, 3], duplicates='drop').astype(int)
    else:
        decile = pd.qcut(rank, 10, labels=np.arange(1, 11), duplicates='drop')
        df['Volume_Decile'] = decile.astype(int)
    return df


def assign_volume_group(df):
    """Set Volume_Group_Label: GROUP1_High, GROUP2_Mid, GROUP3_Low."""
    df = df.copy()
    dec = df['Volume_Decile'].astype(int)
    group = pd.Series('', index=df.index)
    for name, deciles in GROUP_DEFS:
        group = np.where(dec.isin(deciles), name, group)
    df['Volume_Group_Label'] = group
    return df


def _last_two_period_cols(df, metric, exclude_target_period=PREDICT_T):
    """Return (t_second_last, t_last) column names for extrapolation."""
    import re
    pat = re.compile(rf'^{re.escape(metric)}_T(\d+)$')
    tis = []
    for c in df.columns:
        m = pat.match(c)
        if m and int(m.group(1)) != exclude_target_period:
            tis.append(int(m.group(1)))
    tis = sorted(set(tis))
    if len(tis) >= 2:
        t_second, t_last = tis[-2], tis[-1]
        return f'{metric}_T{t_second}', f'{metric}_T{t_last}'
    if len(tis) == 1:
        return f'{metric}_T{tis[0]}', f'{metric}_T{tis[0]}'
    return None, None


def extrapolate_target(df, target_period=PREDICT_T):
    """Predict target = T_last + (T_last - T_second_last). Returns dict."""
    out = {}
    for metric in ['TRx', 'NBRx', 'NRx']:
        t_actual_col = f'{metric}_T{target_period}'
        col_second, col_last = _last_two_period_cols(df, metric, exclude_target_period=target_period)
        if col_second and col_last and col_second in df.columns and col_last in df.columns:
            v_second = pd.to_numeric(df[col_second], errors='coerce').fillna(0).values
            v_last = pd.to_numeric(df[col_last], errors='coerce').fillna(0).values
            pred = v_last + (v_last - v_second)
            pred = np.maximum(pred, 0.0)
            out[f'{metric}_T{target_period}_pred'] = pred
        if t_actual_col in df.columns:
            out[f'{metric}_T{target_period}_actual'] = pd.to_numeric(df[t_actual_col], errors='coerce').fillna(0).values
    return out


def load_and_prepare(dataset_path, train_till=TRAIN_TILL, predict_t=PREDICT_T):
    """Load dataset; exclude predict period and later. Returns (X, ids_df, df, scalers, feature_cols, target_cols)."""
    df = pd.read_csv(dataset_path, low_memory=False)

    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    ids_df = df[id_cols].copy()

    exclude_patterns = tuple(f'T{t}' for t in range(predict_t, 24))
    feature_cols = [
        c for c in df.columns
        if c not in id_cols
        and not any(p in c for p in exclude_patterns)
    ]

    target_cols = [c for c in df.columns if f'T{predict_t}' in c and c not in id_cols]

    if not feature_cols:
        raise ValueError('No feature columns after excluding target period.')

    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == 'string':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].dtype not in (np.float64, np.float32, np.int64, np.int32):
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = np.abs(X).sum(axis=1)
    valid = row_sum > 0
    X = X[valid]
    ids_df = ids_df.iloc[valid].reset_index(drop=True)
    df = df.iloc[valid].reset_index(drop=True)

    scaler_std = StandardScaler()
    X_scaled = scaler_std.fit_transform(X)
    scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_minmax.fit_transform(X_scaled)

    return (
        X_scaled.astype(np.float32),
        ids_df,
        df,
        scaler_std,
        scaler_minmax,
        feature_cols,
        target_cols,
    )


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(X, noise_dim=NOISE_DIM, hidden_g=HIDDEN_DIM_G, hidden_d1=HIDDEN_DIM_D,
              lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS_DEFAULT, device=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError('PyTorch required: pip install torch')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]

    generator = Generator(noise_dim, hidden_g, input_dim).to(device)
    discriminator = Discriminator(input_dim, hidden_d1).to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=BETAS)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)
    criterion = nn.BCELoss()

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=min(batch_size, len(X) - 1), shuffle=True, drop_last=True)

    generator.train()
    discriminator.train()
    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for (batch,) in loader:
            batch = batch.to(device)
            bs = batch.size(0)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            d_optimizer.zero_grad()
            real_out = discriminator(batch)
            d_loss_real = criterion(real_out, real_labels)
            z = torch.randn(bs, noise_dim, device=device)
            fake = generator(z).detach()
            fake_out = discriminator(fake)
            d_loss_fake = criterion(fake_out, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()
            d_losses.append(d_loss.item())

            g_optimizer.zero_grad()
            z = torch.randn(bs, noise_dim, device=device)
            fake = generator(z)
            fake_out = discriminator(fake)
            g_loss = criterion(fake_out, real_labels)
            g_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss.item())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'    Epoch [{epoch+1}/{epochs}] D_loss={np.mean(d_losses):.4f} G_loss={np.mean(g_losses):.4f}')

    return generator, discriminator


def build_results(df, ids_df, X_scaled, discriminator, scaler_std, scaler_minmax,
                  feature_cols, group_name, device=None):
    """
    Novartis-style: discriminator score on T1-T17 features; extrapolation for T18 pred.
    Add anomaly flags: LowDiscriminator, Spike, ZScore, etc.
    """
    if not TORCH_AVAILABLE:
        return None
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator.eval()

    batch_size = 256
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.from_numpy(X_scaled[i:i+batch_size]).float().to(device)
            out = discriminator(batch)
            scores.append(out.cpu().numpy().flatten())
    discriminator_score = np.concatenate(scores)

    join_key = ID_COLUMNS[0] if ID_COLUMNS[0] in ids_df.columns else None
    result = ids_df.copy()
    result['Discriminator_Score'] = discriminator_score

    extrap = extrapolate_target(df, target_period=PREDICT_T)
    for k, v in extrap.items():
        result[k] = v

    trx_pred_col = f'TRx_T{PREDICT_T}_pred'
    trx_actual_col = f'TRx_T{PREDICT_T}_actual'
    if trx_pred_col not in result.columns or trx_actual_col not in result.columns:
        return result

    result[f'TRx_T{PREDICT_T}_Error'] = np.abs(result[trx_actual_col] - result[trx_pred_col])
    _, trx_last_col = _last_two_period_cols(df, 'TRx', exclude_target_period=PREDICT_T)
    trx_last = pd.to_numeric(df[trx_last_col], errors='coerce').fillna(1).values if trx_last_col in df.columns else np.ones(len(df))
    result[f'TRx_T{PREDICT_T}_PctError'] = np.where(trx_last > 0, result[f'TRx_T{PREDICT_T}_Error'] / trx_last * 100, 0)
    t_actual = result[trx_actual_col].replace(0, np.nan)
    result[f'PctError_vs_T{PREDICT_T}'] = np.where(t_actual.notna(), result[f'TRx_T{PREDICT_T}_Error'] / t_actual * 100, np.nan)

    result['Anomaly_LowDiscriminator'] = (result['Discriminator_Score'] < 0.5).astype(int)
    pct_err = result[f'TRx_T{PREDICT_T}_PctError']
    vol = result[trx_actual_col].astype(float)
    vol_p25 = vol.quantile(0.25)
    result['Volume_Tier'] = np.where(vol < vol_p25, 'Low', 'High')
    low_tier = (result['Volume_Tier'] == 'Low').values

    anomaly_pct = 80
    error_pct = 75
    floor_pct = 5
    spike_threshold_val = max(floor_pct, np.percentile(pct_err, anomaly_pct))
    error_threshold_val = max(floor_pct, np.percentile(pct_err, error_pct))
    if low_tier.any():
        spike_threshold = np.where(low_tier,
            np.maximum(floor_pct, np.percentile(pct_err[low_tier], min(95, anomaly_pct + 10))),
            spike_threshold_val)
        error_threshold = np.where(low_tier,
            np.maximum(floor_pct, np.percentile(pct_err[low_tier], min(90, error_pct + 10))),
            error_threshold_val)
    else:
        spike_threshold = np.full(len(result), spike_threshold_val)
        error_threshold = np.full(len(result), error_threshold_val)
    result['Anomaly_Threshold_Used_Pct'] = spike_threshold_val

    result['Anomaly_Spike20'] = (pct_err > spike_threshold).astype(int)
    result['Anomaly_Score'] = result['Anomaly_LowDiscriminator'] + result['Anomaly_Spike20']
    result['Anomaly_Flag'] = (result['Anomaly_Score'] >= 1).astype(int)

    t_actual = result[trx_actual_col].astype(float)
    mean_low = t_actual[low_tier].mean()
    std_low = t_actual[low_tier].std()
    mean_high = t_actual[~low_tier].mean()
    std_high = t_actual[~low_tier].std()
    z_low = np.where(std_low > 0, (t_actual - mean_low) / std_low, 0)
    z_high = np.where(std_high > 0, (t_actual - mean_high) / std_high, 0)
    result[f'TRx_T{PREDICT_T}_ZScore'] = np.where(low_tier, z_low, z_high)
    result[f'TRx_T{PREDICT_T}_ZScore_Population'] = (t_actual - t_actual.mean()) / t_actual.std() if t_actual.std() > 0 else 0

    result['Flag_Error_Above_Threshold'] = (pct_err > error_threshold).astype(int)
    result['Flag_Spike_Above_20'] = (pct_err > spike_threshold).astype(int)
    result['Flag_ZScore_Above_2'] = (result[f'TRx_T{PREDICT_T}_ZScore'].abs() > 2).astype(int)
    nbrx_actual = f'NBRx_T{PREDICT_T}_actual'
    if nbrx_actual in result.columns and trx_actual_col in result.columns:
        ratio = result[nbrx_actual] / result[trx_actual_col].replace(0, np.nan)
        q1, q3 = ratio.quantile(0.25), ratio.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr if iqr > 0 else q1 - 1.5
        hi = q3 + 1.5 * iqr if iqr > 0 else q3 + 1.5
        result['Flag_Relationship_Violation'] = ((ratio < lo) | (ratio > hi)).fillna(False).astype(int)
    else:
        result['Flag_Relationship_Violation'] = 0

    result['Threshold_Score'] = (
        result['Flag_Error_Above_Threshold'] + result['Flag_Spike_Above_20'] +
        result['Flag_ZScore_Above_2'] + result['Flag_Relationship_Violation']
    )
    result['Threshold_Flag'] = (result['Threshold_Score'] >= 3).astype(int)
    result['Priority'] = np.where(result['Threshold_Score'] >= 3, 'High',
                        np.where(result['Threshold_Score'] == 2, 'Med',
                        np.where(result['Threshold_Score'] == 1, 'Low', 'Normal')))
    result['Flagged_for_Review'] = (result['Threshold_Flag'] == 1).astype(int)
    result['Volume_Group'] = group_name

    def _explain(r):
        p = []
        thresh = result['Anomaly_Threshold_Used_Pct'].iloc[0] if 'Anomaly_Threshold_Used_Pct' in result.columns else 20
        if r['Flag_Error_Above_Threshold']: p.append(f'Error>{thresh:.0f}%')
        if r['Flag_Spike_Above_20']: p.append(f'Spike>{thresh:.0f}%')
        if r['Flag_ZScore_Above_2']: p.append('Z>2')
        if r['Flag_Relationship_Violation']: p.append('RelViolation')
        return '; '.join(p) if p else 'None'
    result['Explanation'] = result.apply(_explain, axis=1)

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GAN: Train T1-T17, predict T18, volumetric decile groups')
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS_DEFAULT)
    parser.add_argument('--skip-train', action='store_true', help='Only build groups, skip training')
    parser.add_argument('--no-decile', action='store_true', help='Train single GAN on all data (no groups)')
    args = parser.parse_args()

    if not os.path.exists(DATASET_PATH):
        print(f'ERROR: Dataset not found: {DATASET_PATH}')
        return 1

    print('='*80)
    print('GAN PAYER-PBM: Train T1-T17, Predict T18 (Novartis-style volumetric decile groups)')
    print('='*80)
    print(f'  Dataset: {DATASET_PATH}')
    print(f'  Train: T1-T17 | Predict: T18 (extrapolation: T17 + (T17 - T16))')
    print(f'  Volume col for deciling: {VOLUME_COL}')
    print('='*80)

    df_full = pd.read_csv(DATASET_PATH, low_memory=False)
    if VOLUME_COL not in df_full.columns:
        trx_cols = [c for c in df_full.columns if c.startswith('TRx_T') and 'MoM' not in c]
        trx_t1_17 = [c for c in trx_cols if any(f'T{i}' in c for i in range(1, 18))]
        df_full['TRx_Mean_12M'] = df_full[[c for c in trx_t1_17 if c in df_full.columns]].mean(axis=1)

    df_full = assign_volumetric_decile(df_full)
    df_full = assign_volume_group(df_full)

    decile_path = os.path.join(PROJECT_ROOT, 'ML_DATASET_PAYER_PBM_wide_FULL_WITH_DECILE.csv')
    df_full.to_csv(decile_path, index=False)
    print(f'  Saved with decile: {decile_path}')

    if args.no_decile:
        groups_to_run = [('ALL', df_full)]
    else:
        groups_to_run = []
        for name, _ in GROUP_DEFS:
            sub = df_full[df_full['Volume_Group_Label'] == name]
            if len(sub) > 0:
                groups_to_run.append((name, sub))
                print(f'  {name}: {len(sub)} entities')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for group_name, df_sub in groups_to_run:
        group_path = os.path.join(PROJECT_ROOT, f'ML_DATASET_PAYER_PBM_T17_GROUP_{group_name}.csv')
        df_sub.to_csv(group_path, index=False)

        print(f'\n--- {group_name} ({len(df_sub)} rows) ---')
        try:
            X, ids_df, df_sub, scaler_std, scaler_minmax, feature_cols, target_cols = load_and_prepare(group_path)
        except Exception as e:
            print(f'  Load error: {e}')
            continue

        print(f'  Features: {X.shape[1]}, Samples: {X.shape[0]}')

        if args.skip_train:
            continue

        if not TORCH_AVAILABLE:
            print('  PyTorch not available, skip training')
            continue

        generator, discriminator = train_gan(X, epochs=args.epochs, batch_size=min(BATCH_SIZE, len(X) - 1))

        group_out = os.path.join(OUTPUT_DIR, f'gan_{group_name}')
        os.makedirs(group_out, exist_ok=True)
        suffix = f'_{group_name}'
        torch.save(generator.state_dict(), os.path.join(group_out, f'generator{suffix}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(group_out, f'discriminator{suffix}.pth'))
        with open(os.path.join(group_out, f'scaler_std{suffix}.pkl'), 'wb') as f:
            pickle.dump(scaler_std, f)
        with open(os.path.join(group_out, f'scaler_minmax{suffix}.pkl'), 'wb') as f:
            pickle.dump(scaler_minmax, f)
        with open(os.path.join(group_out, f'feature_cols{suffix}.pkl'), 'wb') as f:
            pickle.dump(feature_cols, f)
        config = {'train_till': TRAIN_TILL, 'predict_t': PREDICT_T, 'n_features': X.shape[1]}
        with open(os.path.join(group_out, f'config{suffix}.pkl'), 'wb') as f:
            pickle.dump(config, f)

        result = build_results(df_sub, ids_df, X, discriminator, scaler_std, scaler_minmax,
                              feature_cols, group_name)
        if result is not None:
            all_results.append(result)
            out_csv = os.path.join(RESULTS_DIR, f'GAN_T18_RESULTS_{group_name}.csv')
            result.to_csv(out_csv, index=False)
            print(f'  Results: {out_csv}')
            err_col = f'TRx_T{PREDICT_T}_Error'
            pct_col = f'TRx_T{PREDICT_T}_PctError'
            if err_col in result.columns:
                print(f'  Mean |error|: {result[err_col].mean():.2f}, Mean % error: {result[pct_col].mean():.2f}%')
            print(f'  Anomaly_Flag: {result["Anomaly_Flag"].sum()} / {len(result)}')
            print(f'  Threshold_Flag (Score>=3): {result["Threshold_Flag"].sum()} / {len(result)}')

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, 'GAN_T18_RESULTS_ALL_GROUPS.csv'), index=False)
        print(f'\n  Combined results: {os.path.join(RESULTS_DIR, "GAN_T18_RESULTS_ALL_GROUPS.csv")}')

    print('\n' + '='*80)
    print('Done. Models:', OUTPUT_DIR, '| Results:', RESULTS_DIR)
    print('='*80)
    return 0


if __name__ == '__main__':
    exit(main())
