"""
Data pipeline for FedShield:
 - robust CSV read (utf-8 / latin-1 fallback)
 - cleaning & numeric feature selection (keeps numeric columns + label)
 - scaling (Z-score) and saving processed csv
 - stratified 70/20/10 split (train/val/test)
 - Dirichlet partition on TRAIN only to create N non-IID client CSVs

Usage:
    python src/data_pipeline.py

Config at top of file can be changed for paths / n_clients / alpha / random seed.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# ----------------- CONFIG -----------------
RAW_CSV = "data/raw/dataset.csv"
PROCESSED_CSV = "data/processed/features.csv"
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"
TEST_CSV = "data/splits/test.csv"
CLIENT_DIR = "data/clients/"

RANDOM_STATE = 42
N_CLIENTS = 20
DIRICHLET_ALPHA = 0.5   # smaller -> more skew (non-iid)
MIN_CLIENT_SIZE = 10    # minimum samples per client after balancing
# ------------------------------------------

def robust_read_csv(path):
    """Read CSV with utf-8, fallback to latin-1. Uses python engine to tolerate complex CSVs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CSV not found at: {path}")
    try:
        df = pd.read_csv(path, engine='python', encoding='utf-8')
    except Exception:
        df = pd.read_csv(path, engine='python', encoding='latin-1')
    return df

def select_numeric_features(df, drop_cols=None, keep_cols=None):
    """
    Keep numeric columns by default (and label), drop obvious textual columns unless specified.
    Provide keep_cols list to force keep specific columns.
    """
    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        return df[cols].copy()

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    to_drop = drop_cols or ['FILENAME', 'URL', 'Domain', 'Title', 'Title,Domain']
    to_drop = [c for c in to_drop if c in df.columns]
    keep = [c for c in numeric if c not in to_drop]
    if 'label' in df.columns and 'label' not in keep:
        keep.append('label')
    return df[keep].copy()

def clean_dataframe(df):
    """
    Basic cleaning:
     - strip column names
     - coerce label to integer type and drop missing labels
     - replace infinities
     - fill numeric NaNs with median
     - clip extremes
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
        df = df[df['label'].notna()]

    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric NaNs with median
    for col in df.select_dtypes(include=[np.number]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Clip extreme values (safe guard)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].clip(lower=-1e9, upper=1e9)

    df = df.reset_index(drop=True)
    return df

def scale_and_save(df, out_csv=PROCESSED_CSV):
    """
    Standardize numeric features (Z-score). Returns scaled df and fitted scaler.
    """
    if 'label' in df.columns:
        X = df.drop(columns=['label'])
        y = df['label'].astype(int)
    else:
        X = df.copy()
        y = None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Xs = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    if y is not None:
        Xs['label'] = y.values

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    Xs.to_csv(out_csv, index=False)
    print(f"[+] Saved processed features to {out_csv} (n={len(Xs)})")
    return Xs, scaler

def split_70_20_10_and_save(df_scaled,
                            out_train=TRAIN_CSV,
                            out_val=VAL_CSV,
                            out_test=TEST_CSV,
                            random_state=RANDOM_STATE):
    """
    Stratified split into train (70%), val (20%), test (10%).
    Splits are stratified by 'label'.
    """
    if 'label' not in df_scaled.columns:
        raise ValueError("Dataframe must contain 'label' column for stratified split")

    X = df_scaled.drop(columns=['label'])
    y = df_scaled['label'].values

    # First split: train (70%) and temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(sss1.split(X, y))

    # temp -> val (20%) and test (10%) overall -> val = 2/3 of temp, test = 1/3 of temp
    val_fraction_of_temp = 2.0 / 3.0
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_fraction_of_temp, random_state=random_state)
    temp_X = X.iloc[temp_idx]
    temp_y = y[temp_idx]
    val_sub_idx, test_sub_idx = next(sss2.split(temp_X, temp_y))

    val_idx = np.array(temp_idx)[val_sub_idx].tolist()
    test_idx = np.array(temp_idx)[test_sub_idx].tolist()

    train_df = df_scaled.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df_scaled.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df_scaled.iloc[test_idx].copy().reset_index(drop=True)

    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)
    os.makedirs(os.path.dirname(out_test), exist_ok=True)

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"[+] Saved train ({len(train_df)}) -> {out_train}")
    print(f"[+] Saved val   ({len(val_df)}) -> {out_val}")
    print(f"[+] Saved test  ({len(test_df)}) -> {out_test}")
    return train_df, val_df, test_df

def dirichlet_partition(y, n_clients=N_CLIENTS, alpha=DIRICHLET_ALPHA, min_size=MIN_CLIENT_SIZE, rng=None):
    """
    Create label-aware Dirichlet partitions for non-IID client splits.
    Returns list of index lists (length n_clients), indices referring to 'y' (series/array).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    y = np.array(y)
    labels = np.unique(y)
    idx_by_label = {lab: np.where(y == lab)[0] for lab in labels}
    client_idx = [[] for _ in range(n_clients)]

    for lab in labels:
        idxs = idx_by_label[lab].tolist()
        if len(idxs) == 0:
            continue
        proportions = rng.dirichlet([alpha] * n_clients)
        counts = (proportions * len(idxs)).astype(int)
        # fix rounding diff
        diff = len(idxs) - counts.sum()
        for i in range(abs(diff)):
            counts[i % n_clients] += int(np.sign(diff))
        rng.shuffle(idxs)
        pointer = 0
        for c in range(n_clients):
            cnt = counts[c]
            if cnt > 0:
                client_idx[c].extend(idxs[pointer:pointer+cnt])
                pointer += cnt

    # Simple balancing: ensure each client has at least min_size samples by moving from largest buckets
    for attempt in range(10):
        sizes = [len(ci) for ci in client_idx]
        if min(sizes) >= min_size:
            break
        for i, s in enumerate(sizes):
            if s < min_size:
                largest = int(np.argmax(sizes))
                if sizes[largest] <= min_size:
                    break
                moved = client_idx[largest].pop()
                client_idx[i].append(moved)
                sizes = [len(ci) for ci in client_idx]

    return client_idx

def save_clients_csv(train_df, client_idx_list, out_dir=CLIENT_DIR):
    """Save per-client CSV files from train_df by provided index lists (indices refer to train_df)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, idxs in enumerate(client_idx_list):
        if len(idxs) == 0:
            # create empty CSV with columns for safety
            empty = pd.DataFrame(columns=train_df.columns)
            empty.to_csv(os.path.join(out_dir, f"client_{i}.csv"), index=False)
            continue
        client_df = train_df.iloc[idxs].copy().reset_index(drop=True)
        client_df.to_csv(os.path.join(out_dir, f"client_{i}.csv"), index=False)
    print(f"[+] Saved {len(client_idx_list)} client files to {out_dir}")

def run_full_pipeline(raw_csv=RAW_CSV,
                      processed_csv=PROCESSED_CSV,
                      train_csv=TRAIN_CSV,
                      val_csv=VAL_CSV,
                      test_csv=TEST_CSV,
                      client_dir=CLIENT_DIR,
                      n_clients=N_CLIENTS,
                      dir_alpha=DIRICHLET_ALPHA):
    print("[*] Reading raw CSV...")
    df_raw = robust_read_csv(raw_csv)

    print("[*] Cleaning dataframe...")
    df_clean = clean_dataframe(df_raw)

    print("[*] Selecting numeric features...")
    df_num = select_numeric_features(df_clean, drop_cols=['FILENAME','URL','Domain','Title','Title,Domain'])
    print("    Numeric columns kept:", df_num.columns.tolist())

    print("[*] Scaling features & saving processed CSV...")
    df_scaled, scaler = scale_and_save(df_num, out_csv=processed_csv)

    print("[*] Performing stratified 70/20/10 split...")
    train_df, val_df, test_df = split_70_20_10_and_save(df_scaled,
                                                        out_train=train_csv,
                                                        out_val=val_csv,
                                                        out_test=test_csv,
                                                        random_state=RANDOM_STATE)

    print("[*] Creating Dirichlet non-iid partitions on TRAIN only...")
    client_idxs = dirichlet_partition(train_df['label'], n_clients=n_clients, alpha=dir_alpha)
    save_clients_csv(train_df, client_idxs, out_dir=client_dir)

    print("[+] Pipeline finished successfully.")
    return {
        "processed_csv": processed_csv,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "client_dir": client_dir,
        "n_clients": n_clients,
        "dir_alpha": dir_alpha
    }

if __name__ == "__main__":
    run_full_pipeline()
