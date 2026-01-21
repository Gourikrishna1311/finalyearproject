"""
Data pipeline for FedShield (REFACTORED - Fixed Data Leakage):
 - robust CSV read (utf-8 / latin-1 fallback)
 - cleaning & numeric feature selection (keeps numeric columns + label)
 - stratified 70/20/10 split (train/val/test) BEFORE scaling
 - Z-score scaling fitted ONLY on training data (prevents data leakage)
 - scaler persistence for deployment
 - Dirichlet partition on TRAIN only to create N non-IID client CSVs

Usage:
    python src/data_pipeline.py

Config at top of file can be changed for paths / n_clients / alpha / random seed.

CRITICAL FIX: Split happens BEFORE scaling to prevent data leakage!
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# ----------------- CONFIG -----------------
RAW_CSV = "data/raw/dataset.csv"
PROCESSED_CSV = "data/processed/features.csv"  # kept for compatibility, not used
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV = "data/splits/val.csv"
TEST_CSV = "data/splits/test.csv"
CLIENT_DIR = "data/clients/"
SCALER_PATH = "data/processed/scaler.pkl"  # NEW: save fitted scaler

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
        print(f"[+] Read CSV with utf-8 encoding: {len(df)} rows")
    except Exception as e:
        warnings.warn(f"UTF-8 failed, trying latin-1")
        df = pd.read_csv(path, engine='python', encoding='latin-1')
        print(f"[+] Read CSV with latin-1 encoding: {len(df)} rows")
    return df

def select_numeric_features(df, drop_cols=None, keep_cols=None):
    """
    Keep numeric columns by default (and label), drop obvious textual columns unless specified.
    Provide keep_cols list to force keep specific columns.
    """
    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            warnings.warn(f"Requested columns not found: {missing}")
        return df[cols].copy()

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    to_drop = drop_cols or ['FILENAME', 'URL', 'Domain', 'Title', 'Title,Domain']
    to_drop = [c for c in to_drop if c in df.columns]
    keep = [c for c in numeric if c not in to_drop]
    if 'label' in df.columns and 'label' not in keep:
        keep.append('label')
    
    print(f"[+] Selected {len(keep)} numeric columns")
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
    
    original_len = len(df)

    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').astype('Int64')
        df = df[df['label'].notna()]
        dropped = original_len - len(df)
        if dropped > 0:
            print(f"[!] Dropped {dropped} rows with missing labels")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numeric NaNs with median
    for col in df.select_dtypes(include=[np.number]).columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # Clip extreme values (safe guard)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].clip(lower=-1e9, upper=1e9)

    df = df.reset_index(drop=True)
    
    # Validation
    if len(df) == 0:
        raise ValueError("Dataframe is empty after cleaning!")
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"[+] Class distribution: {label_counts.to_dict()}")
        if any(label_counts < 10):
            warnings.warn(f"Some classes have < 10 samples")
    
    print(f"[+] Cleaned dataframe: {len(df)} rows")
    return df

def split_70_20_10_and_save(df,
                            out_train=TRAIN_CSV,
                            out_val=VAL_CSV,
                            out_test=TEST_CSV,
                            random_state=RANDOM_STATE):
    """
    Stratified split into train (70%), val (20%), test (10%).
    IMPORTANT: This happens BEFORE scaling to prevent data leakage.
    Splits are stratified by 'label'.
    """
    if 'label' not in df.columns:
        raise ValueError("Dataframe must contain 'label' column for stratified split")

    X = df.drop(columns=['label'])
    y = df['label'].values

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

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    print(f"[+] Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def scale_splits_and_save(train_df, val_df, test_df,
                          out_train=TRAIN_CSV,
                          out_val=VAL_CSV,
                          out_test=TEST_CSV,
                          scaler_path=SCALER_PATH):
    """
    Standardize numeric features (Z-score) fitted ONLY on training data.
    This prevents data leakage - scaler never sees validation or test data during fitting.
    Returns scaled dataframes and fitted scaler.
    """
    # Separate features and labels
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label'].astype(int)
    
    X_val = val_df.drop(columns=['label'])
    y_val = val_df['label'].astype(int)
    
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label'].astype(int)
    
    # Fit scaler ONLY on training data - CRITICAL for preventing data leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print(f"[+] Scaler fitted on training data only")
    print(f"    Mean of first feature: {scaler.mean_[0]:.4f}")
    print(f"    Std of first feature: {scaler.scale_[0]:.4f}")
    
    # Transform all splits using training statistics
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Reconstruct DataFrames
    train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_scaled['label'] = y_train.values
    
    val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    val_scaled['label'] = y_val.values
    
    test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_scaled['label'] = y_test.values
    
    # Save splits
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    train_scaled.to_csv(out_train, index=False)
    val_scaled.to_csv(out_val, index=False)
    test_scaled.to_csv(out_test, index=False)
    
    print(f"[+] Saved train ({len(train_scaled)}) -> {out_train}")
    print(f"[+] Saved val   ({len(val_scaled)}) -> {out_val}")
    print(f"[+] Saved test  ({len(test_scaled)}) -> {out_test}")
    
    # Save scaler for deployment
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved scaler to {scaler_path}")
    
    return train_scaled, val_scaled, test_scaled, scaler

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

    # Enhanced balancing: ensure each client has at least min_size samples
    initial_sizes = [len(ci) for ci in client_idx]
    print(f"    Initial client sizes - Min: {min(initial_sizes)}, Max: {max(initial_sizes)}, Mean: {np.mean(initial_sizes):.1f}")
    
    for attempt in range(100):  # Increased from 10 to 100
        sizes = [len(ci) for ci in client_idx]
        if min(sizes) >= min_size:
            break
        for i, s in enumerate(sizes):
            if s < min_size:
                largest = int(np.argmax(sizes))
                if sizes[largest] <= min_size:
                    warnings.warn(f"Cannot balance client {i} to {min_size} samples")
                    break
                moved = client_idx[largest].pop()
                client_idx[i].append(moved)
                sizes = [len(ci) for ci in client_idx]
    
    final_sizes = [len(ci) for ci in client_idx]
    print(f"    Final client sizes - Min: {min(final_sizes)}, Max: {max(final_sizes)}, Mean: {np.mean(final_sizes):.1f}")
    
    if min(final_sizes) < min_size:
        warnings.warn(f"Some clients have fewer than {min_size} samples")

    return client_idx

def save_clients_csv(train_df, client_idx_list, out_dir=CLIENT_DIR):
    """Save per-client CSV files from train_df by provided index lists (indices refer to train_df)."""
    os.makedirs(out_dir, exist_ok=True)
    for i, idxs in enumerate(client_idx_list):
        if len(idxs) == 0:
            # create empty CSV with columns for safety
            empty = pd.DataFrame(columns=train_df.columns)
            empty.to_csv(os.path.join(out_dir, f"client_{i}.csv"), index=False)
            warnings.warn(f"Client {i} has 0 samples")
            continue
        client_df = train_df.iloc[idxs].copy().reset_index(drop=True)
        client_df.to_csv(os.path.join(out_dir, f"client_{i}.csv"), index=False)
    print(f"[+] Saved {len(client_idx_list)} client files to {out_dir}")

def run_full_pipeline(raw_csv=RAW_CSV,
                      processed_csv=PROCESSED_CSV,  # kept for compatibility, not used
                      train_csv=TRAIN_CSV,
                      val_csv=VAL_CSV,
                      test_csv=TEST_CSV,
                      client_dir=CLIENT_DIR,
                      n_clients=N_CLIENTS,
                      dir_alpha=DIRICHLET_ALPHA):
    """
    Run complete pipeline with CORRECTED order to prevent data leakage:
    1. Read → 2. Clean → 3. Select features → 4. SPLIT → 5. SCALE → 6. Partition
    
    CRITICAL: Split happens BEFORE scaling!
    """
    print("="*60)
    print("FEDSHIELD DATA PIPELINE (Data Leakage Fixed)")
    print("="*60)
    
    print("\n[*] Reading raw CSV...")
    df_raw = robust_read_csv(raw_csv)

    print("\n[*] Cleaning dataframe...")
    df_clean = clean_dataframe(df_raw)

    print("\n[*] Selecting numeric features...")
    df_num = select_numeric_features(df_clean, drop_cols=['FILENAME','URL','Domain','Title','Title,Domain'])
    print("    Numeric columns kept:", df_num.columns.tolist())

    print("\n[*] Performing stratified 70/20/10 split (BEFORE scaling)...")
    train_df, val_df, test_df = split_70_20_10_and_save(df_num,
                                                         out_train=train_csv,
                                                         out_val=val_csv,
                                                         out_test=test_csv,
                                                         random_state=RANDOM_STATE)

    print("\n[*] Scaling features (fitted on TRAIN only) & saving...")
    train_scaled, val_scaled, test_scaled, scaler = scale_splits_and_save(
        train_df, val_df, test_df,
        out_train=train_csv,
        out_val=val_csv,
        out_test=test_csv,
        scaler_path=SCALER_PATH
    )

    print("\n[*] Creating Dirichlet non-iid partitions on TRAIN only...")
    client_idxs = dirichlet_partition(train_scaled['label'], n_clients=n_clients, alpha=dir_alpha)
    save_clients_csv(train_scaled, client_idxs, out_dir=client_dir)

    print("\n" + "="*60)
    print("[+] Pipeline finished successfully!")
    print("="*60)
    print(f"\nKey outputs:")
    print(f"  - Scaler saved to: {SCALER_PATH}")
    print(f"  - Train/val/test splits saved to: {os.path.dirname(train_csv)}")
    print(f"  - {n_clients} client files saved to: {client_dir}")
    
    return {
        "processed_csv": processed_csv,  # kept for compatibility
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "scaler_path": SCALER_PATH,
        "client_dir": client_dir,
        "n_clients": n_clients,
        "dir_alpha": dir_alpha
    }

if __name__ == "__main__":
    run_full_pipeline()