# src/eda.py
"""
Exploratory Data Analysis (EDA) for FedShield dataset.
Saves plots to outputs/eda/*.png and prints summary stats to console.

Run:
    python src/eda.py
"""

import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# CONFIG
PROCESSED_CSV = "data/processed/features.csv"
CLIENT_DIR = "data/clients"
OUT_DIR = "outputs/eda"
SAMPLE_HIST_N = 50   # if many features, sample first N to avoid 200 plots
PCA_SAMPLE = 20000   # sample points for PCA scatter (reduce if memory constrained)
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

def print_header(msg):
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))

def load_data():
    if not os.path.exists(PROCESSED_CSV):
        raise FileNotFoundError(f"Processed file not found: {PROCESSED_CSV}")
    df = pd.read_csv(PROCESSED_CSV)
    return df

def summary_stats(df):
    print_header("DATA SUMMARY")
    print("Rows:", len(df))
    print("Columns:", df.shape[1])
    if 'label' in df.columns:
        print("Label counts:\n", df['label'].value_counts(), "\n")
        print("Label distribution (normalized):\n", df['label'].value_counts(normalize=True).round(4))
    print("\nFeature types:\n", df.dtypes.value_counts())
    print("\nFeature statistics (first 10):")
    print(df.describe().T.iloc[:10])

def feature_correlations(df, out_png=os.path.join(OUT_DIR, "corr_heatmap.png")):
    print_header("CORRELATIONS")
    numeric = df.select_dtypes(include=[np.number]).copy()
    # remove constant columns if any
    nunique = numeric.nunique()
    to_keep = nunique[nunique > 1].index.tolist()
    numeric = numeric[to_keep]
    corr = numeric.corr()
    # save heatmap (large heatmaps can be big; plot size adjusts)
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.01)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved correlation heatmap ->", out_png)
    return corr

def top_label_correlations(df, corr, n=20, out_png=os.path.join(OUT_DIR, "top_label_correlations.png")):
    if 'label' not in df.columns:
        return
    corr_with_label = corr['label'].drop('label').abs().sort_values(ascending=False)
    top = corr_with_label.head(n)
    print("\nTop features correlated with label (abs corr):")
    print(top)
    # barplot
    plt.figure(figsize=(8, max(4, 0.3*len(top))))
    sns.barplot(x=top.values, y=top.index, orient='h')
    plt.xlabel("Absolute correlation with label")
    plt.title(f"Top {len(top)} features by |corr| with label")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved top-label correlation barplot ->", out_png)

def plot_histograms(df, out_dir=OUT_DIR, max_plots=SAMPLE_HIST_N):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric:
        numeric = [c for c in numeric if c != 'label']
    n = len(numeric)
    print_header("HISTOGRAMS")
    print(f"Plotting {min(n, max_plots)} of {n} numeric features (first {max_plots}).")
    for i, col in enumerate(numeric[:max_plots]):
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=60, kde=False)
        plt.title(col)
        plt.tight_layout()
        p = os.path.join(out_dir, f"hist_{i:02d}_{col}.png")
        plt.savefig(p, dpi=120)
        plt.close()
    print(f"Saved histograms to {out_dir} (first {min(n,max_plots)})")

def pca_scatter(df, out_png=os.path.join(OUT_DIR, "pca_scatter.png"), n_samples=PCA_SAMPLE):
    print_header("PCA 2D SCATTER")
    numeric = df.select_dtypes(include=[np.number]).copy()
    if 'label' in numeric.columns:
        labels = numeric['label'].values
        numeric = numeric.drop(columns=['label'])
    else:
        labels = None
    nrows = len(numeric)
    if nrows > n_samples:
        sampled = numeric.sample(n=n_samples, random_state=RANDOM_STATE)
        if labels is not None:
            sampled_labels = labels[sampled.index]
        else:
            sampled_labels = None
    else:
        sampled = numeric
        sampled_labels = labels
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    proj = pca.fit_transform(sampled)
    plt.figure(figsize=(8,6))
    if sampled_labels is not None:
        sns.scatterplot(x=proj[:,0], y=proj[:,1], hue=sampled_labels, s=10, alpha=0.6, palette="Set1")
        plt.legend(title="label")
    else:
        plt.scatter(proj[:,0], proj[:,1], s=6, alpha=0.6)
    plt.title("PCA 2D projection")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved PCA scatter ->", out_png)
    print("PCA explained variance ratios:", pca.explained_variance_ratio_.round(4).tolist())

def per_client_stats(client_dir=CLIENT_DIR, out_png=os.path.join(OUT_DIR, "clients_label_ratio.png")):
    print_header("PER-CLIENT STATS")
    files = sorted([f for f in os.listdir(client_dir) if f.endswith(".csv")])
    sizes = []
    pos_ratios = []
    names = []
    for f in files:
        dfc = pd.read_csv(os.path.join(client_dir, f))
        sizes.append(len(dfc))
        if 'label' in dfc.columns:
            pos = int(dfc['label'].sum())
            ratio = pos / len(dfc) if len(dfc) > 0 else 0
        else:
            ratio = np.nan
        pos_ratios.append(ratio)
        names.append(f)
    summary = pd.DataFrame({"client": names, "size": sizes, "pos_ratio": pos_ratios})
    print("Clients:", len(names))
    print(summary.sort_values("size", ascending=False).head(10).to_string(index=False))
    # plots: sizes and pos_ratio
    plt.figure(figsize=(10,4))
    sns.barplot(x="client", y="size", data=summary.sort_values("size", ascending=False), palette="Blues_d")
    plt.xticks(rotation=90)
    plt.title("Client sample sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "clients_sizes.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10,4))
    sns.barplot(x="client", y="pos_ratio", data=summary.sort_values("pos_ratio", ascending=False), palette="RdBu")
    plt.xticks(rotation=90)
    plt.ylabel("Positive label ratio")
    plt.title("Client positive (phish) label ratio")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved client plots ->", os.path.join(OUT_DIR, "clients_sizes.png"), "and", out_png)

def feature_importance_tree(df, out_png=os.path.join(OUT_DIR, "feature_importance_tree.png"), top_n=20):
    """Quick feature importance from a shallow RandomForest for insight (not feature selection)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        print("sklearn not available for feature importance. Install scikit-learn.")
        return
    if 'label' not in df.columns:
        return
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)
    # sample to speed up
    n = len(X)
    sample_n = min(20000, n)
    Xs = X.sample(sample_n, random_state=RANDOM_STATE)
    ys = y.loc[Xs.index]
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=4, random_state=RANDOM_STATE)
    rf.fit(Xs, ys)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.3*len(imp))))
    sns.barplot(x=imp.values, y=imp.index, orient='h')
    plt.title("Top feature importances (RandomForest)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved feature importance ->", out_png)

def main():
    df = load_data()
    summary_stats(df)
    corr = feature_correlations(df)
    top_label_correlations(df, corr)
    plot_histograms(df)
    pca_scatter(df)
    per_client_stats()
    feature_importance_tree(df)
    print_header("EDA COMPLETE")
    print("All plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
