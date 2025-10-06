import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load dataset
df = pd.read_csv(
    r"C:\Users\Joshen Paul\finalyearproject\finalyearproject\PhiUSIIL_Phishing_URL_Dataset.csv"
)

# --- Step 1: Drop identifier columns ---
identifier_cols = ["URL", "Domain", "Title", "FILENAME"]
df = df.drop(columns=[col for col in identifier_cols if col in df.columns])

# --- Step 2: Separate labels ---
y = df["label"]
X = df.drop(columns=["label"])

# --- Step 3: Encode categorical column(s) ---
if "TLD" in X.columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    tld_encoded = encoder.fit_transform(X[["TLD"]])
    tld_encoded_df = pd.DataFrame(tld_encoded, columns=encoder.get_feature_names_out(["TLD"]))
    X = pd.concat([X.drop(columns=["TLD"]).reset_index(drop=True), tld_encoded_df.reset_index(drop=True)], axis=1)

# --- Step 4: Scale numeric features ---
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])
X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

# --- Step 5: Rebuild dataset ---
non_numeric_cols = [col for col in X.columns if col not in numeric_cols]
df_normalized = pd.concat(
    [X_scaled.reset_index(drop=True),
     X[non_numeric_cols].reset_index(drop=True),
     y.reset_index(drop=True)],
    axis=1
)

print("Dropped identifier columns:", identifier_cols)
print("Scaled numeric columns:", list(numeric_cols))
print("Encoded categorical columns: ['TLD']")
print(df_normalized.head())

# Save the normalized dataset to CSV
df_normalized.to_csv(
    r"C:\Users\Joshen Paul\finalyearproject\finalyearproject\normalized_dataset.csv",
    index=False
)

print("✅ Normalized dataset saved as 'normalized_dataset.csv'")