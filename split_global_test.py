import pandas as pd
from sklearn.model_selection import train_test_split

# Load the normalized dataset
df_normalized = pd.read_csv(
    r"C:\Users\Joshen Paul\finalyearproject\finalyearproject\normalized_dataset.csv"
)

# Split into Global Test Set (10%) and Training Pool (90%)
train_pool, global_test_set = train_test_split(
    df_normalized, test_size=0.10, random_state=42, stratify=df_normalized["label"]
)

# Save both sets
train_pool.to_csv(
    r"C:\Users\Joshen Paul\finalyearproject\finalyearproject\train_pool.csv", index=False
)
global_test_set.to_csv(
    r"C:\Users\Joshen Paul\finalyearproject\finalyearproject\global_test_set.csv", index=False
)

print("✅ Global Test Set created and saved.")
print("Training Pool shape:", train_pool.shape)
print("Global Test Set shape:", global_test_set.shape)