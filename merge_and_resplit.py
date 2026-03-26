import os
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

BASE_DIR = "cattle diseases.v2i.multiclass"
SPLITS = ["train", "valid", "test"]

OUTPUT_DIR = "balanced_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MERGE ALL DATA
# =========================
dfs = []

for split in SPLITS:
    path = os.path.join(BASE_DIR, split, "_classes.csv")
    df = pd.read_csv(path)

    # Add full image path
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: os.path.join(split, x))

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("✅ Total samples:", len(df))

X = df.iloc[:, 0].values
Y = df.iloc[:, 1:].values

# =========================
# STRATIFIED SPLIT
# =========================
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, temp_idx in msss.split(X, Y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    Y_train, Y_temp = Y[train_idx], Y[temp_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for val_idx, test_idx in msss2.split(X_temp, Y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    Y_val, Y_test = Y_temp[val_idx], Y_temp[test_idx]

# =========================
# SAVE SPLITS
# =========================
def save_split(X, Y, name):
    split_df = pd.DataFrame(np.column_stack([X, Y]), columns=df.columns)
    split_df.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)

save_split(X_train, Y_train, "train")
save_split(X_val, Y_val, "valid")
save_split(X_test, Y_test, "test")

print("✅ Balanced stratified splits created")