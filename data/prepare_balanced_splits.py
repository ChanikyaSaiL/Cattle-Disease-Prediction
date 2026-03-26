import os
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

BASE_DIR = "../cattle diseases.v2i.multiclass"
OUTPUT_DIR = "balanced_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_SAMPLES = 10   # remove rare classes
MIN_TEST_SAMPLES = 5

# =========================
# STEP 1: MERGE ALL DATA
# =========================
def load_all_data():
    dfs = []

    for split in ["train", "valid", "test"]:
        path = os.path.join(BASE_DIR, split, "_classes.csv")
        df = pd.read_csv(path)

        # Add correct image path
        df.iloc[:, 0] = split + "/" + df.iloc[:, 0]
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

df = load_all_data()

print("Original dataset size:", len(df))

# =========================
# STEP 2: REMOVE RARE CLASSES
# =========================
label_cols = df.columns[1:]

class_counts = df[label_cols].sum()

print("\nClass distribution BEFORE filtering:")
print(class_counts)

keep_classes = class_counts[class_counts >= MIN_SAMPLES].index

df = df[[df.columns[0]] + list(keep_classes)]

print("\nKeeping classes:", list(keep_classes))

# Remove samples with no labels
df = df[df.iloc[:, 1:].sum(axis=1) > 0]

print("Dataset size after filtering:", len(df))

# =========================
# STEP 3: STRATIFIED SPLIT
# =========================
X = df.iloc[:, 0].values
Y = df.iloc[:, 1:].values

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, temp_idx in msss.split(X, Y):
    X_train, X_temp = X[train_idx], X[temp_idx]
    Y_train, Y_temp = Y[train_idx], Y[temp_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

for val_idx, test_idx in msss2.split(X_temp, Y_temp):
    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    Y_val, Y_test = Y_temp[val_idx], Y_temp[test_idx]

# =========================
# STEP 4: CHECK DISTRIBUTION
# =========================
def check_distribution(name, Y):
    print(f"\n{name} distribution:")
    print(np.sum(Y, axis=0))

check_distribution("Train", Y_train)
check_distribution("Val", Y_val)
check_distribution("Test", Y_test)

# =========================
# STEP 5: SAVE SPLITS
# =========================
def save_split(X, Y, name):
    split_df = pd.DataFrame(np.column_stack([X, Y]), columns=df.columns)
    split_df.to_csv(f"{OUTPUT_DIR}/{name}.csv", index=False)

save_split(X_train, Y_train, "train")
save_split(X_val, Y_val, "valid")
save_split(X_test, Y_test, "test")

print("\n✅ Balanced splits created successfully!")