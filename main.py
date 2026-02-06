import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
INPUT_FILE = "DataSet_A.xlsx"
OUTPUT_FILE = "DataSet_A_imputed_safe.xlsx"
DESIRED_K = 20        # will be capped to n_samples - 1
NUMERIC_THRESHOLD = 0.9  # fraction of values that must parse as numeric

# =========================
# READ DATA (SAFE)
# =========================
# Read as object to avoid pandas auto-mangling dates/strings
df = pd.read_excel(INPUT_FILE, dtype=object)
df_original = df.copy(deep=True)

# =========================
# COLUMN TYPE DETECTION
# =========================
def is_numeric_like(series, threshold=0.9):
    s = series.replace("nan", np.nan)
    non_null = s.dropna()
    if len(non_null) == 0:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    return (coerced.notna().sum() / len(non_null)) >= threshold

numeric_cols = []
non_numeric_cols = []

for col in df.columns:
    if is_numeric_like(df[col], NUMERIC_THRESHOLD):
        numeric_cols.append(col)
    else:
        non_numeric_cols.append(col)

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Non-numeric columns ({len(non_numeric_cols)}): {non_numeric_cols}")

# Output starts as exact copy of original
df_out = df_original.copy(deep=True)

# =========================
# 1) NUMERIC → MEDIAN + SCALE + KNN
# =========================
if numeric_cols:
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Temporary fill ONLY to fit scaler
    medians = numeric.median()
    temp_numeric = numeric.fillna(medians)

    scaler = StandardScaler()
    scaled_temp = pd.DataFrame(
        scaler.fit_transform(temp_numeric),
        index=temp_numeric.index,
        columns=temp_numeric.columns,
    )

    # Restore NaNs before KNN
    scaled_temp[numeric.isna()] = np.nan

    n_samples = scaled_temp.shape[0]
    k = min(DESIRED_K, max(1, n_samples - 1))

    knn = KNNImputer(
        n_neighbors=k,
        weights="distance",
        metric="nan_euclidean",
    )

    scaled_imputed = pd.DataFrame(
        knn.fit_transform(scaled_temp),
        index=scaled_temp.index,
        columns=scaled_temp.columns,
    )

    numeric_imputed = pd.DataFrame(
        scaler.inverse_transform(scaled_imputed),
        index=scaled_imputed.index,
        columns=scaled_imputed.columns,
    )

    # Write back column-by-column (NO concatenation)
    for col in numeric_cols:
        df_out[col + "_orig"] = df_original[col]

        col_vals = numeric_imputed[col]

        # Preserve integers where possible
        if col_vals.dropna().apply(float.is_integer).all():
            df_out[col] = col_vals.round().astype("Int64")
        else:
            df_out[col] = col_vals.astype(float)

    print(f"✔ Numeric KNN imputation complete (k={k})")
else:
    print("⚠ No numeric columns detected")

# =========================
# 2) NON-NUMERIC → MODE (SAFE)
# =========================
if non_numeric_cols:
    imp = SimpleImputer(strategy="most_frequent")

    non_num = df[non_numeric_cols].copy()
    non_num_imputed = pd.DataFrame(
        imp.fit_transform(non_num),
        index=non_num.index,
        columns=non_num.columns,
    )

    for col in non_numeric_cols:
        df_out[col + "_orig"] = df_original[col]
        df_out[col] = non_num_imputed[col]

    print(f"✔ Mode imputation applied to {len(non_numeric_cols)} non-numeric columns")
else:
    print("⚠ No non-numeric columns detected")

# =========================
# FINAL SAFETY CHECKS
# =========================
assert df_out.shape[0] == df_original.shape[0], "Row count changed!"
assert list(df_original.index) == list(df_out.index), "Row order changed!"

# Preserve original column order, backups appended at end
final_cols = list(df_original.columns)
backup_cols = [c for c in df_out.columns if c.endswith("_orig")]
df_out = df_out[final_cols + backup_cols]

# =========================
# SAVE
# =========================
df_out.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Saved safely imputed file to: {OUTPUT_FILE}")
print("🔁 Original columns preserved as *_orig")
