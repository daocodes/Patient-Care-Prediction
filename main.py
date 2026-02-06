import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Read Excel file
df = pd.read_excel('DataSet_A.xlsx')   # keeps column names and index

# Identify numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# --- 1) Impute numeric columns using KNN in scaled space --------------------
# We'll:
#  a) compute column medians and temporarily fill NaNs to fit the scaler
#  b) scale the temporarily-filled numeric data
#  c) put back NaNs where they were, then run KNNImputer on the scaled data
#  d) inverse-transform imputed scaled data back to original scale

numeric = df[numeric_cols].copy()

# a) medians to temporarily fill for scaler-fitting
modes = numeric.mode()

# temp_numeric: used only to fit the scaler (no effect on original NaNs)
temp_numeric = numeric.fillna(modes)

# b) fit scaler on temp_numeric and transform
scaler = StandardScaler()
temp_scaled = pd.DataFrame(scaler.fit_transform(temp_numeric),
                           index=temp_numeric.index, columns=temp_numeric.columns)

# c) restore NaNs in scaled data where original was NaN
mask_na = numeric.isna()
scaled_with_nans = temp_scaled.mask(mask_na)  # sets those positions back to NaN

# d) run KNNImputer on scaled data
knn = KNNImputer(n_neighbors=20, weights='distance', metric='nan_euclidean')
scaled_imputed_array = knn.fit_transform(scaled_with_nans)

# convert back to DataFrame with same columns/index
scaled_imputed = pd.DataFrame(scaled_imputed_array, index=numeric.index, columns=numeric.columns)

# inverse-transform to original numeric scale
numeric_imputed = pd.DataFrame(scaler.inverse_transform(scaled_imputed),
                               index=scaled_imputed.index, columns=scaled_imputed.columns)

# --- 2) Impute categorical columns (simple, common approach) ----------------
# If you have categorical strings, a common choice is most-frequent (mode).
# If categories are ordinal, you might want a different strategy.

if cat_cols:
    cat = df[cat_cols].copy()
    cat_imp = SimpleImputer(strategy='most_frequent')
    cat_imputed_array = cat_imp.fit_transform(cat)
    cat_imputed = pd.DataFrame(cat_imputed_array, index=cat.index, columns=cat.columns)
else:
    cat_imputed = pd.DataFrame(index=df.index)  # empty

# --- 3) Recombine numeric + categorical into final DataFrame ---------------
df_imputed = pd.concat([numeric_imputed, cat_imputed], axis=1)

# optional: keep original column order
df_imputed = df_imputed[df.columns]

# --- 4) Save to Excel (or continue processing) -----------------------------
df_imputed.to_excel('DataSet_A_imputed.xlsx', index=True)

print("Imputation finished. Saved to 'DataSet_A_imputed.xlsx'.")