import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("oasis_longitudinal_missingdata.csv")

print("Missing values before imputation:")
print(df.isna().sum())

# Define column groups
numeric_cols = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
categorical_cols = ["Subject ID", "MRI ID", "Group", "M/F", "Hand"]

# Apply KNN imputation on numeric columns
knn_imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])

# Impute categorical columns with constant value
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\nMissing values after imputation:")
print(df.isna().sum())

# Save the imputed dataset
df.to_csv("oasis_longitudinal_imputed_knn.csv", index=False)
print("\nImputation complete. The KNN-imputed dataset has been saved to 'oasis_longitudinal_imputed_knn.csv'.")
