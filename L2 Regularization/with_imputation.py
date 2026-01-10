import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("oasis_longitudinal.csv")

print("Missing values before imputation:")
print(df.isna().sum())

# Define the column groups based on their data types.
numeric_cols = ["Visit", "MR Delay", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
categorical_cols = ["Subject ID", "MRI ID", "Group", "M/F", "Hand"]

# Impute numeric columns by replacing missing values with the median
num_imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])


cat_imputer = SimpleImputer(strategy="constant", fill_value = "missing")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\nMissing values after imputation:")
print(df.isna().sum())

# Saving the imputed dataset to a new CSV file.
df.to_csv("oasis_longitudinal_imputed.csv", index=False)
print("\nImputation complete. The imputed dataset has been saved to 'oasis_longitudinal_imputed.csv'.")
