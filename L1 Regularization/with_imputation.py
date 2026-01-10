import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("oasis_longitudinal_imputed.csv")

# Create a binary target:
# Map 'Nondemented' to 0 and all other groups (e.g., Demented, Converted) to 1.
df['target'] = df['Group'].apply(lambda x: 0 if x.strip() == 'Nondemented' else 1)

# Select the numeric features that may be relevant for predicting the outcome.
# (You can adjust the feature list if needed.)
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['target']

# Split the data into training, testing, and validation sets.
# First, split into training (60%) and a temporary set (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6,
                                                    random_state=42)

# Then, split the temporary set so that 70% of it is used for testing
# and the remaining 30% is used for validation.
# 70% of 40% equals 28% of the full dataset for testing;
# 30% of 40% equals 12% of the full dataset for validation.
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp,
                                                train_size=0.7, random_state=42)

# Build a pipeline to standardize the features and then perform L1-regularized logistic regression.
# Here, L1 regularization is equivalent to Lasso in the context of logistic regression.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso_logreg', LogisticRegression(
        penalty='l1',           # Use L1 penalty (lasso)
        solver='liblinear',     # liblinear solver supports L1 penalty
        max_iter=1000,
        random_state=42))
])

# Fit the model pipeline on the training data.
pipeline.fit(X_train, y_train)

# Predict on the test set.
y_pred = pipeline.predict(X_test)

# Evaluate the model.
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Print out the performance on the test set.
print("L1 (Lasso) Logistic Regression Performance on Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
