import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("oasis_longitudinal_removed_empty.csv")

# Create a binary target: 0 for Nondemented and 1 for others (such as Demented or Converted)
df['target'] = df['Group'].apply(lambda x: 0 if x.strip() == 'Nondemented' else 1)

# Select numeric features that are likely relevant for the prediction
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['target']

# First, split into training (60%) and temporary (40%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42)

# Then, split the temporary set into validation and testing sets
# We want a ratio of test:validation = 28%:12% of the full dataset.
# Since the temporary set is 40% of the data, setting test_size=0.7 in this split
# gives us 70% of 40% (28%) for testing and 30% of 40% (12%) for validation.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)

# Build a pipeline that standardizes data and then applies logistic regression with elastic net
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', LogisticRegression(
        penalty='elasticnet',     # Use elastic net regularization.
        solver='saga',            # 'saga' solver supports elastic net penalty.
        l1_ratio=0.5,             # Balance between l1 (sparse coefficients) and l2 regularization.
        max_iter=1000,
        random_state=42))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the performance on the test set
print("Elastic Net Regression (Logistic with Elastic Net) Performance on Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
