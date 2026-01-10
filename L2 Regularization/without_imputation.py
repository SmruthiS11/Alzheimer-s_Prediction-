import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset from CSV
df = pd.read_csv("oasis_longitudinal_removed_empty.csv")

# Create a binary target variable:
# Map 'Nondemented' to 0 and all other groups (Demented, Converted, etc.) to 1.
df['target'] = df['Group'].apply(lambda x: 0 if x.strip() == 'Nondemented' else 1)

# Define the numeric features for the model.
# (Feel free to adjust this feature list as needed.)
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['target']

# First, split the dataset into training (60%) and a temporary set (40%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42)

# Next, split the temporary set into test and validation sets such that
# 70% of the temporary data forms the test set (70% of 40% = 28% overall)
# and the remaining 30% forms the validation set (30% of 40% = 12% overall).
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)

# Create a pipeline that first standardizes the features then fits a logistic regression model
# with an L2 penalty (ridge). The default solver 'lbfgs' supports L2 regularization.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge_logreg', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))
])

# Fit the pipeline on the training data.
pipeline.fit(X_train, y_train)

# Predict on the test set.
y_pred = pipeline.predict(X_test)

# Compute evaluation metrics.
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Print the performance metrics on the test set.
print("L2 (Ridge) Logistic Regression Performance on Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
