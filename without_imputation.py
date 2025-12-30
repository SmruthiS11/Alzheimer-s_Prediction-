import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("oasis_longitudinal_removed_empty.csv")

df['target'] = df['Group'].apply(lambda x: 0 if x.strip() == 'Nondemented' else 1)

# Select numeric features that may be relevant for predicting dementia
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features]
y = df['target']

# Split the data into training, validation, and testing sets.
X_train, X_temp, y_train, y_temp = train_test_split( X, y, train_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, test_size=0.7, random_state=42)

#Logitic Regression
pipeline = Pipeline([('scaler', StandardScaler()),('logreg', LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)

y_pred_test = pipeline.predict(X_test)
y_pred_val  = pipeline.predict(X_val)

accuracy_test  = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test    = recall_score(y_test, y_pred_test)
f1_test        = f1_score(y_test, y_pred_test)

print("Logistic Regression Performance on Test Set:")
print(f"Accuracy:  {accuracy_test:.4f}")
print(f"Precision: {precision_test:.4f}")
print(f"Recall:    {recall_test:.4f}")
print(f"F1 Score:  {f1_test:.4f}")

