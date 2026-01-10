import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pymc as pm
import arviz as az

# Load the dataset
df = pd.read_csv("oasis_longitudinal_removed_empty.csv")

# Map target: 'Nondemented' to 0, all else to 1
df['target'] = df['Group'].apply(lambda x: 0 if x.strip() == 'Nondemented' else 1)

# Select relevant numeric features
features = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
X = df[features].values
y = df['target'].values

# Split into training and testing data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Bayesian Logistic Regression using PyMC
with pm.Model() as logistic_model:
    # Priors
    coeffs = pm.Normal("coeffs", mu=0, sigma=5, shape=X_train.shape[1])
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Logistic function
    logits = pm.math.dot(X_train, coeffs) + intercept
    theta = pm.Deterministic("theta", pm.math.sigmoid(logits))

    # Likelihood
    y_obs = pm.Bernoulli("y_obs", p=theta, observed=y_train)

    # Sampling
    trace = pm.sample(1000, tune=1000, target_accept=1, return_inferencedata=True)

# Get posterior mean of coefficients and intercept
coeff_means = trace.posterior["coeffs"].mean(dim=("chain", "draw")).values
intercept_mean = trace.posterior["intercept"].mean(dim=("chain", "draw")).values

# Predict on test set
logits_test = np.dot(X_test, coeff_means) + intercept_mean
probs_test = 1 / (1 + np.exp(-logits_test))
y_pred = (probs_test >= 0.5).astype(int)

# Evaluate performance
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

# Print results
print("Bayesian Logistic Regression Performance on Test Set:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
