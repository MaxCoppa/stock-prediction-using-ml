"""
model_comparaison.py

Workflow script for training and evaluating a single model.
Steps:
1. Load data
2. Optional: feature engineering, missing value handling, scaling
3. Model selection via K-Fold cross-validation
4. Train final model on full dataset
5. Evaluate and generate predictions on test set
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from feature_enginneering import create_new_features
from tree_based_models import model_selection_using_kfold, get_model, evaluate_model


# %% --- Configuration ---
add_features = True  # Whether to create additional engineered features
scale = False  # Whether to scale features
replace_na = False  # Whether to impute missing values
model_name = "xgb"  # Model type: "rf", "xgb", "lgbm", "cat"


# %% --- Load data ---
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# %% --- Define features and target ---
features = [f"var{i}" for i in range(1, 11)]  # Example features
target = "TARGET"  # Example target


# %% --- Feature engineering ---
if add_features:
    train, new_features = create_new_features(train)
    test, new_features = create_new_features(test)
    features += new_features


# %% --- Handle missing values ---
if replace_na:
    imputer = SimpleImputer()
    train[features] = imputer.fit_transform(train[features])
    test[features] = imputer.transform(test[features])
else:
    train[features] = train[features].fillna(0)
    test[features] = test[features].fillna(0)


# %% --- Feature scaling ---
if scale:
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])


# %% --- Cross-validation ---
model_selection_using_kfold(
    train=train,
    target=target,
    features=features,
    model_type=model_name,
    unique_id="ID",
    plot_ft_importance=True,
)


# %% --- Train final model ---
X_train = train[features]
y_train = train[target]

model = get_model(model_name)
model.fit(X_train, y_train)


# %% --- Evaluate on training data ---
dict_eval = evaluate_model(
    model,
    X_train,
    y_train,
    threshold=0.55,
    verbose=True,
)

# %% --- Generate test predictions ---
X_test = test[features]
threshold = 0.55

proba = model.predict_proba(X_test)[:, 1]
preds = (proba > threshold).astype(int)

test[target] = preds
