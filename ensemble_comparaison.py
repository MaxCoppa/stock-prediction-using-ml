"""
ensemble_comparaison.py

Workflow script for training and evaluating an ensemble of models.
Steps:
1. Load data
2. Optional: feature engineering, missing value handling, scaling
3. Model selection via K-Fold cross-validation
4. Train multiple models on full dataset
5. Evaluate ensemble on training data
6. Generate ensemble predictions on test set
"""

# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from feature_enginneering import create_new_features
from tree_based_models import (
    model_selection_using_kfold,
    get_model,
    predict_ensembler_models,
    evaluate_ensemble_model,
)

# %% --- Configuration ---
add_features = True  # Whether to create engineered features
scale = False  # Whether to scale features
replace_na = False  # Whether to impute missing values
model_kfold = "xgb"  # Model used for K-Fold validation
model_prediction = "xgb:cat"  # Models used for ensemble prediction (":" separated)


# %% --- Load data ---
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# %%
train.head()

# %% --- Define features and target ---
features = [f"var{i}" for i in range(1, 11) if i not in [5]]
target = "TARGET"


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
    model_type=model_kfold,
    unique_id="ID",
    plot_ft_importance=True,
)


# %% --- Train ensemble models ---
X_train = train[features]
y_train = train[target]

models = []
for model_name in model_prediction.split(":"):
    model = get_model(model_name)
    model.fit(X_train, y_train)
    models.append(model)


# %% --- Evaluate ensemble ---
dict_eval = evaluate_ensemble_model(
    models,
    X_train,
    y_train,
    threshold=0.55,
    verbose=True,
)


# %% --- Test predictions ---
X_test = test[features]
avg_proba, avg_preds = predict_ensembler_models(models, X_test, threshold=0.55)

test[target] = avg_preds
