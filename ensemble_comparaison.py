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

# %%

add_features = True
scale = False
replace_na = False
model_kfold = "xgb"
model_prediction = "xgb:cat"


# %%
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# %%
train.head()

# %%
features = [f"var{i}" for i in range(1, 11) if i not in [5]]
target = "TARGET"


# %%
if add_features:
    train, new_features = create_new_features(train)
    test, new_features = create_new_features(test)

    features += new_features

# %%

if replace_na:
    imputer = SimpleImputer()
    train[features] = imputer.fit_transform(train[features])
    test[features] = imputer.transform(test[features])

# %%

if scale:
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

# %%
model_selection_using_kfold(
    train=train,
    target=target,
    features=features,
    model_type=model_kfold,
    unique_id="ID",
    plot_ft_importance=True,
)


# %%
X_train = train[features]
y_train = train[target]


models = []
for model_name in model_prediction.split(":"):
    # Train model
    model = get_model(model_name)
    model.fit(X_train, y_train)
    models.append(model)


# %%
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

# %%
