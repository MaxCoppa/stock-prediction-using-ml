"""
A Random Forest (RF) model is chosen for the Benchmark. We consider a large number of tree with a quiet small depth. The missing values are simply filled with 0. A KFold is done on the dates (using `DATE`) for a local scoring of the model.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

from .initialise_model import get_model
from .evaluate import evaluate_model


def model_selection_using_kfold(
    train,
    target,
    features,
    model_type,
    unique_id: str = "ID",
    plot_ft_importance: bool = False,
):

    X_train = train[features]
    y_train = train[target]

    unique_ids = train[unique_id].unique()

    n_splits = 4  # Split Kfold

    metrics = {  # Store all metrics
        "accuracy": [],
        "auc": [],
        "f1": [],
    }
    scores = []  # List for scores
    models = []  # Keep the different models

    splits = KFold(n_splits=n_splits, random_state=0, shuffle=True).split(unique_ids)

    for i, (local_unique_ids_ids, local_test_dates_ids) in enumerate(splits):

        X_local_train, y_local_train, _ = get_data(
            local_unique_ids_ids, unique_ids, train[unique_id], X_train, y_train
        )
        X_local_test, y_local_test, local_test_ids = get_data(
            local_test_dates_ids, unique_ids, train[unique_id], X_train, y_train
        )

        X_local_train = X_local_train.fillna(0)  # Fill missing values with 0
        X_local_test = X_local_test.fillna(0)  # Fill missing values with 0

        model = get_model(model_type)  # model definition

        model.fit(
            X_local_train, y_local_train
        )  # Fit the model using the X,y definied above

        model_eval_k = evaluate_model(
            model=model,
            X=X_local_test,
            y=y_local_test,
        )

        models.append(
            model
        )  # Here on fit several models to test the pertinence of random forest

        # Compute metrics
        acc = model_eval_k["accuracy"]
        roc_auc = model_eval_k["roc_auc"]
        f1 = model_eval_k["f1"]

        metrics["accuracy"].append(model_eval_k["accuracy"])
        metrics["auc"].append(roc_auc)
        metrics["f1"].append(f1)

        print(
            f"Fold {i+1} - Accuracy: {acc*100:.2f}% | "
            f"AUC: {roc_auc:.3f} | F1: {f1:.3f}"
        )

    # Aggregate results
    for m in metrics:
        mean = np.mean(metrics[m]) * (100 if m == "accuracy" else 1)
        std = np.std(metrics[m]) * (100 if m == "accuracy" else 1)
        u = mean + std
        l = mean - std
        unit = "%" if m == "accuracy" else ""
        print(
            f"{m.capitalize()}: {mean:.2f}{unit} [{l:.2f}{unit} ; {u:.2f}{unit}] (+- {std:.2f}{unit})"
        )

    if plot_ft_importance:
        plot_feature_importance(models, features)


def get_data(
    ids, dates_unique, dates_col: pd.Series, X_data: pd.DataFrame, y_data: pd.Series
):

    local_dates = dates_unique[ids]
    local_ids = dates_col.isin(local_dates)

    return X_data.loc[local_ids], y_data.loc[local_ids], local_ids


def plot_feature_importance(models, features):

    feature_importances = pd.DataFrame(
        [model.feature_importances_ for model in models], columns=features
    )

    sns.barplot(
        data=feature_importances,
        orient="h",
        order=feature_importances.mean().sort_values(ascending=False).index,
    )

    return True
