"""
Utility for initializing tree-based classifiers (Random Forest, XGBoost,
LightGBM, CatBoost) with predefined hyperparameters.
"""

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Default hyperparameters for supported models
model_params = {
    "rf": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    },
    "xgb": {
        "n_estimators": 100,  # Number of boosting rounds (trees to build)
        "max_depth": 5,  # Maximum depth of each decision tree
        "learning_rate": 0.05,  # Step size shrinkage
        "subsample": 0.8,  # Fraction of training samples used per boosting round
        "colsample_bytree": 0.8,  # Fraction of features (columns) used per tree
        "eval_metric": "aucpr",  # Evaluation metric
        "random_state": 42,  # Random seed for reproducibility
        "scale_pos_weight": 5,  # Weighting factor to handle class imbalance
    },
    "lgbm": {
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": 5,
    },
    "cat": {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.05,
        "eval_metric": "F1",
        "random_seed": 42,
        "verbose": 0,
        "scale_pos_weight": 5,
        "train_dir": None,
    },
}


def get_model(model_type):
    """
    Return an instance of a tree-based model with predefined parameters.
    """
    if model_type == "rf":
        return RandomForestClassifier(**model_params["rf"])
    elif model_type == "xgb":
        return XGBClassifier(**model_params["xgb"])
    elif model_type == "lgbm":
        return LGBMClassifier(**model_params["lgbm"])
    elif model_type == "cat":
        return CatBoostClassifier(**model_params["cat"])
    else:
        raise ValueError("Invalid model_type. Choose 'rf', 'xgb', 'lgbm', or 'cat'.")
