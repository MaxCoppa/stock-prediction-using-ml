from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

model_params = {
    "rf": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": 42,
        "scale_pos_weight": 5,
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
        "verbose": 0,  # suppress training logs
        "scale_pos_weight": 5,  # handle imbalance
        "train_dir": None,
    },
}


def get_model(model_type):
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
