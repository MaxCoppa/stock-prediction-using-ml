from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from .ensemble import predict_ensembler_models


def evaluate_model(model, X, y, threshold: float = 0.5, verbose=False) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba > threshold).astype(int)

    results = {
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, preds),
    }

    if verbose:
        print(
            "Model evaluation:",
            " | ".join(f"{k}: {v:.3f}" for k, v in results.items()),
        )

    return results


def evaluate_ensemble_model(
    models, X, y, threshold: float = 0.5, verbose=False
) -> dict:
    avg_proba, avg_preds = predict_ensembler_models(
        models=models, X=X, threshold=threshold
    )
    results = {
        "f1": f1_score(y, avg_preds),
        "roc_auc": roc_auc_score(y, avg_proba),
        "accuracy": accuracy_score(y, avg_preds),
    }

    if verbose:
        print(
            "Ensemble evaluation:",
            " | ".join(f"{k}: {v:.3f}" for k, v in results.items()),
        )

    return results
