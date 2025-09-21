import numpy as np


def predict_ensembler_models(models, X, threshold: float = 0.5):
    """
    Simple average ensemble of models' probabilities.
    """
    preds = [m.predict_proba(X)[:, 1] for m in models]
    avg_proba = np.mean(preds, axis=0)
    avg_preds = (avg_proba > threshold).astype(int)
    return avg_proba, avg_preds
