"""
Utility for combining multiple models into an ensemble prediction
using simple probability averaging.
"""

import numpy as np


def predict_ensembler_models(models, X, threshold: float = 0.5):
    """
    Perform an average ensemble of model predictions.
    """
    # Collect predicted probabilities from each model
    preds = [m.predict_proba(X)[:, 1] for m in models]

    # Average across models
    avg_proba = np.mean(preds, axis=0)

    # Convert to binary labels using threshold
    avg_preds = (avg_proba > threshold).astype(int)

    return avg_proba, avg_preds
