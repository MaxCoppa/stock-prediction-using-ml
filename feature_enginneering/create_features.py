"""
Pipeline for generating new features from a dataset.
Combines transformations, binning, group statistics, ratios, products,
indicators, and row-level statistics into a single interface.
"""

import pandas as pd

from .add_features import (
    add_group_features,
    add_product_features,
    add_indicator_features,
    add_binning_features,
    add_ratio_features,
    add_row_stats,
    add_transform_features,
)

from .config_features import get_feature_config


def create_new_features(df: pd.DataFrame, config: dict | None = None):
    """
    Apply a series of feature engineering steps to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    config : dict, optional
        Feature configuration dictionary. If None, defaults are loaded
        via `get_feature_config()`.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with new features added.
    all_new_features : list of str
        Names of all newly created features.
    """
    if config is None:
        config = get_feature_config()

    all_new_features = []

    # Apply transformations sequentially
    df, feats = add_transform_features(df, config["transform_vars"])
    all_new_features += feats

    df, feats = add_binning_features(df, config["bin_vars"])
    all_new_features += feats

    df, feats = add_group_features(df, config["group_vars"])
    all_new_features += feats

    df, feats = add_ratio_features(df, config["ratio_pairs"])
    all_new_features += feats

    df, feats = add_product_features(df, config["product_pairs"])
    all_new_features += feats

    df, feats = add_indicator_features(df, config["indicator_vars"])
    all_new_features += feats

    df, feats = add_row_stats(df, config["all_vars"])
    all_new_features += feats

    return df, all_new_features
