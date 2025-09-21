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
    if config is None:
        config = get_feature_config()

    all_new_features = []

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
