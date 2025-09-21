"""
shift_features.py

Feature engineering utilities for creating shifted/lagged features and 
aggregated statistics over groups.
"""

from typing import List, Tuple
import pandas as pd


def create_shifted_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    shifts: List[int] = [1],
    statistics: List[str] = ["mean"],
    gb_features: List[str] = ["SECTOR", "DATE"],
    target_feature: str = "RET",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Create group-based shifted features with aggregated statistics.

    For each specified shift and statistic, computes an aggregated value
    of the lagged target feature grouped by the given features.
    """
    new_features = []
    group_name = "_".join(gb_features)

    for shift in shifts:
        for stat in statistics:
            feat = f"{target_feature}_{shift}"
            name = f"{target_feature}_{shift}_{group_name}_{stat}"
            new_features.append(name)

            # Apply group-based aggregation consistently to train and test
            for data in [train, test]:
                data[name] = data.groupby(gb_features)[feat].transform(stat)

    return train, test, new_features
