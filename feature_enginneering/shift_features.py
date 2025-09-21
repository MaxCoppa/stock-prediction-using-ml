from typing import List, Tuple
import pandas as pd

"""
Feature Engineering

The main drawback in this challenge would be to deal with the noise. 
To do that, we could create some feature that aggregate features with some statistics. 

The following cell computes statistics on a given target conditionally to some features. 
For example, we want to generate a feature that describe the mean of `RET_1` conditionally to the `SECTOR` and the `DATE`.
"""


def create_shifted_features(
    train,
    test,
    shifts: List[str] = [1],
    statistics: List[str] = ["mean"],
    gb_features: List[str] = ["SECTOR", "DATE"],
    target_feature: str = "RET",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    new_features = []

    tmp_name = "_".join(gb_features)

    for shift in shifts:
        for stat in statistics:
            name = f"{target_feature}_{shift}_{tmp_name}_{stat}"
            feat = f"{target_feature}_{shift}"
            new_features.append(name)
            for data in [train, test]:
                data[name] = data.groupby(gb_features)[feat].transform(
                    stat
                )  # groupby.transform creates a DataFrame same indexes as the initial

    return train, test, new_features
