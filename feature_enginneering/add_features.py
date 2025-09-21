import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def add_transform_features(df, vars):
    new_features = []
    for col in vars:
        df[f"log_{col}"] = np.log1p(df[col].fillna(0))
        df[f"sqrt_{col}"] = np.sqrt(df[col].fillna(0))
        new_features += [f"log_{col}", f"sqrt_{col}"]
    return df, new_features


def add_binning_features(df, bin_config):
    new_features = []
    for col, cfg in bin_config.items():
        if "qcut" in cfg:
            df[f"{col}_bin"] = pd.qcut(
                df[col], q=cfg["qcut"], duplicates="drop"
            ).cat.codes
        elif "cut" in cfg:
            df[f"{col}_bin"] = pd.cut(df[col], bins=cfg["cut"], labels=False)
        new_features.append(f"{col}_bin")
    return df, new_features


def add_group_features(df, group_vars):
    new_features = []
    grp = df[group_vars].fillna(0)
    df[f"grp_{'_'.join(group_vars)}_mean"] = grp.mean(axis=1)
    df[f"grp_{'_'.join(group_vars)}_min"] = grp.min(axis=1)
    df[f"grp_{'_'.join(group_vars)}_max"] = grp.max(axis=1)
    df[f"grp_{'_'.join(group_vars)}_std"] = grp.std(axis=1)
    new_features += [
        f"grp_{'_'.join(group_vars)}_mean",
        f"grp_{'_'.join(group_vars)}_min",
        f"grp_{'_'.join(group_vars)}_max",
        f"grp_{'_'.join(group_vars)}_std",
    ]

    pca = PCA(n_components=1, random_state=42)
    df[f"grp_{'_'.join(group_vars)}_pca1"] = pca.fit_transform(grp)
    new_features.append(f"grp_{'_'.join(group_vars)}_pca1")
    return df, new_features


def add_ratio_features(df, ratio_pairs):
    new_features = []
    for num, den in ratio_pairs:
        name = f"ratio_{num}_{den}"
        df[name] = df[num] / (1 + df[den].fillna(0))
        new_features.append(name)
    return df, new_features


def add_product_features(df, product_pairs):
    new_features = []
    for a, b in product_pairs:
        name = f"prod_{a}_{b}"
        df[name] = df[a] * df[b]
        new_features.append(name)
    return df, new_features


def add_indicator_features(df, config):
    new_features = []
    for col in config.get("isna", []):
        name = f"{col}_isna"
        df[name] = df[col].isna().astype(int)
        new_features.append(name)

    for col, thr in config.get("threshold", {}).items():
        name = f"is_{col}_high"
        df[name] = (df[col] > thr).astype(int)
        new_features.append(name)

    for col in config.get("nonzero", []):
        name = f"is_{col}_nonzero"
        df[name] = (df[col] > 0).astype(int)
        new_features.append(name)

    return df, new_features


def add_row_stats(df, cols):
    new_features = []
    df["row_sum"] = df[cols].sum(axis=1, skipna=True)
    df["row_mean"] = df[cols].mean(axis=1, skipna=True)
    df["row_std"] = df[cols].std(axis=1, skipna=True)
    df["nb_zeros"] = (df[cols] == 0).sum(axis=1)
    df["nb_nonzeros"] = len(cols) - df["nb_zeros"]
    new_features += ["row_sum", "row_mean", "row_std", "nb_zeros", "nb_nonzeros"]

    return df, new_features
