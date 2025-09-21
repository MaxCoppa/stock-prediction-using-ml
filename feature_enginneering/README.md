## feature_engineering Submodule

This submodule provides utilities for creating new features from financial or tabular datasets.  
It includes transformations, binning, group-based statistics, ratios, products, indicators, and row-level statistics.

---

### Workflow

1. Define feature configuration (via `config_features.py` or a custom config).
2. Use `create_new_features()` to apply all selected feature engineering steps.
3. New features are added directly to the dataframe and a list of their names is returned.

---

### Main Components

- **create_features.py**
  - `create_new_features(df, config=None)`  
    Orchestrates the entire feature engineering pipeline based on a configuration dictionary.

- **add_features.py**
  - `add_transform_features` – log/sqrt transformations  
  - `add_binning_features` – discretization with `qcut` or `cut`  
  - `add_group_features` – group-level statistics and PCA  
  - `add_ratio_features` – ratios of variable pairs  
  - `add_product_features` – products of variable pairs  
  - `add_indicator_features` – missingness, threshold, and nonzero indicators  
  - `add_row_stats` – row-level summary statistics  

- **shift_features.py**
  - `create_shifted_features` – create lagged features and group-based statistics for shifted variables.

- **config_features.py**
  - Defines the default configuration for which variables are used in each feature engineering step.

---

### Customization

- **Changing which features are created**  
  Edit `config_features.py` to update variable groups, ratios, products, thresholds, etc.  
  You can also pass a custom config dictionary directly into `create_new_features()`.

- **Adding new feature types**  
  To introduce a new transformation:
  1. Implement a new function in `add_features.py` (or a new file).  
  2. Update `create_features.py` to call this new function.  
  3. Add the necessary configuration options in `config_features.py`.

- **Shifts and temporal features**  
  Use `create_shifted_features()` to create lag-based features.  
  The grouping keys (`gb_features`) and the target feature can be customized.

---
