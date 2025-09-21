# Stock Prediction Using Machine Learning

This repository investigates **machine learning methods for stock price prediction**.  
It provides a reproducible pipeline for experimenting with feature engineering,  
tree-based models, and ensemble methods on financial time series.

Stock prediction is notoriously challenging due to noisy signals,  
non-stationarity, and changing market conditions.  
This project does not claim to “solve” stock forecasting,  
but rather offers a structured framework to:

- **Test machine learning approaches** on financial datasets,  
- **Compare models and features** under consistent evaluation,  
- **Generalize experiments** to different assets and time periods,  
- **Provide building blocks** (modules + notebooks) for further research.  

All modules are designed to be flexible, so the same pipeline can be applied to other financial assets or extended to new data sources.  

It is structured as a set of **reusable modules** (for feature engineering, model training, evaluation, and ensembling)  
and **notebooks** (for experiments and analysis).  
Users must provide their own datasets (e.g., historical stock price data).

---

## Machine Learning Focus

The project addresses the predictive modeling pipeline in a structured way:

- **Feature Engineering**  
  Creation of derived features to improve signal extraction in noisy financial data:  
  transformations, ratios, products, binning, group statistics, PCA, row-level stats, and lagged features.  

- **Predictive Modeling**  
  Benchmarking of tree-based classifiers widely used in financial tabular problems:  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - CatBoost  

- **Model Selection and Evaluation**  
  K-Fold cross-validation on grouped data to assess stability and avoid temporal leakage.  
  Metrics: Accuracy, ROC AUC, and F1.  

- **Ensembling**  
  Combination of multiple models through probability averaging, with extensibility towards weighted ensembles or stacking.  

This design isolates preprocessing, feature engineering, and modeling into independent modules,  
supporting transparent experimentation and reproducibility.

---

## Project Structure

```text
├── README.md                   # Project documentation
├── requirements.txt             # Dependencies
├── feature_enginneering/        # Feature engineering utilities
├── tree_based_models/           # Model training, evaluation, and ensembling
├── model_comparaison.py         # Notebook script: single model training & evaluation
├── ensemble_comparaison.py      # Notebook script: ensemble training & evaluation
└── data_visualisation.ipynb     # Exploratory data analysis and visualization
```
---

## Usage

Experiments are provided as **notebooks/scripts**.  
At this stage, the workflow is exploratory rather than a packaged CLI.

### Single Model Workflow
Notebook: `model_comparaison.py`  
Covers feature engineering → cross-validation → model training → evaluation.

### Ensemble Workflow
Notebook: `ensemble_comparaison.py`  
Trains multiple models and evaluates ensemble performance.

### Data Visualization
Notebook: `data_visualisation.ipynb`  
Explores raw and engineered features.

> **Note**: Users must provide their own dataset (e.g., `train.csv`, `test.csv`) in a suitable format.  
A typical dataset contains:  
- an identifier column (e.g., ID or DATE),  
- multiple feature columns,  
- one target column.

---

## Generalization and Limitations

The pipeline is designed to **generalize across stock datasets**:  
- Modular feature creation without hard-coded stock-specific assumptions.  
- Configurable model initialization to adapt to different market data.  
- Grouped cross-validation for robust generalization estimates.  

Limitations include:  
- Dependence on tree-based methods (no deep learning yet).  
- Probability averaging as the only ensembling strategy.  
- Manual feature configuration required by the user.  

---

## Future Directions

Potential extensions include:  
- Integration of deep learning methods for sequential modeling (LSTMs, Transformers).  
- Real-time stock data ingestion from APIs.  
- Automated hyperparameter optimization (Optuna, Ray Tune).  
- Backtesting and evaluation of trading strategies.  
- Advanced ensembling (weighted voting, stacking, blending).  

---

