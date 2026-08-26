# Model selection

The primary classifier is `RandomForestClassifier`. Grid search evaluates `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` and `max_features` using ROC-AUC. The selected parameters are stored in `metadata.json` and `metrics.json`.
