# Training pipeline

Supply a CSV containing coordinates, the configured environmental predictors and a binary `tick_presence` target. The pipeline validates columns, removes invalid coordinates/targets, imputes numeric values and encodes categories within an sklearn pipeline. It uses a stratified train/test split, GridSearchCV and cross-validation before writing artifacts to `data/ml/artifacts`.
