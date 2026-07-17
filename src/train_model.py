"""
train_model.py

Trains the final player projection model and saves it to models/ for reuse
by the draft simulator. Hyperparameters below were selected via a 30-trial
random search in notebooks/03_modeling.ipynb -- this script reproduces the
final chosen model, it doesn't re-run the search itself.

Usage:
    python src/train_model.py
"""

import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from build_features import build_features, get_feature_columns, SKILL_POSITIONS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Selected via random search -- see notebooks/03_modeling.ipynb Step 13.
BEST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
}

TRAIN_SEASONS_MAX = 2021   # train: 2015-2021
VAL_SEASONS = [2022, 2023]  # validation: 2022-2023
TEST_SEASON = 2024          # test: 2024 (untouched holdout)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading raw data...")
    seasonal = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'seasonal_player_stats.parquet'))
    rosters = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'seasonal_rosters.parquet'))
    injuries = pd.read_parquet(os.path.join(RAW_DATA_DIR, 'injuries.parquet'))

    print("Building features...")
    features = build_features(seasonal, rosters, injuries)
    features_encoded = pd.get_dummies(features, columns=['position'], prefix='pos')

    feature_cols = get_feature_columns(features)
    position_dummy_cols = [f'pos_{p}' for p in SKILL_POSITIONS]
    feature_cols_final = feature_cols + position_dummy_cols

    train = features_encoded[features_encoded['season'] <= TRAIN_SEASONS_MAX]
    val = features_encoded[features_encoded['season'].isin(VAL_SEASONS)]
    test = features_encoded[features_encoded['season'] == TEST_SEASON]

    X_train, y_train = train[feature_cols_final], train['fantasy_points_ppr']
    X_val, y_val = val[feature_cols_final], val['fantasy_points_ppr']
    X_test, y_test = test[feature_cols_final], test['fantasy_points_ppr']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("Training model...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='mae',
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    for name, X_split, y_split in [('Validation', X_val, y_val), ('Test', X_test, y_test)]:
        preds = model.predict(X_split)
        mae = mean_absolute_error(y_split, preds)
        rmse = mean_squared_error(y_split, preds) ** 0.5
        print(f"{name} -- MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    model_path = os.path.join(MODELS_DIR, 'xgb_projection_model.json')
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # Save the exact feature column list/order -- required to build a
    # matching input matrix at prediction time later.
    feature_list_path = os.path.join(MODELS_DIR, 'feature_columns.json')
    with open(feature_list_path, 'w') as f:
        json.dump(feature_cols_final, f, indent=2)
    print(f"Saved feature column list to {feature_list_path}")


if __name__ == '__main__':
    main()
