import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
try:
    X_train = pd.read_csv("data/processed/splits/X_train.csv")
    X_test = pd.read_csv("data/processed/splits/X_test.csv")
    y_train = pd.read_csv("data/processed/splits/y_train.csv")
    y_test = pd.read_csv("data/processed/splits/y_test.csv")
    data = pd.read_csv("data/processed/new_features2.csv")
    logging.info("Data loaded successfully")
    logging.info(f"X_train columns: {X_train.columns.tolist()}")
    logging.info(f"Data columns: {data.columns.tolist()}")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logging.info(f"Data shape: {data.shape}")
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise

# Ensure dates are in datetime format
data['date'] = pd.to_datetime(data['date'])

# Check for NaNs in data
logging.info(f"NaNs in X_train: {X_train.isna().sum().sum()}")
logging.info(f"NaNs in y_train: {y_train.isna().sum().sum()}")
logging.info(f"NaNs in data: {data.isna().sum().sum()}")

# Time-series cross-validation setup
n_folds = 3
train_size = int(0.6 * len(X_train))  # 60% of training data
val_size = int(0.2 * len(X_train))    # 20% for validation
step_size = (len(X_train) - train_size - val_size) // (n_folds - 1)
logging.info(f"Cross-validation setup: {n_folds} folds, train_size={train_size}, val_size={val_size}")

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, force_row_wise=True)
}

# Select key features for Prophet
prophet_regressors = [
    'rfh', 'r1h', 'r3h', 'rfh_avg', 'r1h_avg', 'r3h_avg', 'rfq', 'r1q', 'r3q',
    'month_sin', 'month_cos', 'rfq_norm', 'r1q_norm', 'r3q_norm',
    'rfh_lag_1', 'rfh_lag_7', 'rfh_lag_30', 'r1h_lag_1', 'r1h_lag_7', 'r3h_lag_1',
    'rfh_roll_mean_7', 'rfh_roll_std_7', 'r1h_roll_mean_7', 'r1h_roll_std_7'
]

# Function to plot actual vs predicted
def plot_predictions(dates, y_true, predictions, target, output_dir="plots"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
        for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
            plt.subplot(2, 2, i)
            plt.plot(dates, y_true, label="Actual", color="blue", alpha=0.7)
            plt.plot(dates, y_pred, label=f"Predicted ({model_name})", color="orange", alpha=0.7)
            plt.title(f"{model_name} - {target}")
            plt.xlabel("Date")
            plt.ylabel("Rainfall (mm)")
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{target}_predictions.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot for {target} to {plot_path}")
    except Exception as e:
        logging.error(f"Error in plotting for {target}: {e}")

# Evaluate models for target_monthly
target = "target_monthly"
logging.info(f"\nEvaluating models for {target}")
y_train_target = y_train[target].values
y_test_target = y_test[target].values
test_dates = data["date"].iloc[len(X_train):].reset_index(drop=True)

# Cross-validation results
cv_results = {name: [] for name in models.keys()}
cv_results["Prophet"] = []

# Time-series cross-validation
for fold in range(n_folds):
    start_train = fold * step_size
    end_train = start_train + train_size
    start_val = end_train
    end_val = start_val + val_size
    logging.info(f"Fold {fold+1} - Indices: train [{start_train}:{end_train}], val [{start_val}:{end_val}]")

    X_train_fold = X_train.iloc[start_train:end_train]
    y_train_fold = y_train_target[start_train:end_train]
    X_val_fold = X_train.iloc[start_val:end_val]
    y_val_fold = y_train_target[start_val:end_val]

    # Train and evaluate sklearn models
    for name, model in models.items():
        try:
            logging.info(f"Fold {fold+1} - Training {name}")
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, y_pred)
            cv_results[name].append(mae)
            logging.info(f"Fold {fold+1} - {name} MAE: {mae:.4f}")
        except Exception as e:
            logging.error(f"Error in {name} for fold {fold+1}: {e}")

    # Prophet
    try:
        logging.info(f"Fold {fold+1} - Training Prophet")
        available_columns = [col for col in prophet_regressors if col in data.columns and col in X_train.columns]
        if not available_columns:
            logging.warning(f"No common columns for Prophet in {target}, skipping Prophet")
            continue
        prophet_df = data[["date", target] + available_columns].rename(columns={"date": "ds", target: "y"})
        prophet_train = prophet_df.iloc[start_train:end_train].copy()
        prophet_val = prophet_df.iloc[start_val:end_val].copy()
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        for col in available_columns:
            prophet_model.add_regressor(col)
        prophet_model.fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=len(prophet_val), freq="M")
        future = future.merge(data[["date"] + available_columns], left_on="ds", right_on="date", how="left")
        if future[available_columns].isna().any().any():
            logging.warning(f"NaNs detected in future DataFrame for {target}, filling with 0")
            future[available_columns] = future[available_columns].fillna(0)
        forecast = prophet_model.predict(future)
        mae = mean_absolute_error(prophet_val["y"], forecast["yhat"].iloc[-len(prophet_val):])
        cv_results["Prophet"].append(mae)
        logging.info(f"Fold {fold+1} - Prophet MAE: {mae:.4f}")
    except Exception as e:
        logging.error(f"Error in Prophet for fold {fold+1}: {e}")

# Print cross-validation results
logging.info(f"\nCross-validation results for {target} (Average MAE):")
for name in cv_results:
    avg_mae = np.mean(cv_results[name]) if cv_results[name] else float('nan')
    logging.info(f"{name}: {avg_mae:.4f}")

# Train final models on full training data and evaluate on test set
logging.info(f"\nFinal evaluation on test set for {target}")
predictions = {}
for name, model in models.items():
    try:
        logging.info(f"Training final {name}")
        model.fit(X_train, y_train_target)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        mae = mean_absolute_error(y_test_target, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_target, y_pred))
        logging.info(f"{name} Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
    except Exception as e:
        logging.error(f"Error in final {name} evaluation: {e}")

# Prophet final model
try:
    logging.info(f"Training final Prophet")
    available_columns = [col for col in prophet_regressors if col in data.columns and col in X_train.columns]
    if not available_columns:
        logging.warning(f"No common columns for Prophet in {target}, skipping Prophet")
    else:
        prophet_df = data[["date", target] + available_columns].rename(columns={"date": "ds", target: "y"})
        prophet_train = prophet_df.iloc[:len(X_train)].copy()
        prophet_test = prophet_df.iloc[len(X_train):].copy()
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        for col in available_columns:
            prophet_model.add_regressor(col)
        prophet_model.fit(prophet_train)
        future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq="M")
        future = future.merge(data[["date"] + available_columns], left_on="ds", right_on="date", how="left")
        if future[available_columns].isna().any().any():
            logging.warning(f"NaNs detected in future DataFrame for {target}, filling with 0")
            future[available_columns] = future[available_columns].fillna(0)
        forecast = prophet_model.predict(future)
        y_pred_prophet = forecast["yhat"].iloc[-len(prophet_test):].values
        predictions["Prophet"] = y_pred_prophet
        mae = mean_absolute_error(prophet_test["y"], y_pred_prophet)
        rmse = np.sqrt(mean_squared_error(prophet_test["y"], y_pred_prophet))
        logging.info(f"Prophet Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
except Exception as e:
    logging.error(f"Error in final Prophet evaluation: {e}")

# Visualize results
plot_predictions(test_dates, y_test_target, predictions, target)
