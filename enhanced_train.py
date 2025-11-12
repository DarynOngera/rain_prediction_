# train_optimized.py - Fast version with smart sampling and reduced models
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------- Optimized Feature Engineering ----------
def create_optimized_features(df, target_col):
    """Create essential features only - focus on most important ones"""
    df = df.copy()
    
    # Essential time features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding (most important for seasonality)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Key lag features (most predictive)
    key_lags = [1, 7, 30]  # Reduced from 9 to 3
    for lag in key_lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Essential rolling stats (reduced windows)
    key_windows = [7, 30]  # Reduced from 7 to 2
    for window in key_windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    
    # Key trend indicators
    df['pct_change_1'] = df[target_col].pct_change(1)
    df['pct_change_7'] = df[target_col].pct_change(7)
    
    # Essential weather patterns
    df['dry_days'] = (df[target_col] < df[target_col].quantile(0.1)).astype(int)
    df['wet_days'] = (df[target_col] > df[target_col].quantile(0.9)).astype(int)
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

# ---------- Smart Sampling ----------
def smart_sample(df, sample_size=50000, stratify_by_season=True):
    """Intelligently sample data to maintain representativeness"""
    if len(df) <= sample_size:
        return df
    
    logging.info(f"Sampling {sample_size} points from {len(df)} total points...")
    
    if stratify_by_season:
        # Sample proportionally from each month to maintain seasonal patterns
        df['month'] = df['date'].dt.month if 'month' not in df.columns else df['month']
        sampled_dfs = []
        
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            month_sample_size = int(sample_size * len(month_data) / len(df))
            if month_sample_size > 0:
                if len(month_data) > month_sample_size:
                    month_sample = month_data.sample(n=month_sample_size, random_state=42)
                else:
                    month_sample = month_data
                sampled_dfs.append(month_sample)
        
        return pd.concat(sampled_dfs, ignore_index=True).sort_values('date').reset_index(drop=True)
    else:
        return df.sample(n=sample_size, random_state=42).sort_values('date').reset_index(drop=True)

# ---------- Fast Models Only ----------
def create_fast_lstm(n_input, n_features):
    """Smaller, faster LSTM"""
    model = Sequential([
        LSTM(32, input_shape=(n_input, n_features)),  # Reduced from 128
        Dropout(0.2),
        Dense(16, activation='relu'),  # Reduced from 32
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

# ---------- Enhanced Evaluation ----------
def enhanced_evaluate(y_true, y_pred):
    """Comprehensive evaluation metrics"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    }

def plot_quick_results(y_true, y_pred, model_name):
    """Quick 2-panel plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time series plot
    axes[0].plot(y_true.index[-1000:], y_true.iloc[-1000:], label="Actual", alpha=0.7)
    axes[0].plot(y_true.index[-1000:], y_pred[-1000:], label="Predicted", alpha=0.7)
    axes[0].set_title(f"{model_name} - Last 1000 Points")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=1)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f"{model_name} - Scatter")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ---------- Optimized Training Pipeline ----------
def run_optimized_models(df, target_col):
    """Fast training with essential models only"""
    
    # Smart sampling for large datasets
    if len(df) > 75000:
        df_sample = smart_sample(df, sample_size=50000)
        logging.info(f"Using sample of {len(df_sample)} points for training")
    else:
        df_sample = df.copy()
        logging.info(f"Using full dataset of {len(df)} points")
    
    # Optimized feature engineering
    logging.info("Creating features...")
    df_features = create_optimized_features(df_sample, target_col)
    
    # Select features
    feature_cols = [col for col in df_features.columns if col not in ['date', target_col]]
    logging.info(f"Created {len(feature_cols)} features for training")
    
    X = df_features[feature_cols].values
    y = df_features[target_col].values
    
    # Data quality checks
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Time series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    metrics_summary = {}
    figs = []
    
    # Fast models only - focus on best performers
    models_to_try = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest_Fast': RandomForestRegressor(
            n_estimators=100,  # Reduced from 200
            max_depth=15,     # Reduced from 15
            n_jobs=-1,        # Use all cores
            random_state=42
        ),
        'GradientBoosting_Fast': GradientBoostingRegressor(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.15, # Increased for faster convergence
            max_depth=3,      # Reduced from 4
            random_state=42
        ),
        'XGBoost_Fast': XGBRegressor(
            n_estimators=100,  # Reduced from 200
            learning_rate=0.15, # Increased
            max_depth=4,      # Reduced
            n_jobs=-1,        # Use all cores
            random_state=42
        )
    }
    
    for name, model in models_to_try.items():
        logging.info(f"Training {name}...")
        start_time = pd.Timestamp.now()
        
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        metrics_summary[name] = enhanced_evaluate(y_test, preds)
        
        # Create quick plots
        fig = plot_quick_results(
            pd.Series(y_test, index=df_features.index[split_idx:]), 
            preds, 
            name
        )
        figs.append(fig)
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logging.info(f"{name} completed in {elapsed:.1f} seconds - R²: {metrics_summary[name]['r2']:.4f}")
    
    # Fast LSTM (only if dataset is reasonable size)
    if len(X_train_scaled) > 100 and len(X_train_scaled) < 30000:
        logging.info("Training Fast LSTM...")
        start_time = pd.Timestamp.now()
        
        n_input = 7  # Reduced from 14
        n_features = X_train_scaled.shape[1]
        
        lstm_model = create_fast_lstm(n_input, n_features)
        callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        
        # Create sequences
        train_gen = TimeseriesGenerator(X_train_scaled, y_train, length=n_input, batch_size=64)
        val_gen = TimeseriesGenerator(X_test_scaled, y_test, length=n_input, batch_size=64)
        
        history = lstm_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,  # Reduced from 50
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        test_gen = TimeseriesGenerator(X_test_scaled, y_test, length=n_input, batch_size=1)
        preds = lstm_model.predict(test_gen, verbose=0).flatten()
        y_eval = y_test[n_input:]
        
        metrics_summary['LSTM_Fast'] = enhanced_evaluate(y_eval, preds)
        
        fig = plot_quick_results(
            pd.Series(y_eval, index=df_features.index[split_idx+n_input:]), 
            preds, 
            'LSTM_Fast'
        )
        figs.append(fig)
        
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logging.info(f"LSTM_Fast completed in {elapsed:.1f} seconds - R²: {metrics_summary['LSTM_Fast']['r2']:.4f}")
    else:
        logging.info("Skipping LSTM (dataset too large or small for fast training)")
    
    return metrics_summary, figs, feature_cols

# ---------- Main Function ----------
def main():
    total_start = pd.Timestamp.now()
    
    # Load clean CSV
    input_file = "data/processed/rainfall_clean.csv"
    
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logging.info(f"Loaded data: {df.shape}")
    
    # Create targets
    df["daily_target"] = df["rfh"]
    df["monthly_target"] = df["r1h"] 
    df["yearly_target"] = df["r3h"]
    
    horizons = {
        "daily": df[["date", "daily_target"]].copy(),
        "monthly": df[["date", "monthly_target"]].copy(),
        "yearly": df[["date", "yearly_target"]].copy()
    }
    
    os.makedirs("reports", exist_ok=True)
    consolidated_pdf_path = os.path.join("reports", "optimized_rainfall_report.pdf")
    
    with PdfPages(consolidated_pdf_path) as pdf:
        all_metrics = {}
        
        for horizon_name, data in horizons.items():
            logging.info(f"=== Training optimized models on {horizon_name} data ===")
            horizon_start = pd.Timestamp.now()
            
            target_col = data.columns[1]
            metrics, figs, features = run_optimized_models(data, target_col)
            all_metrics[horizon_name] = metrics
            
            # Save plots
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            elapsed = (pd.Timestamp.now() - horizon_start).total_seconds()
            logging.info(f"{horizon_name} horizon completed in {elapsed/60:.1f} minutes")
    
    # Save results
    with open(os.path.join("reports", "optimized_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    # Create summary
    summary_report = []
    for horizon in all_metrics.keys():
        if all_metrics[horizon]:  # Check if any models were trained
            best_model = min(all_metrics[horizon].keys(), 
                            key=lambda x: all_metrics[horizon][x]['rmse'])
            summary_report.append({
                'horizon': horizon,
                'best_model': best_model,
                'rmse': all_metrics[horizon][best_model]['rmse'],
                'mae': all_metrics[horizon][best_model]['mae'],
                'r2': all_metrics[horizon][best_model]['r2'],
                'mape': all_metrics[horizon][best_model]['mape']
            })
    
    summary_df = pd.DataFrame(summary_report)
    summary_df.to_csv(os.path.join("reports", "optimized_summary.csv"), index=False)
    
    total_elapsed = (pd.Timestamp.now() - total_start).total_seconds()
    
    logging.info("=== OPTIMIZED ANALYSIS COMPLETED ===")
    logging.info(f"Total runtime: {total_elapsed/60:.1f} minutes (vs ~150 minutes original)")
    logging.info(f"Reports saved in: {consolidated_pdf_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZED MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Runtime: {total_elapsed/60:.1f} minutes")
    print("="*60)
    if len(summary_df) > 0:
        print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
