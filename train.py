# train_enhanced.py
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
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

# ---------- Enhanced Feature Engineering ----------
def create_enhanced_features(df, target_col):
    """Create comprehensive feature set for rainfall prediction"""
    df = df.copy()
    
    # Time-based features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['season'] = ((df['month'] - 1) // 3) % 4
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Multiple lag features
    lags = [1, 2, 3, 5, 7, 10, 14, 21, 30]
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    windows = [3, 7, 14, 21, 30, 60, 90]
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
    
    # Exponential weighted moving averages
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        df[f'ewm_{alpha}'] = df[target_col].ewm(alpha=alpha).mean()
    
    # Trend and change features (with safe operations)
    def safe_trend(x):
        try:
            if len(x) > 1 and not x.isnull().all():
                slope = np.polyfit(range(len(x)), x, 1)[0]
                return slope if np.isfinite(slope) else 0
            return 0
        except:
            return 0
    
    df['trend_7'] = df[target_col].rolling(7, min_periods=2).apply(safe_trend, raw=False)
    
    # Safe percentage changes
    df['pct_change_1'] = df[target_col].pct_change(1)
    df['pct_change_7'] = df[target_col].pct_change(7)
    
    # Drought/wet period indicators
    q10 = df[target_col].quantile(0.1)
    q90 = df[target_col].quantile(0.9)
    df['dry_days'] = (df[target_col] < q10).astype(int)
    df['wet_days'] = (df[target_col] > q90).astype(int)
    
    # Statistical features (with safe operations)
    rolling_mean_30 = df[target_col].rolling(30, min_periods=1).mean()
    rolling_std_30 = df[target_col].rolling(30, min_periods=1).std()
    
    df['zscore'] = (df[target_col] - rolling_mean_30) / (rolling_std_30 + 1e-8)
    df['percentile_rank'] = df[target_col].rolling(30, min_periods=1).rank(pct=True)
    
    # Clean up any remaining issues
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Final safety check - clip extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'date':
            # Clip to reasonable bounds
            q1, q99 = df[col].quantile([0.01, 0.99])
            if q99 > q1:  # Avoid issues with constant columns
                df[col] = df[col].clip(q1, q99)
    
    return df

# ---------- Enhanced Models ----------
def create_enhanced_lstm(n_input, n_features):
    """Create improved LSTM with regularization"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(n_input, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

def create_gru_model(n_input, n_features):
    """Create GRU model as alternative to LSTM"""
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(n_input, n_features)),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',
        metrics=['mae']
    )
    return model

# ---------- Enhanced Evaluation ----------
def enhanced_evaluate(y_true, y_pred):
    """Comprehensive evaluation metrics"""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
        "max_error": float(np.max(np.abs(y_true - y_pred)))
    }

def plot_enhanced_results(y_true, y_pred, model_name):
    """Enhanced plotting with diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0,0].plot(y_true.index, y_true, label="Actual", alpha=0.7)
    axes[0,0].plot(y_true.index, y_pred, label="Predicted", alpha=0.7)
    axes[0,0].set_title(f"{model_name} - Time Series")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0,1].scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0,1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0,1].set_xlabel('Actual')
    axes[0,1].set_ylabel('Predicted')
    axes[0,1].set_title(f"{model_name} - Scatter Plot")
    axes[0,1].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.6)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Predicted')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title(f"{model_name} - Residuals")
    axes[1,0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1,1].hist(residuals, bins=30, alpha=0.7, density=True)
    axes[1,1].axvline(residuals.mean(), color='r', linestyle='--', 
                     label=f'Mean: {residuals.mean():.2f}')
    axes[1,1].set_title(f"{model_name} - Error Distribution")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ---------- Enhanced Training Pipeline ----------
def run_enhanced_models(df, target_col):
    """Enhanced model training with comprehensive features"""
    
    # Feature engineering
    logging.info("Creating enhanced features...")
    df_features = create_enhanced_features(df, target_col)
    
    # Select features (excluding date and target)
    feature_cols = [col for col in df_features.columns if col not in ['date', target_col]]
    logging.info(f"Created {len(feature_cols)} features")
    
    X = df_features[feature_cols].values
    y = df_features[target_col].values
    
    # Additional safety checks
    logging.info("Performing data quality checks...")
    
    # Check for infinite values
    inf_mask = np.isinf(X).any(axis=1)
    if inf_mask.sum() > 0:
        logging.warning(f"Found {inf_mask.sum()} rows with infinite values, removing...")
        X = X[~inf_mask]
        y = y[~inf_mask]
        df_features = df_features[~inf_mask].reset_index(drop=True)
    
    # Check for extreme values
    extreme_mask = np.abs(X) > 1e10
    if extreme_mask.any():
        logging.warning("Found extreme values, clipping...")
        X = np.clip(X, -1e10, 1e10)
    
    # Check for all-NaN columns
    nan_cols = np.isnan(X).all(axis=0)
    if nan_cols.any():
        logging.warning(f"Removing {nan_cols.sum()} all-NaN columns")
        good_cols = ~nan_cols
        X = X[:, good_cols]
        feature_cols = [col for i, col in enumerate(feature_cols) if good_cols[i]]
    
    # Final NaN filling
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    logging.info(f"Final data shape: {X.shape}, Target shape: {y.shape}")
    
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
    
    # Enhanced traditional models
    models_to_try = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, 
            learning_rate=0.1, 
            max_depth=4,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    }
    
    for name, model in models_to_try.items():
        logging.info(f"Training {name}...")
        
        if name == 'XGBoost':
            try:
                # Try the newer XGBoost API first
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    verbose=False
                )
            except TypeError:
                # Fallback for older XGBoost versions
                model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train_scaled, y_train)
            
        preds = model.predict(X_test_scaled)
        metrics_summary[name] = enhanced_evaluate(y_test, preds)
        
        # Create plots
        fig = plot_enhanced_results(
            pd.Series(y_test, index=df_features.index[split_idx:]), 
            pd.Series(preds, index=df_features.index[split_idx:]), 
            name
        )
        figs.append(fig)
    
    # Enhanced Deep Learning Models
    logging.info("Training deep learning models...")
    n_input = 14  # Increased sequence length
    n_features = X_train_scaled.shape[1]
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]
    
    # Enhanced LSTM
    if len(X_train_scaled) > n_input:
        lstm_model = create_enhanced_lstm(n_input, n_features)
        
        # Create sequences
        train_gen = TimeseriesGenerator(
            X_train_scaled, y_train, 
            length=n_input, batch_size=32
        )
        val_gen = TimeseriesGenerator(
            X_test_scaled, y_test, 
            length=n_input, batch_size=32
        )
        
        history = lstm_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        test_gen = TimeseriesGenerator(X_test_scaled, y_test, length=n_input, batch_size=1)
        preds = lstm_model.predict(test_gen).flatten()
        y_eval = y_test[n_input:]
        
        metrics_summary['LSTM_Enhanced'] = enhanced_evaluate(y_eval, preds)
        
        fig = plot_enhanced_results(
            pd.Series(y_eval, index=df_features.index[split_idx+n_input:]), 
            pd.Series(preds, index=df_features.index[split_idx+n_input:]), 
            'LSTM_Enhanced'
        )
        figs.append(fig)
        
        # GRU Model
        gru_model = create_gru_model(n_input, n_features)
        gru_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=50,
            callbacks=callbacks,
            verbose=0
        )
        
        preds_gru = gru_model.predict(test_gen).flatten()
        metrics_summary['GRU'] = enhanced_evaluate(y_eval, preds_gru)
        
        fig = plot_enhanced_results(
            pd.Series(y_eval, index=df_features.index[split_idx+n_input:]), 
            pd.Series(preds_gru, index=df_features.index[split_idx+n_input:]), 
            'GRU'
        )
        figs.append(fig)
    
    return metrics_summary, figs, feature_cols

# ---------- Main Function ----------
def main():
    # Load clean CSV
    input_file = "data/processed/rainfall_clean.csv"
    
    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return
    
    df = pd.read_csv(input_file, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    logging.info(f"Loaded data: {df.shape}")
    
    # Create enhanced targets
    df["daily_target"] = df["rfh"]
    df["monthly_target"] = df["r1h"]
    df["yearly_target"] = df["r3h"]
    
    horizons = {
        "daily": df[["date", "daily_target"]].copy(),
        "monthly": df[["date", "monthly_target"]].copy(),
        "yearly": df[["date", "yearly_target"]].copy()
    }
    
    os.makedirs("reports", exist_ok=True)
    consolidated_pdf_path = os.path.join("reports", "enhanced_rainfall_report.pdf")
    
    with PdfPages(consolidated_pdf_path) as pdf:
        all_metrics = {}
        
        for horizon_name, data in horizons.items():
            logging.info(f"=== Training enhanced models on {horizon_name} data ===")
            
            target_col = data.columns[1]  # Get target column name
            
            metrics, figs, features = run_enhanced_models(data, target_col)
            all_metrics[horizon_name] = metrics
            
            # Save plots
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
    
    # Save results
    with open(os.path.join("reports", "enhanced_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    # Create comprehensive metrics DataFrame
    metrics_data = []
    for horizon in all_metrics.keys():
        for model in all_metrics[horizon].keys():
            for metric in all_metrics[horizon][model].keys():
                metrics_data.append({
                    'horizon': horizon,
                    'model': model,
                    'metric': metric,
                    'value': all_metrics[horizon][model][metric]
                })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_pivot = metrics_df.pivot_table(
        index=['horizon', 'model'], 
        columns='metric', 
        values='value'
    )
    metrics_pivot.to_csv(os.path.join("reports", "enhanced_metrics.csv"))
    
    # Create summary report
    summary_report = []
    for horizon in all_metrics.keys():
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
    summary_df.to_csv(os.path.join("reports", "model_summary.csv"), index=False)
    
    logging.info("=== ENHANCED ANALYSIS COMPLETED ===")
    logging.info(f"Reports saved in: {consolidated_pdf_path}")
    logging.info("Check enhanced_metrics.csv and model_summary.csv for detailed results")
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
