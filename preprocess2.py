# preprocess.py
import pandas as pd
import numpy as np

class RainfallPreprocessor:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Load dataset and parse dates"""
        self.df = pd.read_csv(self.filepath, parse_dates=["date"])
        self.df.sort_values("date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def feature_engineering(self):
        """Generate targets + temporal + lags + rolling + seasonal features"""
        df = self.df.copy()

        # -----------------------
        # 1. Target Variables
        # -----------------------
        df["target_weekly"] = df["rfh"].shift(-7)
        df["target_monthly"] = df["rfh"].shift(-30)
        df["target_yearly"] = df["rfh"].shift(-365)

        # -----------------------
        # 2. Temporal Features
        # -----------------------
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # -----------------------
        # 3. Lag Features (multiple horizons)
        # -----------------------
        lag_days = [1, 7, 30, 90, 365, 1825, 3650]  # 1d, 1w, 1m, 3m, 1y, 5y, 10y
        for col in ["rfh", "r1h", "r3h"]:
            for lag in lag_days:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # -----------------------
        # 4. Rolling Window Features
        # -----------------------
        rolling_windows = [7, 30, 90, 365]
        for col in ["rfh", "r1h", "r3h"]:
            for window in rolling_windows:
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window, min_periods=1).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window, min_periods=1).std()

        # -----------------------
        # 5. Seasonal Encoding
        # -----------------------
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # -----------------------
        # 6. Normalized Anomalies
        # -----------------------
        for col in ["rfq", "r1q", "r3q"]:
            df[f"{col}_norm"] = df[col] / 100.0

        # Drop rows with NaN introduced by shifts/rolling
        df.dropna(inplace=True)

        self.df = df

    def save(self, output_path: str):
        """Save preprocessed dataset"""
        self.df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess rainfall dataset")
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--output", required=True, help="Path to save processed CSV file")
    args = parser.parse_args()

    preprocessor = RainfallPreprocessor(args.input)
    preprocessor.load_data()
    preprocessor.feature_engineering()
    preprocessor.save(args.output)

    print(f"✅ Preprocessing complete. File saved to {args.output}")

