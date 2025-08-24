# preprocess.py
import os
import argparse
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    def load_data(self):
        print(f"[INFO] Loading dataset from {self.input_file} ...")
        self.df = pd.read_csv(self.input_file, parse_dates=["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)
        print(f"[INFO] Data shape after load: {self.df.shape}")

    def add_targets(self):
        # Predict future rainfall based on rfh (10-day rainfall)
        self.df["target_weekly"] = self.df["rfh"].shift(-7)
        self.df["target_monthly"] = self.df["rfh"].shift(-30)
        self.df["target_yearly"] = self.df["rfh"].shift(-365)
        print("[OK] Target variables created")

    def add_temporal_features(self):
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day"] = self.df["date"].dt.day
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["is_weekend"] = self.df["day_of_week"].isin([5,6]).astype(int)

        # Cyclical encoding for month
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month"] / 12)

        print("[OK] Temporal features created")

    def add_lag_features(self):
        self.df["rainfall_lag_1"] = self.df["rfh"].shift(1)
        self.df["rainfall_lag_7"] = self.df["rfh"].shift(7)
        print("[OK] Lag features created")

    def add_rolling_features(self):
        self.df["rainfall_roll_mean_7"] = self.df["rfh"].rolling(window=7).mean()
        self.df["rainfall_roll_std_7"] = self.df["rfh"].rolling(window=7).std()
        print("[OK] Rolling window features created")

    def run_pipeline(self):
        self.load_data()
        self.add_targets()
        self.add_temporal_features()
        self.add_lag_features()
        self.add_rolling_features()

        # Drop rows with NaN (introduced by shifts/rolling)
        before = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        after = len(self.df)
        print(f"[INFO] Dropped {before - after} rows due to NaNs from feature engineering")

        # Save
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.df.to_csv(self.output_file, index=False)
        print(f"[SUCCESS] Preprocessed data saved to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess rainfall dataset with feature engineering")
    parser.add_argument("--input_file", type=str, required=True, help="Path to cleaned CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save processed CSV")
    args = parser.parse_args()

    pre = Preprocessor(input_file=args.input_file, output_file=args.output_file)
    pre.run_pipeline()

