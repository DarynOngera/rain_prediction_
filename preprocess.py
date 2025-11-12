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
        return self.df

    def feature_engineering(self):
        """Generate targets + temporal + lags + rolling + seasonal + anomalies + persistence features"""
        df = self.df.copy()

        # -----------------------
        # 1. Target Variables (shift by dekads, not days)
        # -----------------------
        df["target_next_dekad"] = df["rfh"].shift(-1)    # 10 days ahead
        df["target_next_month"] = df["rfh"].shift(-3)    # ~1 month (3 dekads)
        df["target_next_season"] = df["rfh"].shift(-9)   # ~3 months
        df["target_next_year"] = df["rfh"].shift(-36)    # ~1 year

        # -----------------------
        # 2. Temporal Features
        # -----------------------
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Dekad of year (1–36)
        df["dekad"] = ((df["date"].dt.month - 1) * 3) + (df["date"].dt.day // 10) + 1
        df["dekad_sin"] = np.sin(2 * np.pi * df["dekad"] / 36)
        df["dekad_cos"] = np.cos(2 * np.pi * df["dekad"] / 36)

        # -----------------------
        # 3. Lag Features (dekadal horizons)
        # -----------------------
        lag_dekads = [1, 3, 6, 9, 18, 36]  # 1 dekad, 1m, 2m, season, half-year, year
        for col in ["rfh", "r1h", "r3h", "rfq", "r1q", "r3q"]:
            for lag in lag_dekads:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # -----------------------
        # 4. Rolling Window Features (in dekads)
        # -----------------------
        rolling_windows = [3, 9, 36]  # 1m, season, year
        for col in ["rfh", "r1h", "r3h"]:
            for window in rolling_windows:
                df[f"{col}_roll_mean_{window}"] = df[col].rolling(window, min_periods=1).mean()
                df[f"{col}_roll_std_{window}"] = df[col].rolling(window, min_periods=1).std()
                df[f"{col}_roll_min_{window}"] = df[col].rolling(window, min_periods=1).min()
                df[f"{col}_roll_max_{window}"] = df[col].rolling(window, min_periods=1).max()
                df[f"{col}_roll_skew_{window}"] = df[col].rolling(window, min_periods=1).skew()

        # -----------------------
        # 5. Deviation & Ratios from LTA
        # -----------------------
        df["rfh_dev"] = df["rfh"] - df["rfh_avg"]
        df["r1h_dev"] = df["r1h"] - df["r1h_avg"]
        df["r3h_dev"] = df["r3h"] - df["r3h_avg"]

        df["rfh_ratio"] = df["rfh"] / (df["rfh_avg"] + 1e-6)
        df["r1h_ratio"] = df["r1h"] / (df["r1h_avg"] + 1e-6)
        df["r3h_ratio"] = df["r3h"] / (df["r3h_avg"] + 1e-6)

        # -----------------------
        # 6. Change Rates
        # -----------------------
        for col in ["rfh", "r1h", "r3h", "rfq", "r1q", "r3q"]:
            df[f"{col}_diff"] = df[col].diff()

        # -----------------------
        # 7. Persistence Features
        # -----------------------
        df["dry_spell"] = (df["rfh"] < df["rfh_avg"]).astype(int)
        df["dry_spell_len"] = df["dry_spell"] * (
            df["dry_spell"].groupby((df["dry_spell"] != df["dry_spell"].shift()).cumsum()).cumcount() + 1
        )

        df["wet_spell"] = (df["rfh"] > df["rfh_avg"]).astype(int)
        df["wet_spell_len"] = df["wet_spell"] * (
            df["wet_spell"].groupby((df["wet_spell"] != df["wet_spell"].shift()).cumsum()).cumcount() + 1
        )

        # -----------------------
        # 8. Extreme Event Flags
        # -----------------------
        df["extreme_wet"] = (df["rfh"] > 1.5 * df["rfh_avg"]).astype(int)
        df["extreme_dry"] = (df["rfh"] < 0.5 * df["rfh_avg"]).astype(int)

        # -----------------------
        # 9. Normalized Anomalies (scale to [-1, 1] roughly)
        # -----------------------
        for col in ["rfq", "r1q", "r3q"]:
            df[f"{col}_norm"] = df[col] / 100.0

        # -----------------------
        # Final cleanup
        # -----------------------
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

