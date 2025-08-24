# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split_dataset(input_file: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    # Load processed dataset
    df = pd.read_csv(input_file, parse_dates=["date"])
    df.sort_values("date", inplace=True)  # maintain chronology
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Split features and target(s)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols + ["date"]]

    X = df[feature_cols]
    y = df[target_cols]

    # Time-series aware split: use the first (1 - test_size)% as training
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"[SUCCESS] Train/Test split complete")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split processed rainfall data into train/test sets")
    parser.add_argument("--input", required=True, help="Path to processed CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save train/test CSVs")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for test set")
    args = parser.parse_args()

    split_dataset(args.input, args.output_dir, test_size=args.test_size)

