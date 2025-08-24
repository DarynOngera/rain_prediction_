# clean.py
import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "rainfall_clean.csv")

# Columns we actually care about
KEEP_COLS = [
    "date",
    "rfh", "r1h", "r3h",          # rainfall (10-day, 1-month, 3-month)
    "rfh_avg", "r1h_avg", "r3h_avg",  # long-term averages
    "rfq", "r1q", "r3q"           # anomalies
]

def clean_data():
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    datasets = []

    for file in csv_files:
        path = os.path.join(RAW_DIR, file)
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"[OK] Loaded {file} with {len(df)} rows")

            # Ensure datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

            # Keep only the relevant columns (if they exist in file)
            df = df[[c for c in KEEP_COLS if c in df.columns]]

            datasets.append(df)
        except Exception as e:
            print(f"[ERR] Could not load {file}: {e}")

    if not datasets:
        print("[ERR] No CSV files loaded, exiting.")
        return

    # Combine
    df = pd.concat(datasets, ignore_index=True).drop_duplicates()

    # Drop rows with invalid dates
    df = df.dropna(subset=["date"])

    # Drop rows where rainfall (rfh) is missing
    if "rfh" in df.columns:
        df = df.dropna(subset=["rfh"])

    # Fill other missing values with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(f"[INFO] Final cleaned shape: {df.shape}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_data()

