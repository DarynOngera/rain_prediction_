import pandas as pd
import requests
import os 

DATA_DIR = 'raw'
PROCESSED_DIR = 'processed'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def download(url, path):
    r = requests.get(url)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            f.write(r.content)
    else:
        raise Exception(f"Failed to download {url}, status code: {r.status_code}")

def get_data(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f'Downloading {url}')
        download(url, path)
    else:
        print(f'{path} already exists')

    return pd.read_csv(path)

def main():
    url = input("Paste direct CSV link: ").strip()
    filename = input("Filename (with extension): ").strip()

    df = get_data(url, filename)
    print("\nDataset loaded successfully. Preview:")
    print(df.head())

if __name__ == "__main__":
    main()

