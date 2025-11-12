import subprocess

subprocess.run([
    "python", "train.py",
    "--file", "data/processed/newFeatures.csv",
    "--target", "rfh"
])

