import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(encoding="utf-8-sig")

dataset_root = os.getenv("PELAK_DATASET_ROOT")

if not dataset_root:
    raise RuntimeError(
        "PELAK_DATASET_ROOT is not defined in the .env file."
    )

dataset_root = Path(dataset_root)

print("=" * 70)
print("Pelak-Khan Environment Check")
print("=" * 70)

print(f"Dataset root : {dataset_root}")
print(f"Exists       : {dataset_root.exists()}")
print(f"Is directory : {dataset_root.is_dir()}")

if not dataset_root.exists():
    raise FileNotFoundError(
        f"Dataset root does not exist: {dataset_root}"
    )

datasets = sorted(
    item.name
    for item in dataset_root.iterdir()
    if item.is_dir()
)

print()
print(f"Datasets found: {len(datasets)}")

for index, name in enumerate(datasets, start=1):
    print(f"{index:02d}. {name}")

print()
print("Environment check completed successfully.")

