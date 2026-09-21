"""Download the Tetouan power-consumption dataset."""

from time_series_foundation_models.data import download_dataset

if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset written to: {path}")
