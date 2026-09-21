"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "tetouan_power.csv"

UCI_DATASET_ID = 849
DATETIME_COLUMN = "datetime"
TARGET_COLUMN = "zone_1_power_consumption"

SEASONAL_PERIOD = 144  # 24 hours at 10-minute frequency
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
