from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_RAW = PROJECT_DIR / "customer_churn_dataset-testing-master.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_CLEANED = OUTPUT_DIR / "cleaned_data.csv"  # <-- Add this line
TARGET = "Churn"
