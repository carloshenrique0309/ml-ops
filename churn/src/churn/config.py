from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
RANDOM_STATE = 42
TARGET_COLUMN = "Churn"
