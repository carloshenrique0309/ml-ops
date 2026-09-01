from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT_DIR / "data" / "churn.csv"

MODEL_PATH = ROOT_DIR / "modelo_final_v3_ok.pkl"