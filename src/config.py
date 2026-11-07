from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
DEMO_PATH = ROOT / "data" / "telco_demo_SYNTHETIQUE.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
METRICS_PATH = ROOT / "reports" / "metrics.json"
LOG_PATH = ROOT / "monitoring" / "inferences.jsonl"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL
