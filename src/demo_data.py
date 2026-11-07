"""Données SYNTHÉTIQUES pour test technique hors ligne, non destinées à l'évaluation scientifique."""
import numpy as np
import pandas as pd
from src.config import DEMO_PATH

def build_demo(n=500, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    charges = rng.uniform(20, 115, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55,.26,.19])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n)
    data = {
        "customerID": [f"DEMO-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0,1], n),
        "Partner": rng.choice(["Yes","No"], n),
        "Dependents": rng.choice(["Yes","No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes","No"], n),
        "MultipleLines": rng.choice(["Yes","No","No phone service"], n),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes","No","No internet service"], n),
        "OnlineBackup": rng.choice(["Yes","No","No internet service"], n),
        "DeviceProtection": rng.choice(["Yes","No","No internet service"], n),
        "TechSupport": rng.choice(["Yes","No","No internet service"], n),
        "StreamingTV": rng.choice(["Yes","No","No internet service"], n),
        "StreamingMovies": rng.choice(["Yes","No","No internet service"], n),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes","No"], n),
        "PaymentMethod": rng.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], n),
        "MonthlyCharges": charges,
        "TotalCharges": (charges * np.maximum(tenure,1) * rng.uniform(.8,1.2,n)).round(2),
    }
    p = 1/(1+np.exp(-(-.7 + 1.4*(contract=="Month-to-month") -.035*tenure + .009*(charges-70))))
    data["Churn"] = np.where(rng.random(n) < p, "Yes", "No")
    return pd.DataFrame(data)

def main():
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_demo().to_csv(DEMO_PATH,index=False)
    print("Fichier SYNTHÉTIQUE généré :", DEMO_PATH)

if __name__ == "__main__":
    main()
