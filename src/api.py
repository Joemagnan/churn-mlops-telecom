"""Service REST FastAPI. Ne collecte pas les identifiants client dans les logs."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.config import MODEL_PATH, METADATA_PATH, LOG_PATH, FEATURES

app = FastAPI(title="API Churn Télécom", version="1.0.0",
              description="Démonstrateur pédagogique ; ne pas utiliser pour décisions automatiques réelles")
_model = None
_metadata = None
logger = logging.getLogger("uvicorn.error")

class Customer(BaseModel):
    gender: Literal["Female","Male"]
    SeniorCitizen: int = Field(ge=0,le=1)
    Partner: Literal["Yes","No"]
    Dependents: Literal["Yes","No"]
    tenure: int = Field(ge=0,le=120)
    PhoneService: Literal["Yes","No"]
    MultipleLines: Literal["Yes","No","No phone service"]
    InternetService: Literal["DSL","Fiber optic","No"]
    OnlineSecurity: Literal["Yes","No","No internet service"]
    OnlineBackup: Literal["Yes","No","No internet service"]
    DeviceProtection: Literal["Yes","No","No internet service"]
    TechSupport: Literal["Yes","No","No internet service"]
    StreamingTV: Literal["Yes","No","No internet service"]
    StreamingMovies: Literal["Yes","No","No internet service"]
    Contract: Literal["Month-to-month","One year","Two year"]
    PaperlessBilling: Literal["Yes","No"]
    PaymentMethod: Literal["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0,le=10000)
    TotalCharges: float | None = Field(default=None,ge=0)

def get_artifacts():
    global _model,_metadata
    if _model is None:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise HTTPException(status_code=503, detail="Modèle absent ; exécuter l'entraînement")
        # Charger UNIQUEMENT des artefacts entraînés de confiance (joblib/pickle).
        _model=joblib.load(MODEL_PATH)
        _metadata=json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return _model,_metadata

@app.get("/health")
def health():
    return {"status":"ok","model_available":MODEL_PATH.exists() and METADATA_PATH.exists()}

@app.get("/model-info")
def model_info():
    _,meta=get_artifacts()
    info={k:meta[k] for k in ["model_name","version","data_source","trained_at","threshold"]}
    info["sklearn_version"]=meta.get("sklearn_version")  # absent des métadonnées anciennes
    return info

@app.post("/predict")
def predict(customer:Customer):
    model,meta=get_artifacts()
    data=customer.model_dump()
    data["SeniorCitizen"]=str(data["SeniorCitizen"])
    frame=pd.DataFrame([data])[FEATURES]
    try:
        prob=float(model.predict_proba(frame)[0,1])
    except Exception as exc:
        raise HTTPException(status_code=500,detail="Erreur de prédiction") from exc
    label="risque_eleve" if prob>=float(meta["threshold"]) else "risque_faible"
    # Enregistrer uniquement des agrégats sans identifiant, sans attributs sensibles.
    if os.getenv("ENABLE_INFERENCE_LOG", "0")=="1":
        try:
            LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
            with LOG_PATH.open("a",encoding="utf-8") as f:
                f.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),
                                   "probability":round(prob,4),"label":label,
                                   "model_version":meta["version"]})+"\n")
        except OSError as exc:
            # Un journal inaccessible ne doit pas priver le client de sa réponse.
            logger.warning("Journal d'inférence non écrit : %s",exc)
    return {"churn_probability":round(prob,4),"prediction":label,
            "threshold":meta["threshold"],"model_version":meta["version"],
            "notice":"Score indicatif, pas une décision automatique"}
