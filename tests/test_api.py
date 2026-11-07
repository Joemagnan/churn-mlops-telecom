import json
from pathlib import Path
import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
import src.api as api
from src.api import app
from src.demo_data import build_demo
from src.train import load_data, build_pipeline

client=TestClient(app)
SAMPLE=json.loads((Path(__file__).resolve().parents[1]/"sample_request.json").read_text(encoding="utf-8"))

@pytest.fixture
def demo_model(tmp_path, monkeypatch):
    """Petit modèle synthétique, pour tester /predict sans le CSV IBM ni l'entraînement complet."""
    csv=tmp_path/"demo.csv"
    build_demo(200).to_csv(csv,index=False)
    X,y=load_data(csv)
    joblib.dump(build_pipeline(LogisticRegression(max_iter=500)).fit(X,y),tmp_path/"model.joblib")
    (tmp_path/"metadata.json").write_text(json.dumps({"model_name":"logistic_regression","version":"test",
        "data_source":"SYNTHETIQUE","trained_at":"2026-01-01T00:00:00+00:00","threshold":0.5}),encoding="utf-8")
    monkeypatch.setattr(api,"MODEL_PATH",tmp_path/"model.joblib")
    monkeypatch.setattr(api,"METADATA_PATH",tmp_path/"metadata.json")
    monkeypatch.setattr(api,"LOG_PATH",tmp_path/"monitoring"/"inferences.jsonl")
    monkeypatch.setattr(api,"_model",None)
    monkeypatch.setattr(api,"_metadata",None)
    return tmp_path

def test_health_returns_200():
    response=client.get("/health")
    assert response.status_code==200
    assert "model_available" in response.json()

def test_predict_rejects_incomplete_payload():
    response=client.post("/predict",json={"tenure":10})
    assert response.status_code==422

def test_openapi_available():
    assert client.get("/openapi.json").status_code==200

def test_predict_returns_probability(demo_model):
    response=client.post("/predict",json=SAMPLE)
    assert response.status_code==200
    body=response.json()
    assert 0<=body["churn_probability"]<=1
    assert body["prediction"] in {"risque_eleve","risque_faible"}
    assert body["model_version"]=="test"

def test_predict_accepts_missing_totalcharges(demo_model):
    payload={k:v for k,v in SAMPLE.items() if k!="TotalCharges"}
    assert client.post("/predict",json=payload).status_code==200

def test_predict_without_model_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api,"MODEL_PATH",tmp_path/"absent.joblib")
    monkeypatch.setattr(api,"_model",None)
    assert client.post("/predict",json=SAMPLE).status_code==503

def test_inference_log_has_no_personal_data(demo_model, monkeypatch):
    monkeypatch.setenv("ENABLE_INFERENCE_LOG","1")
    client.post("/predict",json=SAMPLE)
    rows=(demo_model/"monitoring"/"inferences.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows)==1
    assert set(json.loads(rows[0]))=={"timestamp","probability","label","model_version"}

def test_unwritable_log_does_not_break_predict(demo_model, monkeypatch):
    # Reproduit le conteneur dont le volume monitoring/ n'était pas accessible en écriture
    (demo_model/"bloque").write_text("ceci est un fichier, pas un dossier",encoding="utf-8")
    monkeypatch.setattr(api,"LOG_PATH",demo_model/"bloque"/"inferences.jsonl")
    monkeypatch.setenv("ENABLE_INFERENCE_LOG","1")
    assert client.post("/predict",json=SAMPLE).status_code==200
