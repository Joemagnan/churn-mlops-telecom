"""Journalisation optionnelle dans MLflow; pas nécessaire à l'inférence."""
import json
import os
from pathlib import Path

def log_experiment(model, chosen, validation, test, metadata_path, model_path, demo):
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError("Installer les dépendances : pip install -r requirements-mlops.txt") from exc
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Churn_Telecom_Demo" if demo else "Churn_Telecom_IBM")
    info = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=f"{chosen}-{'DEMO' if demo else 'IBM'}") as run:
        mlflow.log_params({"algorithme":chosen, "donnees":"synthetiques" if demo else "IBM",
                            "threshold":info["threshold"], "training_rows":info["training_rows"],
                            "data_sha256":info["data_sha256"]})
        for name, scores in validation.items():
            for k, value in scores.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"validation_{name}_{k}", value)
        for k, value in test.items():
            if isinstance(value, (int,float)):
                mlflow.log_metric(f"test_{k}", value)
        mlflow.set_tags({"source":"demo" if demo else "IBM", "statut":"non_deployable" if demo else "evaluation"})
        mlflow.log_artifact(str(metadata_path), artifact_path="metadata")
        mlflow.sklearn.log_model(sk_model=model, name="churn_pipeline")
        return run.info.run_id
