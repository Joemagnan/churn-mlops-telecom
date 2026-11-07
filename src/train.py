"""Entraînement reproductible sans fuite de données : prétraitements dans le Pipeline."""
import argparse
import hashlib
import json
import platform
import os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import (DATA_PATH, DEMO_PATH, MODEL_PATH, METADATA_PATH, METRICS_PATH,
                        FEATURES, NUMERIC, CATEGORICAL)

def load_data(path):
    df = pd.read_csv(path)
    needed = set(FEATURES + ["Churn"])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    df = df.drop_duplicates(subset=["customerID"]) if "customerID" in df.columns else df.drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c],errors="coerce")
    y = df["Churn"].map({"Yes":1,"No":0})
    if y.isna().any():
        raise ValueError("Valeurs Churn différentes de Yes/No")
    if y.nunique()!=2:
        raise ValueError("Le jeu de données doit contenir les deux classes")
    return df[FEATURES].copy(), y.astype(int)

def split_data(X, y):
    """Test 20 % isolé d'abord, puis validation = 25 % du reste (soit 60/20/20)."""
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
    Xa,Xval,ya,yval=train_test_split(Xtrain,ytrain,test_size=.25,stratify=ytrain,random_state=42)
    return Xtrain,Xtest,ytrain,ytest,Xa,Xval,ya,yval

def build_pipeline(estimator):
    num = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    cat = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    preprocess=ColumnTransformer([("numeric",num,NUMERIC),("categorical",cat,CATEGORICAL)])
    return Pipeline([("preprocess",preprocess),("model",estimator)])

def evaluate(model, X, y):
    score=model.predict_proba(X)[:,1]
    pred=(score>=.5).astype(int)
    return {"roc_auc":round(float(roc_auc_score(y,score)),4),
            "pr_auc":round(float(average_precision_score(y,score)),4),
            "accuracy":round(float(accuracy_score(y,pred)),4),
            "precision":round(float(precision_score(y,pred,zero_division=0)),4),
            "recall":round(float(recall_score(y,pred,zero_division=0)),4),
            "f1":round(float(f1_score(y,pred,zero_division=0)),4),
            "confusion_matrix":confusion_matrix(y,pred).tolist()}

def train(path=DATA_PATH, demo=False, tracking=False):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Données introuvables : {path}. Exécutez python -m src.download_data")
    X,y=load_data(path)
    Xtrain,Xtest,ytrain,ytest,Xa,Xval,ya,yval=split_data(X,y)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=150,min_samples_leaf=4,
                                                  class_weight="balanced",random_state=42,n_jobs=-1),
    }
    val_scores={}
    for name,estimator in candidates.items():
        pipeline=build_pipeline(estimator)
        pipeline.fit(Xa,ya)
        val_scores[name]=evaluate(pipeline,Xval,yval)
    chosen=max(val_scores,key=lambda name:val_scores[name]["roc_auc"])
    model=build_pipeline(candidates[chosen])
    model.fit(Xtrain,ytrain)
    test_scores=evaluate(model,Xtest,ytest)  # Une seule évaluation finale, après sélection
    # Les tests synthétiques ne doivent JAMAIS écraser les artefacts issus du fichier IBM.
    model_path = MODEL_PATH.parent / "demo" / MODEL_PATH.name if demo else MODEL_PATH
    metadata_path = METADATA_PATH.parent / "demo" / METADATA_PATH.name if demo else METADATA_PATH
    metrics_path = METRICS_PATH.parent / "demo" / METRICS_PATH.name if demo else METRICS_PATH
    reference_path = MODEL_PATH.parent.parent / "monitoring" / ("demo" if demo else "ibm") / "reference.csv"
    for dest in (model_path, metadata_path, metrics_path, reference_path):
        dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    # Pas d'identifiant client ni de cible Churn dans la référence de surveillance.
    Xtrain.to_csv(reference_path, index=False)

    baseline={col:{"mean":float(Xtrain[col].mean()),
                   "std":float(Xtrain[col].std()) if pd.notna(Xtrain[col].std()) else 0.0} for col in NUMERIC}
    meta={"model_name":chosen,"version":"1.0.0-demo" if demo else "1.0.0",
          "trained_at":datetime.now(timezone.utc).isoformat(),
          "data_source":"SYNTHETIQUE - NE PAS DEPLOYER" if demo else "IBM Telco Customer Churn",
          "training_rows":len(Xtrain),"test_rows":len(Xtest),"threshold":0.5,
          # Traçabilité : quel fichier exact et quelles versions ont produit ce modèle
          "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
          "sklearn_version":sklearn.__version__,"python_version":platform.python_version(),
          "features":FEATURES,"numeric_reference":baseline}
    metadata_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    results={"selected_model":chosen,"validation":val_scores,"test":test_scores,
             "warning":"METRIQUES DE DEMONSTRATION SYNTHETIQUE" if demo else "Evaluation sur jeu IBM tenu à l'écart"}
    metrics_path.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    if tracking:
        from src.mlflow_tracking import log_experiment
        run_id = log_experiment(model, chosen, val_scores, test_scores,
                                metadata_path, model_path, demo)
        results["mlflow_run_id"] = run_id
        metrics_path.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False))
    return results

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--demo",action="store_true",help="Données synthétiques, validation technique uniquement")
    parser.add_argument("--mlflow", action="store_true", help="Journaliser l'expérience dans MLflow")
    args=parser.parse_args()
    train(DEMO_PATH if args.demo else DATA_PATH,demo=args.demo,tracking=args.mlflow)
