import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from src.demo_data import build_demo
from src.train import load_data, build_pipeline, evaluate
from src.config import FEATURES

def test_demo_schema():
    frame=build_demo(100)
    assert set(FEATURES + ["Churn"]).issubset(frame.columns)
    assert frame["Churn"].nunique()==2

def test_pipeline_predict_proba(tmp_path):
    path=tmp_path/"demo.csv"
    build_demo(200).to_csv(path,index=False)
    X,y=load_data(path)
    model=build_pipeline(LogisticRegression(max_iter=500,class_weight="balanced"))
    model.fit(X,y)
    probabilities=model.predict_proba(X.iloc[:5])[:,1]
    assert len(probabilities)==5
    assert all(0<=v<=1 for v in probabilities)

def test_missing_totalcharges(tmp_path):
    frame=build_demo(120)
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[0,"TotalCharges"]=" "
    path=tmp_path/"demo.csv"
    frame.to_csv(path,index=False)
    X,y=load_data(path)
    assert pd.isna(X.loc[0,"TotalCharges"])
    model=build_pipeline(LogisticRegression(max_iter=500))
    model.fit(X,y)
    assert len(model.predict(X))==len(X)

def test_no_target_leakage(tmp_path):
    path=tmp_path/"demo.csv"
    build_demo(100).to_csv(path,index=False)
    X,_=load_data(path)
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns

def test_bad_target(tmp_path):
    frame=build_demo(100)
    frame.loc[0,"Churn"]="maybe"
    path=tmp_path/"bad.csv"
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError):
        load_data(path)
