import json
from pathlib import Path
import pandas as pd
from src.demo_data import build_demo
from src.train import train
from src.drift_demo import generate
from src.drift_report import _load_frame
from src.config import FEATURES

def test_drift_simulated_schema(tmp_path):
    ref=tmp_path/"reference.csv"
    build_demo(100).to_csv(ref,index=False)
    output=tmp_path/"current.csv"
    generate(ref,output)
    current=pd.read_csv(output)
    assert list(current.columns)==FEATURES
    assert len(current)==100
    assert current["tenure"].mean() < pd.read_csv(ref)["tenure"].mean()

def test_drift_drop_target_and_identifier(tmp_path):
    path=tmp_path/"sample.csv"
    build_demo(50).to_csv(path,index=False)
    assert list(_load_frame(path).columns)==FEATURES

def test_drift_missing_columns(tmp_path):
    path=tmp_path/"bad.csv"
    pd.DataFrame({"tenure":[2,4]}).to_csv(path,index=False)
    try: _load_frame(path)
    except ValueError as exc: assert "Colonnes manquantes" in str(exc)
    else: raise AssertionError("Expected ValueError")
