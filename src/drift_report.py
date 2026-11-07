"""Rapport Evidently sur les variables utilisées par le modèle, pas sur les labels."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.config import FEATURES, NUMERIC

def _load_frame(path):
    df = pd.read_csv(path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    df = df[FEATURES].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def make_report(reference_path, current_path, output_path):
    try:
        from evidently import Report, Dataset
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError("Installer : pip install -r requirements-mlops.txt") from exc
    ref = _load_frame(reference_path)
    current = _load_frame(current_path)
    if len(ref) < 30 or len(current) < 30:
        raise ValueError("Au moins 30 observations par groupe pour cette démonstration")
    report = Report([DataDriftPreset()])
    # Evidently 0.7+ : current d'abord, puis reference
    result = report.run(Dataset.from_pandas(current), Dataset.from_pandas(ref))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(out))
    summary = {"reference_rows":len(ref), "current_rows":len(current),
               "reference_source":str(reference_path), "current_source":str(current_path),
               "html_report":str(out), "type":"data_drift_non_performance"}
    out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",required=True)
    parser.add_argument("--current",required=True)
    parser.add_argument("--output",default="reports/data_drift.html")
    args=parser.parse_args()
    print(json.dumps(make_report(args.reference,args.current,args.output),indent=2,ensure_ascii=False))
