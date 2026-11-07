"""Crée un lot de données ARTIFICIELLEMENT MODIFIÉES, pour vérifier Evidently."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import FEATURES

def generate(reference_path, output_path, seed=42):
    frame = pd.read_csv(reference_path)[FEATURES].copy()
    rng = np.random.default_rng(seed)
    frame = frame.sample(n=min(200,len(frame)),replace=False,random_state=seed).reset_index(drop=True)
    # Dérive volontaire, démonstration uniquement, ne représente PAS de vraies observations.
    frame["tenure"] = np.clip(pd.to_numeric(frame["tenure"],errors="coerce") - 18, 0, 120)
    frame["Contract"] = rng.choice(["Month-to-month","One year","Two year"],len(frame),p=[.90,.07,.03])
    path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False)
    return path

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--reference",default="monitoring/demo/reference.csv")
    p.add_argument("--output",default="monitoring/demo/current_SIMULE.csv")
    a=p.parse_args()
    print("JEU SIMULÉ, PAS UNE DÉRIVE RÉELLE :",generate(a.reference,a.output))
