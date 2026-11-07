"""Monitoring minimal des scores : volumes et taux d'alerte, sans données personnelles."""
import json
from pathlib import Path
import numpy as np
from src.config import LOG_PATH

def summarize(path=LOG_PATH):
    path=Path(path)
    if not path.exists():
        return {"count":0,"message":"Aucune inférence enregistrée (ENABLE_INFERENCE_LOG=1)."}
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"count":0}
    probs=np.array([r["probability"] for r in rows])
    return {"count":len(rows),"mean_probability":round(float(probs.mean()),4),
            "high_risk_share":round(float(np.mean(probs>=.5)),4),
            "p95_probability":round(float(np.percentile(probs,95)),4),
            "note":"Un vrai suivi exige aussi données d'entrée agrégées, labels différés, dérive et qualité par segment."}

if __name__=="__main__":
    print(json.dumps(summarize(),indent=2,ensure_ascii=False))
