import json
from src.monitor import summarize

def test_empty_monitor(tmp_path):
    assert summarize(tmp_path/"nothing.jsonl")["count"] == 0

def test_monitor_summary(tmp_path):
    path=tmp_path/"inferences.jsonl"
    path.write_text("\n".join(json.dumps({"probability":p}) for p in [.2,.7,.8]),encoding="utf-8")
    result=summarize(path)
    assert result["count"]==3
    assert abs(result["high_risk_share"]-2/3)<.0001
