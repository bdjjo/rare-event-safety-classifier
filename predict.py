#!/usr/bin/env python3
"""Score a message with a locally trained model (trusted model files only)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=Path("results/models/word_balanced.joblib"))
    args = parser.parse_args()
    model = joblib.load(args.model)
    print(json.dumps({"text": args.text, "synthetic_benchmark_risk_score": float(model.predict_proba([args.text])[0]),
                      "note": "Calibrated on synthetic in-domain data; not validated for real safety decisions."}, indent=2))
