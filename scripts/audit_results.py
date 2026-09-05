#!/usr/bin/env python3
"""Recompute published endpoints from saved predictions and check provenance."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from rare_event.data import generate, fingerprint


def main():
    out = ROOT/"results"
    manifest = json.loads((out/"manifest.json").read_text())
    config = manifest["config"]
    summary = pd.read_csv(out/"summary.csv")
    budget = pd.read_csv(out/"budget_metrics.csv")
    expected = len(config["seeds"])*(len(config["models"])*4+len(config["shift_strengths"]))*2
    assert len(summary) == expected, (len(summary), expected)
    assert not summary.duplicated(["seed", "model", "variant", "condition"]).any()
    assert len(budget) == expected*len(config["budgets"])
    assert (budget.tp+budget.fp == budget.k).all()
    assert (budget.tp+budget.fn == round(config["n_test"]*config["prevalence"])).all()
    assert (budget.k == np.maximum(1, np.floor(budget.budget*config["n_test"]))).all()
    keys = ["seed", "model", "condition"]
    raw = summary[summary.variant == "raw"].set_index(keys).sort_index()
    cal = summary[summary.variant == "calibrated"].set_index(keys).sort_index()
    np.testing.assert_allclose(raw.average_precision, cal.average_precision)
    keys_b = keys+["budget"]
    a = budget[budget.variant == "raw"].set_index(keys_b).sort_index()
    b = budget[budget.variant == "calibrated"].set_index(keys_b).sort_index()
    np.testing.assert_array_equal(a.tp, b.tp)
    audited = 0
    for seed in config["seeds"]:
        base = pd.read_csv(out/f"predictions/seed_{seed}_id.csv.gz")
        for path in sorted((out/"predictions").glob(f"seed_{seed}_*.csv.gz")):
            condition = path.name.removeprefix(f"seed_{seed}_").removesuffix(".csv.gz")
            p = pd.read_csv(path)
            np.testing.assert_array_equal(base.id, p.id)
            np.testing.assert_array_equal(base.label, p.label)
            assert p.reviewed.sum() == int(len(p)*config["policy_budget"])
            row = cal.loc[(seed, "word_balanced", condition)]
            np.testing.assert_allclose(average_precision_score(p.label, p.score), row.average_precision, atol=1e-10)
            np.testing.assert_allclose(brier_score_loss(p.label, p.probability), row.brier, atol=1e-10)
            br = b.loc[(seed, "word_balanced", condition, config["policy_budget"])]
            assert int((p.label*p.reviewed).sum()) == br.tp
            audited += 1
        regenerated = generate(config["n_test"], config["prevalence"], seed+4000, "test")
        assert fingerprint(regenerated) == manifest["data"][str(seed)]["id"]["sha256"]
    for rel, checksum in manifest["source_sha256"].items():
        if rel.startswith("src/") or rel == "scripts/run_pipeline.py":
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == checksum, f"Training source changed: {rel}"
    report_manifest_path = out/"report_manifest.json"
    if report_manifest_path.exists():
        report_manifest = json.loads(report_manifest_path.read_text())
        assert hashlib.sha256((ROOT/"scripts/build_report.py").read_bytes()).hexdigest() == report_manifest["builder_sha256"]
        for name, checksum in report_manifest["input_sha256"].items():
            assert hashlib.sha256((out/name).read_bytes()).hexdigest() == checksum, f"Report input changed: {name}"
    print(f"PASS: {len(summary)} metric records; {len(budget)} budget rows; {audited} prediction files.")
    print("PASS: paired IDs/labels, exact capacities, raw/calibrated rank invariance, independently recomputed AP/Brier/TP.")
    print("PASS: five regenerated test fingerprints and all core training-source checksums (src/ and run_pipeline.py).")
    if report_manifest_path.exists():
        print("PASS: report builder and all report-input hashes; reporting provenance is separate from training.")


if __name__ == "__main__":
    main()
