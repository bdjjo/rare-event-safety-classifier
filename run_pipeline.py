#!/usr/bin/env python3
"""Run frozen synthetic benchmark or explicitly separated JSONL splits."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from rare_event.data import generate, load_jsonl, assert_disjoint, fingerprint
from rare_event.model import SafetyClassifier
from rare_event.metrics import evaluate, rank_indices, choose_threshold


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(config, out, data_dir=None):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "platform": platform.platform(), "config": config, "config_sha256": config_hash,
                "synthetic": data_dir is None, "data": {}, "source_sha256": {},
                "packages": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]}}
    for path in sorted(ROOT.rglob("*.py")):
        manifest["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(out / "manifest.json", manifest)
    results, budget_rows, group_rows, errors, features, paired_rows = [], [], [], [], {}, []
    first_seed = config["seeds"][0]
    for seed in config["seeds"]:
        print(f"Seed {seed}: generating/loading separated splits", flush=True)
        if data_dir:
            frames = {name: load_jsonl(data_dir / f"{name}.jsonl") for name in
                      ["train", "calibration", "policy", "id", "combined"]}
            train, cal, policy = (frames[x] for x in ["train", "calibration", "policy"])
            tests = {name: frames[name] for name in ["id", "combined"]}
            assert_disjoint(list(frames.values()))
        else:
            train, cal, policy = [generate(config[f"n_{name}"], config["prevalence"], seed+offset, name)
                                  for name, offset in [("train", 1000), ("calibration", 2000), ("policy", 3000)]]
            interventions = {"id": (0, 0), "positive_paraphrase": (1, 0),
                             "benign_context": (0, 1), "combined": (1, 1)}
            interventions.update({f"combined_{s:.2f}": (s, s) for s in config["shift_strengths"]})
            tests = {name: generate(config["n_test"], config["prevalence"], seed+4000, "test", a, b)
                     for name, (a, b) in interventions.items()}
            assert_disjoint([train, cal, policy, tests["id"]])
            for frame in tests.values():
                assert np.array_equal(frame.label, tests["id"].label)
                assert np.array_equal(frame.id, tests["id"].id)
            if seed == first_seed:
                examples = pd.concat([f.assign(condition=name).groupby("group", group_keys=False).head(3)
                                      for name, f in tests.items() if name in ["id", "combined"]])
                examples.to_json(out / "data_examples.jsonl", orient="records", lines=True)
        manifest["data"][str(seed)] = {name: {"rows": len(frame), "positives": int(frame.label.sum()),
            "sha256": fingerprint(frame)} for name, frame in
            {"train": train, "calibration": cal, "policy": policy, **tests}.items()}
        for kind in config["models"]:
            print(f"Seed {seed}: fitting {kind}", flush=True)
            model = SafetyClassifier(kind, seed).fit(train.text, train.label, cal.text, cal.label)
            policy_scores = model.decision_function(policy.text)
            thresholds = {variant: choose_threshold(model.predict_from_scores(policy_scores, variant == "calibrated"),
                          config["policy_budget"]) for variant in ["raw", "calibrated"]}
            if seed == first_seed:
                model_dir = out / "models"
                model_dir.mkdir(exist_ok=True)
                joblib.dump(model, model_dir / f"{kind}.joblib", compress=3)
                features[kind] = {"terms": model.top_features(), "calibration_slope": float(model.calibrator.coef_[0, 0]),
                                  "calibration_intercept": float(model.calibrator.intercept_[0]),
                                  "thresholds": thresholds}
            primary_predictions = {}
            for condition, frame in tests.items():
                if condition.startswith("combined_") and kind != "word_balanced":
                    continue
                y = frame.label.to_numpy()
                scores = model.decision_function(frame.text)
                for variant in ["raw", "calibrated"]:
                    p = model.predict_from_scores(scores, variant == "calibrated")
                    metrics = evaluate(y, p, frame.id.to_numpy(), config["budgets"], thresholds[variant], scores)
                    record = {"seed": seed, "model": kind, "variant": variant, "condition": condition, **metrics}
                    results.append(record)
                    for b in metrics["budgets"]:
                        budget_rows.append({"seed": seed, "model": kind, "variant": variant, "condition": condition,
                                            **{k: v for k, v in b.items() if not isinstance(v, list)}})
                    if kind == "word_balanced" and variant == "calibrated":
                        selected = np.zeros(len(y), dtype=bool)
                        selected[rank_indices(scores, frame.id)[:max(1, int(len(y)*config["policy_budget"]))]] = True
                        pred = frame.copy()
                        pred["score"] = scores
                        pred["probability"] = p
                        pred["reviewed"] = selected.astype(int)
                        primary_predictions[condition] = pred
                        prediction_dir = out / "predictions"
                        prediction_dir.mkdir(exist_ok=True)
                        # Retain texts only in examples/errors; regenerate full synthetic text from hashes/config.
                        # Serialize completely before writing to avoid incomplete streamed gzip artifacts.
                        csv_bytes = pred.drop(columns=["text"]).to_csv(index=False, float_format="%.12g").encode()
                        compressed = gzip.compress(csv_bytes, mtime=0)
                        target = prediction_dir / f"seed_{seed}_{condition}.csv.gz"
                        target.write_bytes(compressed)
                        assert gzip.decompress(target.read_bytes()) == csv_bytes
                        for column in ["category", "group"]:
                            for name, g in pred.groupby(column):
                                pos = int(g.label.sum())
                                group_rows.append({"seed": seed, "condition": condition, "slice_type": column,
                                    "slice": name, "n": len(g), "positives": pos,
                                    "reviewed": int(g.reviewed.sum()), "tp": int((g.reviewed*g.label).sum()),
                                    "recall": float((g.reviewed*g.label).sum()/pos) if pos else None})
                        if seed == first_seed and condition in ["id", "combined"]:
                            for error_type, mask in [("false_negative", (y == 1) & ~selected),
                                                     ("false_positive", (y == 0) & selected)]:
                                chosen = pred.loc[mask].sort_values("probability", ascending=error_type == "false_negative").head(12)
                                errors.extend(chosen.assign(condition=condition, error_type=error_type).to_dict("records"))
            if kind == "word_balanced" and not data_dir:
                a, b = primary_predictions["id"], primary_predictions["combined"]
                pos = a.label == 1
                for idx in a.index[pos]:
                    paired_rows.append({"seed": seed, "id": a.loc[idx, "id"],
                                        "id_reviewed": int(a.loc[idx, "reviewed"]),
                                        "shift_reviewed": int(b.loc[idx, "reviewed"])})
            print(f"Seed {seed}: completed {kind}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        write_json(out / "metrics.json", results)
    write_json(out / "metrics.json", results)
    write_json(out / "features.json", features)
    write_json(out / "error_examples.json", errors)
    pd.DataFrame(budget_rows).to_csv(out / "budget_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "subgroup_metrics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "paired_positive_outcomes.csv", index=False)
    scalar_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not isinstance(v, (list, dict))}
        row.update({f"frozen_{k}": v for k, v in r["frozen_threshold"].items()})
        scalar_rows.append(row)
    pd.DataFrame(scalar_rows).to_csv(out / "summary.csv", index=False)
    manifest["elapsed_seconds"] = time.perf_counter()-started
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "manifest.json", manifest)
    print(f"Finished. Results: {out}; elapsed {manifest['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, help="Folder containing train/calibration/policy/id/combined.jsonl")
    parser.add_argument("--smoke", action="store_true", help="Small integration run; not the published benchmark")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(seeds=[17], n_train=4000, n_calibration=4000, n_policy=4000, n_test=4000,
                      models=["word_balanced"], shift_strengths=[])
    if args.data_dir:
        config["seeds"] = [config["seeds"][0]]
        config["shift_strengths"] = []
    run(config, args.output.resolve(), args.data_dir)
