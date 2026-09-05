# Model card

**Release:** 1.0.0, sparse rare-event safety-classifier research baseline.

**Intended use:** reproduce and explain a failure of lexical classification under
distribution shift, evaluate probability calibration, and study capacity-limited
human review. Scores are synthetic-benchmark risk scores.

**Models:** word TF-IDF logistic regression (balanced and unweighted), plus a balanced
character TF-IDF logistic baseline. Each has a sigmoid fitted on independent,
representative-prevalence calibration data. Hyperparameters are fixed in source.
The primary model's balanced weights are approximately 100 for events and 0.5025
for negatives at 0.5% prevalence. This changes the training loss, not test prevalence.

**Inputs:** text only, with ticket identifiers removed. **Outputs:** raw decision
score and calibrated event probability. Monotonic calibration preserves ranking;
it cannot move an unseen event ahead of a high-scoring benign message.

**Evidence:** `docs/REPORT.md` and machine-readable results describe all three
baselines, all five seeds, separate shifts, severity sweeps and calibration ablations.
Trained model files for seed 17 are included under `results/models/` and in the
downloadable bundle; they can also be regenerated. Load only trusted joblib files.

**Limitations:** small authored templates, no real telemetry or independent human
annotation, English only, fixed event prior, no demographic coverage study, no
model-version/time holdout and no production scaling benchmark. Confidence outside
the synthetic ID distribution is unreliable. The frozen threshold is not a
capacity guarantee. Equal-width ECE and aggregate Brier can obscure rare positives.

**Not validated for:** production safety gating, automated sanctions, employee
monitoring, attribution of intent, or comparative claims about frontier models.
No claim of robustness, deployed monitoring, or safety certification is made.

**Operational next step:** obtain authorized deployment-like labels, split by time
and source, reserve adaptation and final-test sets separately, evaluate error costs
with reviewers, and measure shift-sensitive calibration before any deployment.
