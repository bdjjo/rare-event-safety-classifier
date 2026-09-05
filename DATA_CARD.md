# Data card

**Source:** local, deterministic synthetic text generator in `src/rare_event/data.py`.
No private telemetry, downloaded corpus, external API, or foundation model is used.

**Unit:** one short assistant/session message. A positive label means the message
states that an unauthorized action or deliberate false report is being performed
or will be performed. Benign quotation, refusal, discussion, and fictional fixtures
are negative. This is a proxy for a safety-monitoring task, not verified agent action.

**Risk categories:** unauthorized data disclosure, bypass of controls, tampering
with an audit history, and deceptive reporting. Text describes actions abstractly;
there are no real targets, credentials, exploit instructions, or personal records.

**Counts:** 199 benign records per event. See manifest for exact split counts and
SHA-256 fingerprints. IDs and ticket markers are random/split bookkeeping and
are excluded from predictive text by preprocessing. Category, group, split and label
columns are used only for splitting/evaluation, never as model features.

**Shift:** event intent and labels remain fixed while event wording and benign
quotation framing change. Shift strengths are assigned using seed-stable uniforms,
so interventions are nested. The approximate 8% shifted-negative rate is stochastic;
event prevalence is exact. The `subgroup_metrics.csv` counts expose realized rates.

**Leakage limitations:** train/calibration/policy/test IDs and raw messages do not
overlap. After removing ticket markers, messages can repeat because the template
inventory is small. Shared in-domain templates are intentional and make ID easy.
This benchmark tests response to a constructed lexical change; it does not estimate
performance on unseen semantic intents. Treating random template splits as general
language understanding would be an error.

**JSONL loading:** `scripts/run_pipeline.py --data-dir PATH` expects independent
`train.jsonl`, `calibration.jsonl`, `policy.jsonl`, `id.jsonl`, and `combined.jsonl`.
Required fields: unique string `id`, nonempty string `text`, binary `label` with both
classes represented. Optional `category` and `group` enable slices. That mode uses
the first configured seed only, rejects cross-file ID/raw-text overlap, and must be
reported separately; synthetic data size/prevalence claims do not apply to it.
The paired synthetic path is the default; generated paired test files are not
intended as independent inputs to this loader mode.

**Reuse:** regenerate synthetic data with `scripts/generate_data.py`; inspect
`results/data_examples.jsonl` and `results/error_examples.json`. Never infer that
an abstract generated statement proves a real person or model committed misconduct.
