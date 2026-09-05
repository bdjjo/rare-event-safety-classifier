# Experiment protocol

This is a deliberately constructed **synthetic stress test**, not an independent
real-world safety benchmark. Its purpose is to expose lexical generalization failure.
The generator and baseline design were chosen to create that stress, before the full
five-seed experiment. One 4,000-row smoke run checked execution; no hyperparameter
search, seed selection, or metric fabrication was used. This is not a preregistered study.

## Questions and hypotheses

1. Can a sparse classifier rank explicit, familiar rare events well in-domain?
2. Does held-out sigmoid calibration improve probabilities for the balanced model?
3. Do new paraphrases reduce recall at a fixed human-review capacity?
4. Do benign quotations in unfamiliar contexts occupy high-risk ranking positions?
5. Does a character n-gram baseline or removal of class weights resolve the failure?

## Frozen protocol

Configuration: `configs/benchmark.json`. Seeds: 17, 29, 43, 71, 101.
Per seed: 40,000 training rows, 20,000 calibration rows, 20,000 policy rows,
and 40,000 paired test rows per condition. Event prevalence is exactly 0.5%:
200, 100, 100 and 200 events respectively. No resampling changes this prevalence.

Train TF-IDF and logistic regression on training data only. Fit a one-dimensional
sigmoid to calibration decision scores using the original 0.5% prevalence and no
class weights. Choose the calibrated 99th-percentile threshold on the separate
policy set. Freeze all fitted objects and thresholds before scoring the test sets.

Primary model: word unigram/bigram TF-IDF, logistic regression C=1, balanced
class weights. Ablations: same word model without weights; balanced character
3–5-gram TF-IDF. Report raw and calibrated probabilities for each model.
For sigmoid calibration, C=1000 is fixed rather than selected on the test data.

## Controlled interventions

| Test condition | Event wording | Benign context |
|---|---|---|
| `id` | Familiar explicit phrases | Ordinary tasks and 1% familiar refusals among negatives |
| `positive_paraphrase` | Held-out everyday paraphrases | Identical to ID |
| `benign_context` | Identical to ID | ~8% of negatives use novel quotation/discussion contexts |
| `combined` | All events paraphrased | Same benign shift as above |
| `combined_0.25/0.50/0.75` | Intervention probability = strength | Novel benign context probability = 0.08 × strength |

Within each seed all test variants share IDs, labels, latent category choices,
and nuisance fields. These are paired interventions, not independent datasets.
There is **no label-prior shift** and no change in the definition of an event.
No shifted example enters training, calibration, or policy selection.
The family of synthetic templates is intentionally shared in-domain; split
uniqueness does not imply template-level or semantic independence.

## Metrics and uncertainty

Primary endpoint: recall at exactly the top 1% of records (400 reviews per test
stream). Also report precision, lift above random review, TP/FP/FN and recall
at 0.1%, 0.5%, 2%, and 5%. Use floor(N × budget), with a minimum of one review.
Ties are resolved by a label-blind hash of the record ID; ranking uses unrounded
decision scores to avoid artificial ties from sigmoid saturation.

Report average precision (AP; non-interpolated PR summary), ROC-AUC, Brier score,
positive/negative conditional Brier, log loss, 15-bin equal-width ECE, mean
probability, and accuracy at 0.5. AP is not trapezoidal PR-AUC. The primary claims
use AP and review-budget recall, rather than overall accuracy or ROC-AUC alone.

Show exact-budget metrics separately from frozen-threshold alerts. The latter
can exceed the capacity target. Report event-category and error-type slices.

Aggregate each metric as mean ± sample standard deviation across seeds. For
ID-minus-combined recall, resample whole paired seed runs (2,000 bootstrap
replicates; RNG seed 2026). This small-run interval measures variability within
this generator, not uncertainty about real model deployments. Report per-seed
Wilson intervals as descriptive binomial approximations; top-k selection and
shared templates limit their independence assumptions. A degenerate bootstrap
interval is explicitly allowed and does not establish certainty.

## Follow-up experiments (not run in this release)

Use authorized, human-labeled deployment-like data and a frozen time/organization
holdout; include adjudicated negatives and ambiguous cases. Compare encoder-based
classifiers, gradient-boosted structured telemetry, and costed LLM judges. Add a
separate adaptation split of labeled paraphrases and novel benign contexts, then
recalibrate and evaluate on a new untouched shift set. Measure reviewer agreement,
latency, memory, subgroup disparities, and confidence-weighted escalation. Randomly
audit below-threshold traffic to estimate missed events. Test base-rate shifts
separately from lexical shifts. None of these extensions is claimed as completed.
