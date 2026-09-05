#!/usr/bin/env python3
"""Build measured Markdown/standalone HTML findings and publication figures."""
import argparse
import base64
from datetime import datetime, timezone
import html
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rare_event.metrics import wilson

LABELS = {"id": "In-domain", "positive_paraphrase": "Event paraphrases", "benign_context": "Benign context shift", "combined": "Combined shift"}
COLORS = {"id": "#147d92", "positive_paraphrase": "#cc8b22", "benign_context": "#7654a1", "combined": "#cb4949"}


def md_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)


def mean_sd(series, percentage=False, digits=4):
    multiplier = 100 if percentage else 1
    suffix = "%" if percentage else ""
    if not percentage and 0 < abs(series.mean()) < 10**(-digits):
        return f"{series.mean():.3e} ± {series.std(ddof=1):.3e}"
    return f"{series.mean()*multiplier:.{digits}f} ± {series.std(ddof=1)*multiplier:.{digits}f}{suffix}"


def render_html(markdown, base_dir):
    """Small report-only Markdown renderer; no remote resources or scripts."""
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2">\1</a>', text)
        return text
    output, table_open, code_open, para = [], False, False, []
    def flush():
        if para:
            output.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
    for line in markdown.splitlines():
        if line.startswith("```"):
            flush()
            output.append("</code></pre>" if code_open else "<pre><code>")
            code_open = not code_open
            continue
        if code_open:
            output.append(html.escape(line)+"\n")
            continue
        if line.startswith("|"):
            flush()
            if re.match(r'^\|[\s|:\-]+\|$', line):
                continue
            first = not table_open
            if first:
                output.append('<div class="table-wrap"><table>')
                table_open = True
            tag = "th" if first else "td"
            output.append("<tr>"+"".join(f"<{tag}>{inline(x.strip())}</{tag}>" for x in line.strip("|").split("|"))+"</tr>")
            continue
        if table_open:
            output.append("</table></div>")
            table_open = False
        image_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if image_match:
            flush()
            path = (base_dir/image_match[2]).resolve()
            mime = "image/png" if path.suffix == ".png" else "image/svg+xml"
            output.append(f'<figure><img alt="{html.escape(image_match[1])}" src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"></figure>')
        elif line.startswith("#"):
            flush()
            level = len(line)-len(line.lstrip("#"))
            output.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif not line.strip():
            flush()
        else:
            para.append(line)
    flush()
    if table_open:
        output.append("</table></div>")
    css = """
    :root{color-scheme:light}body{margin:0;background:#edf2f5;color:#182b3c;font:16px/1.65 system-ui,sans-serif}
    main{max-width:1080px;margin:40px auto;background:white;padding:48px 56px;border-top:9px solid #147d92}
    h1{font-size:38px;line-height:1.16;margin:0 0 24px}h2{font-size:25px;border-bottom:2px solid #dfe9ed;padding-bottom:9px;margin-top:44px}
    h3{font-size:20px}a{color:#126b82}code{background:#f0f4f7;padding:2px 5px;border-radius:3px;font-size:.88em}
    pre{padding:20px;background:#142c3b;color:#e4f3f7;overflow:auto}pre code{background:none;padding:0;color:inherit}
    table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.45;margin:16px 0}th{background:#173b4e;color:white;text-align:left}
    th,td{padding:12px;border-bottom:1px solid #dce5eb}tr:nth-child(even){background:#f4f7f9}.table-wrap{overflow:auto}
    figure{margin:28px 0}img{max-width:100%;height:auto}strong{color:#103d51}
    @media(max-width:700px){main{padding:24px;margin:0}h1{font-size:29px}}
    @media print{body{background:white}main{margin:0;padding:0;border:0}h2{break-after:avoid}tr,img,figure{break-inside:avoid}a{color:inherit}}
    """
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rare-Event Safety Classifier — Experiment Report</title><style>'+css+'</style><main>'+"\n".join(output)+"</main></html>"


def build(out):
    manifest = json.loads((out/"manifest.json").read_text())
    if not manifest["synthetic"] or len(manifest["config"]["seeds"]) < 2:
        raise ValueError("This report template requires the full multi-seed synthetic benchmark")
    metrics = json.loads((out/"metrics.json").read_text())
    summary, budgets = pd.read_csv(out/"summary.csv"), pd.read_csv(out/"budget_metrics.csv")
    primary = summary.query("model == 'word_balanced' and variant == 'calibrated'")
    pb = budgets.query("model == 'word_balanced' and variant == 'calibrated'")
    b1 = pb[np.isclose(pb.budget, .01)]
    figures = out/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold", "axes.labelcolor": "#203443"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    conditions = list(LABELS)
    for ax, column, source, title in [(axes[0], "average_precision", primary, "Ranking: average precision"),
                                       (axes[1], "recall", b1, "Recall at a 1% review budget")]:
        means = [source[source.condition == c][column].mean() for c in conditions]
        stds = [source[source.condition == c][column].std() for c in conditions]
        bars = ax.bar(range(4), means, color=[COLORS[c] for c in conditions], yerr=stds, capsize=4)
        ax.set_xticks(range(4), ["In-domain", "Event\nparaphrases", "Benign\ncontext", "Combined"])
        ax.set_ylim(0, 1.15)
        ax.set_title(title)
        ax.set_ylabel("Score (mean ± seed SD)")
        for bar, v, sd in zip(bars, means, stds):
            ax.text(bar.get_x()+bar.get_width()/2, v+sd+.025, f"{v:.3f}", ha="center", fontsize=10)
    fig.suptitle("Rare-event safety monitoring: language shift breaks an excellent ID score", fontsize=14)
    fig.savefig(figures/"benchmark.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for c in conditions:
        grouped = pb[pb.condition == c].groupby("budget").recall.agg(["mean", "std"])
        axes[0].errorbar(grouped.index*100, grouped["mean"], yerr=grouped["std"], label=LABELS[c], color=COLORS[c], marker="o", capsize=3)
    axes[0].plot([.1, .5, 1, 2, 5], [.001, .005, .01, .02, .05], "--", color="#777777", label="Random review (expected)")
    axes[0].set(xlabel="Review budget (% of stream)", ylabel="Event recall", title="Capacity sweep", ylim=(-.03, 1.08))
    axes[0].legend(fontsize=8)
    strengths, means, stds = [], [], []
    for s, c in [(0, "id"), (.25, "combined_0.25"), (.5, "combined_0.50"), (.75, "combined_0.75"), (1, "combined")]:
        vals = b1[b1.condition == c].recall
        strengths.append(s); means.append(vals.mean()); stds.append(vals.std())
    axes[1].errorbar(strengths, means, yerr=stds, color=COLORS["combined"], marker="o", capsize=4)
    axes[1].set(xlabel="Combined intervention strength", ylabel="Recall at 1% budget", title="Gradual shift, frozen model", ylim=(-.03, 1.08))
    fig.savefig(figures/"budget_and_shift.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    seed = manifest["config"]["seeds"][0]
    for ax, c in zip(axes, ["id", "combined"]):
        for variant, color in [("raw", "#cc8b22"), ("calibrated", "#147d92")]:
            record = next(r for r in metrics if r["seed"] == seed and r["model"] == "word_balanced" and r["variant"] == variant and r["condition"] == c)
            rows = record["reliability"]
            ax.scatter([r["mean_score"] for r in rows], [r["event_rate"] for r in rows],
                       s=[max(18, np.log10(r["count"]+1)*35) for r in rows], color=color, label=variant, alpha=.8)
        ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
        ax.set(xlabel="Mean predicted event probability", ylabel="Observed event frequency", title=f"{LABELS[c]} — seed {seed}", xlim=(-.04, 1.04), ylim=(-.04, 1.04))
        ax.legend()
    fig.suptitle("Calibration on ID data does not establish calibration under shift", fontsize=14)
    fig.savefig(figures/"calibration.png", dpi=180)
    plt.close(fig)

    rows = []
    for c in conditions:
        s, b = primary[primary.condition == c], b1[b1.condition == c]
        rows.append([LABELS[c], mean_sd(s.average_precision), mean_sd(b.recall, True, 1),
                     mean_sd(b.precision, True, 1), f"{b.tp.mean():.1f} / {int(manifest['config']['n_test']*.005)}", mean_sd(s.brier, digits=6)])
    main_table = md_table(["Condition", "Average precision", "Recall @ 1%", "Precision @ 1%", "Events found / available", "Brier"], rows)
    rows = []
    for budget in sorted(pb.budget.unique()):
        row = [f"{budget*100:g}%", str(int(budget*manifest["config"]["n_test"]))]
        for c in conditions:
            row.append(mean_sd(pb[(pb.condition == c)&np.isclose(pb.budget, budget)].recall, True, 1))
        rows.append(row)
    capacity_table = md_table(["Budget", "Reviews", *LABELS.values()], rows)
    rows = []
    for kind in manifest["config"]["models"]:
        for c in ["id", "combined"]:
            s = summary[(summary.model == kind)&(summary.condition == c)&(summary.variant == "calibrated")]
            b = budgets[(budgets.model == kind)&(budgets.condition == c)&(budgets.variant == "calibrated")&np.isclose(budgets.budget, .01)]
            rows.append([kind, LABELS[c], mean_sd(s.average_precision), mean_sd(b.recall, True, 1), mean_sd(s.brier, digits=6)])
    ablation_table = md_table(["Model", "Condition", "AP", "Recall @ 1%", "Calibrated Brier"], rows)
    rows = []
    for c in ["id", "combined"]:
        for v in ["raw", "calibrated"]:
            s = summary[(summary.model == "word_balanced")&(summary.condition == c)&(summary.variant == v)]
            rows.append([LABELS[c], v, f"{s.brier.mean():.6g}", f"{s.positive_brier.mean():.6g}",
                         f"{s.negative_brier.mean():.6g}", f"{s.log_loss.mean():.6g}", f"{s.ece_15.mean():.6g}", f"{s.mean_probability.mean():.4%}"])
    calibration_table = md_table(["Condition", "Probabilities", "Brier", "Positive Brier", "Negative Brier", "Log loss", "ECE (15)", "Mean score"], rows)
    rows = []
    for c in conditions:
        s = primary[primary.condition == c]
        rows.append([LABELS[c], f"{s.frozen_threshold.mean():.7f}", f"{s.frozen_alerts.mean():.1f}",
                     f"{s.frozen_alert_rate.mean():.2%}", f"{s.frozen_recall.mean():.1%}", f"{s.frozen_precision.mean():.1%}"])
    threshold_table = md_table(["Condition", "Mean ID-selected threshold", "Alerts", "Alert rate", "Recall", "Precision"], rows)
    id_recall = b1[b1.condition == "id"].set_index("seed").recall
    shift_recall = b1[b1.condition == "combined"].set_index("seed").recall
    delta = (id_recall-shift_recall).to_numpy()
    rng = np.random.default_rng(2026)
    samples = rng.choice(delta, size=(manifest["config"]["bootstrap_replicates"], len(delta)), replace=True).mean(axis=1)
    ci = np.quantile(samples, [.025, .975])
    uncertainty = {"endpoint": "ID minus combined recall at 1% review", "seed_count": len(delta),
                   "mean_drop": float(delta.mean()), "seed_bootstrap95": ci.tolist(),
                   "replicates": manifest["config"]["bootstrap_replicates"], "bootstrap_seed": 2026,
                   "scope": "Within synthetic generator only; five seed clusters; no real-world inference"}
    (out/"uncertainty.json").write_text(json.dumps(uncertainty, indent=2)+"\n")
    rows = []
    for seed in manifest["config"]["seeds"]:
        a = b1[(b1.seed == seed)&(b1.condition == "id")].iloc[0]
        b = b1[(b1.seed == seed)&(b1.condition == "combined")].iloc[0]
        lo, hi = wilson(int(b.tp), int(b.tp+b.fn))
        rows.append([seed, f"{a.recall:.1%}", f"{b.recall:.1%}", f"{a.recall-b.recall:.1%}", f"[{lo:.1%}, {hi:.1%}]"])
    seed_table = md_table(["Seed", "ID recall", "Shift recall", "Recall drop", "Shift Wilson 95%"], rows)
    groups = pd.read_csv(out/"subgroup_metrics.csv")
    cats = groups[(groups.slice_type == "category")&(groups.slice != "benign")&groups.condition.isin(["id", "combined"])]
    rows = [[cat, f"{group[group.condition=='id'].recall.mean():.1%}", f"{group[group.condition=='combined'].recall.mean():.1%}"] for cat, group in cats.groupby("slice")]
    category_table = md_table(["Event category", "ID recall @ 1%", "Combined recall @ 1%"], rows)
    feature = json.loads((out/"features.json").read_text())["word_balanced"]["terms"]
    feature_text = ", ".join(f"`{x['term']}` ({x['weight']:.2f})" for x in feature["risk"][:8])
    error_data = json.loads((out/"error_examples.json").read_text())
    example_sections = []
    for error_type, label in [("false_negative", "Missed paraphrased event"), ("false_positive", "Benign message selected for review")]:
        match = next((e for e in error_data if e["condition"] == "combined" and e["error_type"] == error_type), None)
        if match:
            text = re.sub(r"\s*\[ticket=[^\]]*\]", "", match["text"])
            example_sections.append(f"### {label}\n\n{text}\n\nCalibrated score: **{match['probability']:.6f}**. Evaluation group: `{match['group']}`. Selected from seed {manifest['config']['seeds'][0]}; these examples illustrate errors, not their population frequency.")
    iid = primary[primary.condition == "id"]
    shifted = primary[primary.condition == "combined"]
    report = f"""# Rare-Event Safety Classifier

## Findings from a controlled language-shift benchmark

Run completed: {manifest['completed_utc']}. This report is generated from saved experiment outputs. All values below are measured, not illustrative placeholders.

**Main finding:** the calibrated word classifier achieves mean in-domain AP **{iid.average_precision.mean():.4f}** and **{id_recall.mean():.1%} recall** at a 1% review budget. Under the combined language shift, AP falls to **{shifted.average_precision.mean():.4f}** and recall to **{shift_recall.mean():.1%}**. The same 400-review capacity therefore changes from finding {b1[b1.condition=='id'].tp.mean():.1f} to {b1[b1.condition=='combined'].tp.mean():.1f} of 200 events per stream on average.

This is a deliberately authored synthetic demonstration of a classic failure mode. It is evidence that this lexical baseline fails on this specified intervention; it is not an estimate of deployed frontier-model safety performance. The simple shared templates make in-domain evaluation unusually easy.

![Benchmark overview](../results/figures/benchmark.png)

## 1. Data, labels and experimental isolation

Each of five runs uses 40,000 training records (200 events), 20,000 calibration records (100 events), 20,000 policy-selection records (100 events), and 40,000 test records (200 events) per condition. Every split has exactly 0.5% events: a 199:1 imbalance. Seeds are {', '.join(str(s) for s in manifest['config']['seeds'])}. There are 200,000 base test records across seeds, reused under paired interventions; condition totals are not independent sample counts.

An event is a stated unauthorized action involving disclosure, a control bypass, audit-history tampering, or deceptive reporting. Refusals, quotations, discussion and fictional fixtures are benign. Labels come from the generator, with no human adjudication. All text is English and abstract; no private telemetry or real-world harmful instructions are included.

The model fits only training text. A separate sigmoid calibrator fits only calibration scores and labels at the original event prior. A separate policy set chooses the 99th-percentile score threshold. All objects are frozen before any test condition is evaluated. Tickets and IDs are removed from model features; category and group metadata are evaluation-only.

Train, calibration, policy and test have disjoint IDs and raw messages. However, removing ticket markers can reveal repeated rendered templates across splits. This is intentional: the ID result measures familiar-template recognition, not semantic generalization to independent natural language. The checksum manifest verifies generated inputs; it does not eliminate this limitation.

## 2. Model, imbalance and calibration

The primary baseline is word unigram/bigram TF-IDF with logistic regression, C=1 and balanced class weights. The class weights increase the contribution of each rare positive to the loss without altering the evaluation prevalence. Ablations remove weighting or use character 3–5-grams. The calibrator is a one-dimensional logistic sigmoid with C=1000, fitted without balancing. These settings are fixed, with no test-driven tuning.

The implementation uses a positive-slope calibration check. Because the sigmoid is monotonic, it changes probability quality but preserves ranking. Exact-budget evaluation therefore uses the unrounded underlying decision scores for both raw and calibrated variants. This also avoids floating-point saturation creating artificial ranking ties.

There was one smoke run on smaller synthetic data to check execution, followed by the full fixed five-seed configuration. Three incomplete compressed prediction exports were found by the artifact audit; serialization was corrected, and the identical experiment configuration was rerun to regenerate the complete outputs reported here. An independent JSONL-loading smoke check also passed. No model or generator tuning was performed between the full runs. This is not a preregistered study. The generator was constructed to stress a lexical model, and all specified seeds, variants and conditions are retained.

## 3. Fixed-review-budget results

Values are mean ± sample standard deviation across five seeds. At a 1% budget, exactly 400 of 40,000 records are reviewed. Recall is TP divided by all 200 events; precision is TP divided by 400 reviews. Lift divides precision by the 0.5% event rate. Under perfect ranking, precision at this budget can be at most 50%, since the budget exceeds the number of events.

{main_table}

The positive-only intervention replaces event language with held-out everyday paraphrases while keeping benign records unchanged. The benign-only intervention retains explicit events but introduces unfamiliar quotation/discussion contexts in approximately 8% of negatives. The combined condition applies both. The event rate, label definition, record IDs, category assignments and nuisance choices stay fixed across paired conditions. Thus this comparison does not confound language shift with a changing class prior.

The contrast between the two isolated interventions identifies two mechanisms: loss of familiar lexical cues on positives, and competition from high-scoring benign text that contains those cues. Their combination exposes the full failure. This is a mechanistic interpretation supported by the controlled intervention and feature/error inspection, not a claim about model intent.

### Capacity sensitivity

{capacity_table}

At 0.1% capacity, even an oracle can review only 40 of 200 events, so the recall ceiling is 20%. Expected recall for random review equals the review fraction, with expected precision 0.5%. A constant 0.005 probability forecast has Brier score 0.004975. Predicting every record benign yields 99.5% accuracy while finding no events. These analytic baselines clarify why accuracy is a poor headline metric here.

![Review-budget and severity curves](../results/figures/budget_and_shift.png)

The severity sweep uses fixed uniform random draws, so higher intervention strengths extend the same perturbation population. It is a sensitivity analysis on this generator, not evidence that real distribution shift follows a linear progression.

## 4. Calibration findings

{calibration_table}

Sigmoid calibration reduces the distortion caused by balanced training in-domain. That improvement does not confer semantic robustness. Under the combined shift, the calibrated model can assign confident probabilities to benign quotation contexts while giving low scores to paraphrased events. Positive-conditional Brier makes missed rare events visible alongside aggregate Brier.

ECE uses 15 equal-width bins. It depends on binning and can hide rare-event failures when most examples fall near zero. The report therefore includes log loss, mean probability, conditional Brier, reliability plots and ranking metrics. Even a favorable overall calibration statistic alone would not justify a safety claim.

![Reliability plots](../results/figures/calibration.png)

The reliability plots show seed {manifest['config']['seeds'][0]}, not a pooled confidence band. Only populated bins are plotted; marker size increases with log bin count. The diagonal indicates agreement between predicted and observed frequencies. The full bin counts are in `results/metrics.json`.

## 5. Fixed capacity versus a frozen threshold

The threshold below is selected using only the in-domain policy split and then held fixed. Table values are means across seeds; each seed has its own fixed threshold.

{threshold_table}

Unlike a top-k review queue, a fixed score threshold does not guarantee a 1% workload. Shifted benign scores can greatly increase alerts. Keeping every threshold alert and imposing an exact review capacity are different operating policies; their recall values must not be interchanged. A top-k queue enforces capacity but can spend that capacity entirely on the wrong records.

## 6. Ablations and scope of the conclusion

{ablation_table}

Removing class weights and changing the feature representation test whether the finding is specific to a single training choice. These are lightweight baselines, not an exhaustive model comparison. Inspect their measured results directly; the benchmark does not establish that encoders, LLM judges or multimodal monitors would fail by the same amount. No such models were run.

Raw and calibrated AP/top-k recall are identical by construction of monotonic calibration; the saved values verify this. Any claim that post-hoc calibration alone repaired the review ranking would contradict this setup.

## 7. Replication, uncertainty and category slices

{seed_table}

The mean paired recall drop is **{delta.mean()*100:.1f} percentage points**. A whole-seed paired bootstrap with 2,000 replicates gives **[{ci[0]*100:.1f}, {ci[1]*100:.1f}] percentage points**. With only five seed clusters and a small template family, this interval describes synthetic-run variability only. If all runs have the same endpoint, the bootstrap interval collapses; that is not proof of zero uncertainty outside this benchmark.

Per-seed Wilson intervals are descriptive binomial approximations. Dependence from template reuse and top-k selection limits their inferential interpretation. Zero observed detections does not imply a precisely known population recall of zero. We do not pool repeated interventions as if they were independent samples or attach real-world significance to the seed bootstrap.

{category_table}

Category recalls are computed using the global review queue, not a separate budget per category. They cover four synthetic behavior categories, not demographic fairness or real incident subgroups. Exact denominators and benign-group review counts are in `results/subgroup_metrics.csv`.

## 8. Error analysis

Highest positive word-feature coefficients in seed {manifest['config']['seeds'][0]}: {feature_text}. These associations support the interpretation that the model learns lexical cues and familiar refusal framing.

{chr(10).join(example_sections)}

The error taxonomy is: **lexical omission** (paraphrased event lacks familiar cues), **context confusion** (benign discussion contains risk phrases), **review displacement** (false positives outrank true events), and **calibration transfer failure** (ID probability mapping becomes inappropriate after shift). Error examples are a deterministic illustrative selection; aggregate counts, rather than those examples alone, support prevalence claims.

## 9. Reproducibility and validation

The full run completed in {manifest['elapsed_seconds']:.1f} seconds in this environment, using Python {manifest['python']} and scikit-learn {manifest['packages']['scikit-learn']}. This is an observed wall time, not a portable speed or scale claim. CPU training, no GPU and no API key were required. Exact package versions are pinned in `requirements.txt`; environment details, source hashes, input hashes, parameters and timestamps are in `results/manifest.json`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_pipeline.py
python scripts/build_report.py
```

The unit suite checks paired labels/IDs, deterministic generation, split leakage detection, ticket removal, loader validation, exact-capacity arithmetic, label-blind tie handling, empty alert queues, invalid scores, Wilson boundaries, and calibration-only vocabulary exclusion. The separate results audit checks complete seed/condition coverage, capacity counts, AP/rank invariance after calibration, and recomputes key metrics from saved predictions. Test and audit logs are included with the results.

Saved artifacts include scalar CSVs, nested metrics JSON, all primary-model per-record predictions for five seeds and seven conditions, paired positive outcomes, subgroup counts, errors, features, trained seed-{manifest['config']['seeds'][0]} models, figures, and this report. Full synthetic text is reproducible from the generator; small text samples are committed. Intermediate model files and compressed predictions are included in the downloadable bundle and excluded from Git by default to keep the source repository lean.

## 10. Interpretation and next experiments

This project meets the intended demonstration: apparent in-domain success can coexist with severe failure when language changes, despite class-imbalance handling and probability calibration. Operationally, human-review capacity is only useful when ranking generalizes. Familiar-template scores should not be treated as evidence of reliable oversight on deployment traffic.

The next defensible step is evaluation on authorized, human-labeled deployment-like data with time/source holdouts. Introduce a separate adaptation set of paraphrased positives and benign discussion contexts, compare richer text encoders and costed LLM judges, recalibrate on a representative new-domain sample, and preserve a final untouched test set. Measure reviewer agreement and sample below-threshold records to estimate missed events. These experiments remain proposed; none is presented as completed.

The current implementation is an in-memory sparse CPU prototype. It does not demonstrate large-scale distributed ingestion, genuine agent telemetry, causal access to model intent, production latency guarantees, or deployed monitoring. The synthetic scenario contains clean labels and few templates, which are its most important limitations.

## Method references

[scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html) describes fitting probability mappings and inspecting reliability curves. This project uses a disjoint sigmoid fit.

[scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) defines the non-interpolated PR summary reported here. We label it AP rather than equating it with trapezoidal PR-AUC.

[scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) documents the scoring metrics used by the implementation. All empirical findings in this report come from this project's saved outputs, not those references.
"""
    docs = ROOT/"docs"
    (docs/"REPORT.md").write_text(report)
    (docs/"Rare_Event_Safety_Report.html").write_text(render_html(report, docs))
    report_manifest = {"builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "input_sha256": {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
                                        ["manifest.json", "metrics.json", "summary.csv", "budget_metrics.csv", "subgroup_metrics.csv", "features.json", "error_examples.json"]}}
    (out/"report_manifest.json").write_text(json.dumps(report_manifest, indent=2)+"\n")
    print(f"Report and 3 figures built from {len(metrics)} measured records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    build(args.results.resolve())
