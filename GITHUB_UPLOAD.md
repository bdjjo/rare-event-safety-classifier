# Repository and report publishing

The complete project is organized in the repository at
https://github.com/bdjjo/rare-event-safety-classifier.

## Browse the project

- `README.md`: project overview, measured headline results, and run instructions.
- `src/rare_event/`, `scripts/`, `configs/`, `tests/`: implementation and validation.
- `results/`: tables, figures, predictions, models, and experiment provenance.
- `docs/REPORT.md`: detailed findings rendered directly by GitHub.
- `docs/index.html`: standalone illustrated report for GitHub Pages.

## Publish the illustrated report

In repository Settings → Pages, select **Deploy from a branch**, branch **main**,
folder **/docs**, then Save. Wait for GitHub to confirm the deployment.
The site address will be https://bdjjo.github.io/rare-event-safety-classifier/.
This file does not indicate that Pages has already been enabled.

Add the deployed address to the repository About panel and pin the repository on
your profile. Suggested topics: `ai-safety`, `machine-learning`, `calibration`,
`distribution-shift`, `model-evaluation`, `python`.

## Reproduce locally

```bash
git clone https://github.com/bdjjo/rare-event-safety-classifier.git
cd rare-event-safety-classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_all.sh
```

On Windows, activate `.venv\Scripts\Activate.ps1` and use the individual Python
commands documented in the README.

## Release

Optionally create a `v1.0.0` release and attach the full downloadable bundle. Describe
it as a synthetic research benchmark. Preserve the distinction between measured
results, proposed extensions, and validation that has not been performed.
