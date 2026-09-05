# GitHub upload status and exact steps

The implementation, experiments and report are complete. The connected GitHub
account was verified as `bdjjo`. The intended repository
`bdjjo/rare-event-safety-classifier` returned 404, and no relevant repository was
listed among accessible repositories. No remote repository was changed.

The available connection can write to an existing accessible repository but does
not expose repository creation in this session. Create the empty repository with
your chosen visibility and make it accessible to the GitHub connection, or provide
the URL of the existing repository intended for this project. The upload request
is already authorized; this is a missing destination/capability issue.

## Manual upload to a new empty repository

Create `rare-event-safety-classifier` under your account in GitHub, then from the
extracted project folder run:

```bash
git init -b main
git add .
git commit -m "Implement calibrated rare-event safety benchmark and measured shift report"
git remote add origin https://github.com/bdjjo/rare-event-safety-classifier.git
git push -u origin main
```

If using an existing repository with its own files/history, clone it first and
copy this project into the intended subdirectory or branch. Do not force-push.

The default commit includes code, findings, scalar and budget results, error examples,
figures and logs. Predictions and model files are excluded by `.gitignore` but present
in the full bundle. Regenerate them using the documented command or, if desired,
explicitly include them with:

```bash
git add -f results/predictions results/models
git commit -m "Include reproducible prediction and model artifacts"
git push
```

Suggested description: Calibrated rare-event safety detection with fixed-review-budget
evaluation and a reproducible language-shift failure benchmark.

Suggested topics: `ai-safety`, `machine-learning`, `calibration`, `evaluation`,
`distribution-shift`, `model-monitoring`, `python`.
