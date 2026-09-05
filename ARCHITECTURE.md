# Architecture

```mermaid
flowchart TD
  G["Seeded generator or JSONL"] --> T["Training split"]
  G --> C["Calibration split"]
  G --> P["Policy split"]
  G --> E["Paired test conditions"]
  T --> M["TF-IDF + logistic model"]
  M --> S["Frozen scoring model"]
  C --> K["Sigmoid calibrator"]
  M --> K
  K --> S
  P --> H["Select threshold"]
  S --> H
  S --> V["Evaluate probabilities and ranks"]
  E --> V
  H --> V
  V --> R["Metrics, errors, figures, report"]
```

The test path has no connection back to fitting or threshold selection.
The exact-budget queue ranks records and takes exactly k. Frozen-threshold
alerting is evaluated separately because its queue size can change under shift.

The CPU implementation materializes each moderate-size split in memory and uses
sparse matrices. It is a reproducible research prototype, not a demonstrated
distributed pipeline. Production extensions would need streaming ingestion,
partitioned storage, model/data versioning, review feedback, and operational tests.
