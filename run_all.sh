#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python -m unittest discover -s tests -v
python scripts/run_pipeline.py
python scripts/build_report.py
python scripts/audit_results.py
