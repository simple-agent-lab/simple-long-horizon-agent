#!/usr/bin/env bash
# Run the documentation routing and local-link lint.

set -e
source "$(dirname "$0")/_python.sh"

"${PYTHON[@]}" scripts/lint_docs.py
