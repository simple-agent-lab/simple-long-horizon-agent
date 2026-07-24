#!/usr/bin/env bash
# Run the documentation routing and local-link lint.

set -e
cd "$(dirname "$0")/../.."
source runs/lib/_python.sh

"${PYTHON[@]}" -m scripts.lint_docs
