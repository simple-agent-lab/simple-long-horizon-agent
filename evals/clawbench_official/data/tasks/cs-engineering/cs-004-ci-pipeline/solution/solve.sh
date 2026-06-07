#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${1:-workspace}"

PROJECT_DIR="$WORKSPACE/project"

# Create setup.py
cat > "$PROJECT_DIR/setup.py" << EOF
from setuptools import setup, find_packages

setup(
    name='sampleproject',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],
    author='Auto Generated',
    description='Sample Python project for CI pipeline',
)
EOF

# Create Makefile
cat > "$PROJECT_DIR/Makefile" << 'EOF'
.PHONY: lint test build clean all

lint:
	flake8 .

test:
	pytest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +

all: lint test build
EOF

# Run make all
(cd "$PROJECT_DIR" && make all)

# Check lint
if ! (cd "$PROJECT_DIR" && flake8 .); then
    LINT_PASSED=false
else
    LINT_PASSED=true
fi

# Check tests
if ! (cd "$PROJECT_DIR" && pytest -q --tb=short); then
    TESTS_PASSED=false
else
    TESTS_PASSED=true
fi

# Check build
if [ -d "$PROJECT_DIR/dist" ] && ls "$PROJECT_DIR/dist"/*.tar.gz "$PROJECT_DIR/dist"/*.whl >/dev/null 2>&1; then
    BUILD_CREATED=true
else
    BUILD_CREATED=false
fi

# Write pipeline_report.json
cat > "$WORKSPACE/pipeline_report.json" << EOF
{
  "lint_passed": $LINT_PASSED,
  "tests_passed": $TESTS_PASSED,
  "build_created": $BUILD_CREATED
}
EOF
