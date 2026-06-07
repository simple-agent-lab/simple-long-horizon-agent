# Create Makefile Build Pipeline for Python Project

## Task Description

You are given a workspace directory containing a Python project under `workspace/project/`. The project contains multiple Python source files but lacks a proper build pipeline.

Your task is to create a `Makefile` inside `workspace/project/` that defines the following targets:

- `lint`: Runs `flake8` on the project source files to check for style issues.
- `test`: Runs `pytest` to execute all tests in the project.
- `build`: Creates a source distribution and wheel distribution using `python setup.py sdist bdist_wheel`.
- `clean`: Removes all build artifacts, including `build/`, `dist/`, `*.egg-info` directories, and `__pycache__` folders.
- `all`: Runs `lint`, `test`, and `build` targets sequentially.

Additionally, create a minimal `setup.py` file in `workspace/project/` that allows building the project distributions.

Finally, verify that running `make all` inside `workspace/project/` completes successfully.

After running the pipeline, write a JSON report file named `pipeline_report.json` in the `workspace/` directory with the following structure:

```json
{
  "lint_passed": true,
  "tests_passed": true,
  "build_created": true
}
```

where each boolean indicates whether the respective step succeeded.

## Requirements

- The `Makefile` must be located at `workspace/project/Makefile`.
- The `setup.py` must be located at `workspace/project/setup.py`.
- Use `flake8` for linting.
- Use `pytest` for testing.
- Use `python setup.py sdist bdist_wheel` for building.
- The `clean` target must remove all build artifacts and caches.
- The `all` target must run `lint`, `test`, and `build` in order.
- The `pipeline_report.json` must be created in `workspace/` after running `make all`.

## Evaluation

Your solution will be tested by running `make all` inside `workspace/project/` and verifying:

- Linting passes without errors.
- All tests pass.
- Build artifacts (`dist/` directory with `.tar.gz` and `.whl` files) are created.
- The `pipeline_report.json` file exists and correctly reflects the pipeline results.

## Notes

- The project contains multiple Python files including at least one test file.
- You may assume `flake8`, `pytest`, and `setuptools` are installed in the environment.
- Your `Makefile` should use standard shell commands and be POSIX compliant.
- The `clean` target should be thorough but not delete source files.

Good luck!