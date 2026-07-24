"""OneMillion-Bench eval suite — host half.

`OneMillionSuite` (in ``suite.py``) maps a OneMillion-Bench rubric-graded Q&A
case onto the generic `Suite` framework; ``harness.py`` holds the shared
host-side helpers (dataset loading, sanitization, dotenv, generator/judge env).
The container half ships in the wheel at
``simple_agent_lab.evals.suites.onemillion``.
"""
