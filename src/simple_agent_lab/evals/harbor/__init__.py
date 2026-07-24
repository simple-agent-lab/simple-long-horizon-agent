"""Harbor eval integration helpers.

The package is intentionally import-light: Harbor itself is an optional
dependency and is imported only by the installed-agent adapter at Harbor runtime.
"""

AGENT_IMPORT_PATH = "simple_agent_lab.evals.harbor.agent:SimpleAgentLabHarborAgent"
DEFAULT_API_KIND = "openai-responses"
DEFAULT_MAX_TURNS = 150

__all__ = ["AGENT_IMPORT_PATH", "DEFAULT_API_KIND", "DEFAULT_MAX_TURNS"]
