"""Agent-written JavaScript workflows over the existing Agent runtime.

Dynamic workflows are an outer harness: JavaScript owns orchestration, while
every model/tool worker is still a normal ``Agent.run(...)`` call.
"""

from __future__ import annotations

from .bridge import (
    AgentCallOptions,
    AgentCallResult,
    SimpleAgentCallRunner,
)
from .generator import (
    GeneratedWorkflow,
    extract_javascript,
    generate_workflow_script,
)
from .journal import WorkflowJournal
from .runtime import (
    DynamicWorkflowRuntime,
    WorkflowRunResult,
    WorkflowRuntimeOptions,
)

__all__ = [
    "AgentCallOptions",
    "AgentCallResult",
    "DynamicWorkflowRuntime",
    "GeneratedWorkflow",
    "SimpleAgentCallRunner",
    "WorkflowJournal",
    "WorkflowRunResult",
    "WorkflowRuntimeOptions",
    "extract_javascript",
    "generate_workflow_script",
]
