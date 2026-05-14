"""Preset agents that ship with simple-agent-lab.

Each preset is a small, self-contained agent built on top of the
core runtime (`simple_agent_lab.core`), the message protocol
(`simple_agent_lab.messages`), the LLM access layer
(`simple_agent_lab.llm`), and the tool ABI (`simple_agent_lab.tools`).
Presets are not auto-imported by the top-level `simple_agent_lab`
namespace — explicit `from simple_agent_lab.agents.<name> import ...`
keeps the core package's public surface focused on the protocol and
runtime.
"""
