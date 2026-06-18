"""Host-side SWE-bench registration for the self-evolving framework."""

from __future__ import annotations


def register_swebench_self_evolving_factories() -> None:
    """Register SWE-bench's default self-evolving component factories.

    Uses ``setdefault`` so local recipes and tests can pre-register custom
    factories under the same names.
    """

    from evals.swebench.suite import SwebenchSuite
    from simple_agent_lab.evals import LocalDirStore, LocalDockerBackend
    from simple_agent_lab.evals.suites.swebench import agent_package
    from simple_agent_lab.evolution import registry
    from simple_agent_lab.evolution.components.strategy import model_program_strategy
    from simple_agent_lab.evolution.surface import python_agent_surface

    registry.SUITES.setdefault("swebench", lambda **args: SwebenchSuite(**args))
    registry.SURFACES.setdefault(
        "python_agent_package",
        lambda *, artifact_key, version_root="agent/", **_args: python_agent_surface(
            default_files=agent_package.default_agent_package(),
            artifact_key=artifact_key,
            version_root=version_root,
        ),
    )
    registry.BACKENDS.setdefault(
        "local_docker",
        lambda **args: LocalDockerBackend(**args),
    )
    registry.STORES.setdefault("local_dir", lambda root, **_args: LocalDirStore(root))
    registry.STRATEGIES.setdefault("model_program", model_program_strategy)
