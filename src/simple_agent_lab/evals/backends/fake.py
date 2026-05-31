"""In-memory backend for tests and teaching — runs no real container.

Mirrors the LLM layer's ``fake`` adapter: it exercises the full
create → start → wait → logs → remove lifecycle so the framework, a `Suite`,
and a store can be tested end-to-end without Docker.

The "container body" is a caller-supplied callback (`on_start`) that receives
the created handle and may write outputs through whatever store the run uses,
simulating what the in-container runner would produce.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from ..protocols import ContainerPlan

OnStart = Callable[["FakeContainerHandle"], None]


class FakeContainerHandle:
    """Records lifecycle calls; runs `on_start` in place of a real process."""

    def __init__(
        self,
        *,
        name: str,
        env: Mapping[str, str],
        mounts: Mapping[str, Mapping[str, str]],
        command: tuple[str, ...],
        on_start: OnStart | None,
        status_code: int,
        log_text: str,
    ) -> None:
        self.name = name
        self.env = dict(env)
        self.mounts = {k: dict(v) for k, v in mounts.items()}
        self.command = command
        self.started = False
        self.removed = False
        self._on_start = on_start
        self._status_code = status_code
        self._log_text = log_text

    def start(self) -> None:
        self.started = True
        if self._on_start is not None:
            self._on_start(self)

    def wait(self) -> int:
        return self._status_code

    def logs(self) -> str:
        return self._log_text

    def remove(self) -> None:
        self.removed = True


class FakeBackend:
    """Create `FakeContainerHandle`s instead of real Docker containers."""

    def __init__(
        self,
        *,
        on_start: OnStart | None = None,
        status_code: int = 0,
        log_text: str = "",
    ) -> None:
        self._on_start = on_start
        self._status_code = status_code
        self._log_text = log_text
        self.created: list[FakeContainerHandle] = []

    def create(
        self,
        *,
        name: str,
        plan: ContainerPlan,
        command: tuple[str, ...],
        env: Mapping[str, str],
        mounts: Mapping[str, Mapping[str, str]],
        add_hosts: Mapping[str, str] | None = None,
    ) -> FakeContainerHandle:
        del plan, add_hosts
        handle = FakeContainerHandle(
            name=name,
            env=env,
            mounts=mounts,
            command=command,
            on_start=self._on_start,
            status_code=self._status_code,
            log_text=self._log_text,
        )
        self.created.append(handle)
        return handle
