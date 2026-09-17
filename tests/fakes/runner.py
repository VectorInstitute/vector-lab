"""Scripted CommandRunner execute() for tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from vector_lab.exec import CommandResult

Matcher = Callable[[list[str]], bool]


class FakeExecute:
    def __init__(self) -> None:
        self._rules: list[tuple[Matcher, CommandResult | Callable[[list[str]], CommandResult]]] = []
        self.calls: list[list[str]] = []

    def add(
        self,
        match: str | Matcher,
        stdout: str = "",
        *,
        returncode: int = 0,
        stderr: str = "",
        category: str = "test",
    ) -> None:
        if isinstance(match, str):
            needle = match

            def _contains(args: list[str], n: str = needle) -> bool:
                joined = " ".join(args)
                return n in joined

            matcher: Matcher = _contains
        else:
            matcher = match

        result = CommandResult(
            args=[],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            category=category,
        )
        self._rules.append((matcher, result))

    def __call__(
        self,
        args: Sequence[str],
        *,
        category: str,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, env, timeout
        argv = list(args)
        self.calls.append(argv)
        for matcher, result in self._rules:
            if matcher(argv):
                if callable(result):
                    return result(argv)
                return CommandResult(
                    args=argv,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    category=category,
                )
        return CommandResult(args=argv, returncode=0, stdout="", stderr="", category=category)
