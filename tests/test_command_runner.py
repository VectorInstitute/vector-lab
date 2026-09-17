from vector_lab.errors import CommandError
from vector_lab.exec import CommandRunner
from tests.fakes.runner import FakeExecute


def test_dry_run_skips_execution() -> None:
    fake = FakeExecute()
    runner = CommandRunner(dry_run=True, execute=fake)
    result = runner.run(["ssh", "bonecho", "whoami"], category="ssh")
    assert result.skipped is True
    assert result.returncode == 0
    assert fake.calls == []
    assert runner.calls[0].args == ["ssh", "bonecho", "whoami"]


def test_injected_execute_used_when_not_dry_run() -> None:
    fake = FakeExecute()
    fake.add("whoami", stdout="alice\n")
    runner = CommandRunner(dry_run=False, execute=fake)
    result = runner.run(["ssh", "bonecho", "whoami"], category="ssh")
    assert result.stdout == "alice\n"
    assert result.skipped is False
    assert fake.calls


def test_check_raises_command_error() -> None:
    fake = FakeExecute()
    fake.add("false", returncode=2, stderr="boom")
    runner = CommandRunner(execute=fake)
    try:
        runner.run(["false"], category="local", suggestion="install foo")
        assert False, "expected CommandError"
    except CommandError as exc:
        assert exc.exit_status == 2
        assert exc.category == "local"
        assert "boom" in (exc.stderr or "")
        assert exc.suggestion == "install foo"


def test_verbose_prints_command(capsys) -> None:
    fake = FakeExecute()
    fake.add("echo", stdout="hi\n")
    runner = CommandRunner(verbose=True, execute=fake)
    runner.run(["echo", "hi"], category="local")
    out = capsys.readouterr().out
    assert "+ (local) echo hi" in out
    assert "hi" in out
