from vector_lab.ssh.session import parse_ssh_g, resolved_ssh_from_g, wrap_login_shell


SSH_G_SAMPLE = """\
user alice
hostname bonecho.example.edu
port 22
identityfile /home/alice/.ssh/id_ed25519
identityfile /home/alice/.ssh/id_rsa
host bonecho.example.edu
"""


def test_parse_ssh_g_repeated_identityfile() -> None:
    parsed = parse_ssh_g(SSH_G_SAMPLE)
    assert parsed["user"] == "alice"
    assert parsed["hostname"] == "bonecho.example.edu"
    assert parsed["port"] == "22"
    assert parsed["identityfile"] == [
        "/home/alice/.ssh/id_ed25519",
        "/home/alice/.ssh/id_rsa",
    ]


def test_resolved_ssh_from_g() -> None:
    resolved = resolved_ssh_from_g("bonecho", SSH_G_SAMPLE)
    assert resolved.alias == "bonecho"
    assert resolved.hostname == "bonecho.example.edu"
    assert resolved.user == "alice"
    assert len(resolved.identity_files) == 2


def test_login_shell_quoting_simple() -> None:
    assert wrap_login_shell("squeue -u alice") == "bash -l -c 'squeue -u alice'"


def test_login_shell_quoting_nested_quotes() -> None:
    wrapped = wrap_login_shell("sbatch < job.sh")
    assert wrapped.startswith("bash -l -c ")
    assert "sbatch < job.sh" in wrapped
    inner = wrap_login_shell("printf %s \"$HOME\"")
    assert "bash -l -c " in inner
    assert "$HOME" in inner
