"""Behavior contracts for the Windows Job Object commit cap
(tools/environments/win_job_cap.py).

Real processes, real commit — no mocks, no Win32 queries. The scenario pinned
here is the 2026-09-06 incident class: an agent-spawned child COMMITTING
without bound must die INSIDE the job near the cap while the host keeps
committing happily.

Truth source: pure-Python commit accounting. The child commits fixed-size
blocks and reports how much it had committed when it died (``DIED_AT_MB``).
Under a working cap it dies near the cap; a broken/fail-open path lets it
reach system limits (gigabytes). This cannot false-green: a kernel job-limit
query reads 0 under job nesting (uv/pytest ancestors), and a SyntaxError
child's traceback echoing source text once matched a name-based assertion —
both failure modes are structurally impossible with a numeric bound check.
Allocations are commit-only (never touched) to stay load-independent.
Windows-only.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import textwrap

import pytest

pytestmark = pytest.mark.windows_only

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

BOMB = textwrap.dedent("""
    import sys
    blocks = []
    try:
        while True:
            blocks.append(bytearray(4 * 1024 * 1024))  # commit-only growth
    except MemoryError:
        print("DIED_AT_MB=" + str(len(blocks) * 4), flush=True)
        sys.exit(0)
""")


def _spawn_under_job(env_extra: dict, program: str = None):
    """Spawn a program through LocalEnvironment._run_bash; return (output,
    returncode). The program goes via a temp .py file — repr-through-``-c``
    mangles multi-line programs and the child dies in a SyntaxError whose
    traceback ECHOES the source (which once false-matched output assertions).
    """
    from tools.environments.local import LocalEnvironment

    program = program or BOMB
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    env = dict(os.environ)
    env.update(env_extra)
    try:
        local_env = LocalEnvironment(env=env)
        proc = local_env._run_bash(f'"{sys.executable}" "{path}"', timeout=180)
        try:
            out, _ = proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        return out or "", proc.returncode
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _fresh_module(monkeypatch, limit_gb: str):
    """Reset win_job_cap cached state and set the cap env var."""
    import tools.environments.win_job_cap as wjc

    monkeypatch.setattr(wjc, "_job_handle", None)
    monkeypatch.setattr(wjc, "_job_failed", False)
    monkeypatch.setenv("TERMINAL_JOB_COMMIT_LIMIT_GB", limit_gb)
    return wjc


def test_bomb_dies_inside_job_host_survives(monkeypatch):
    """A child committing without bound under a 64 MiB cap dies NEAR the cap
    (not at system limits), and the host then commits 64 MiB without trouble
    — the cap protected the host, not just the child."""
    _fresh_module(monkeypatch, "0.0625")

    out, rc = _spawn_under_job({})
    assert "DIED_AT_MB=" in out, f"bomb produced no accounting: {out!r} rc={rc}"
    died_at = int(out.split("DIED_AT_MB=")[1].split()[0].strip())
    assert died_at <= 256, (
        f"bomb committed {died_at} MB under a 64 MiB cap — job cap NOT "
        f"enforced on the _run_bash path: {out!r}")

    # Host must still be able to commit — the whole point of the cap.
    canary = bytearray(64 * 1024 * 1024)
    assert len(canary) == 64 * 1024 * 1024


def test_disabled_cap_is_noop(monkeypatch):
    """job_commit_limit_gb=0 must leave the module a no-op (fail-open)."""
    wjc = _fresh_module(monkeypatch, "0")

    assert wjc._get_job() is None
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wjc.assign_to_job(proc)   # must be a silent no-op
    finally:
        proc.kill()
        proc.wait()


def test_job_limits_total_commit_across_children(monkeypatch):
    """Two children in the SAME job share one commit budget: a holder
    occupying ~384 MiB of a 512 MiB job cap plus a committing bomb must push
    the bomb over the SHARED total — the cap is on the JOB, not per-process."""
    _fresh_module(monkeypatch, "0.5")

    holder = textwrap.dedent("""
        import time
        blocks = [bytearray(24 * 1024 * 1024) for _ in range(16)]  # ~384MB
        print("HOLDER_READY", flush=True)
        time.sleep(300)
    """)
    from tools.environments.local import LocalEnvironment

    local_env = LocalEnvironment(env=dict(os.environ))
    with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(holder)
        holder_path = f.name
    h = local_env._run_bash(f'"{sys.executable}" "{holder_path}"', timeout=10)
    try:
        out = ""
        for _ in range(300):        # wait for HOLDER_READY
            line = (h.stdout.readline() if h.stdout else "")
            out += line
            if "HOLDER_READY" in out:
                break
            import time as _t
            _t.sleep(0.1)
        assert "HOLDER_READY" in out, f"holder never became ready: {out!r}"

        bomb_out, bomb_rc = _spawn_under_job({})
        assert "DIED_AT_MB=" in bomb_out, (
            f"bomb produced no accounting: {bomb_out!r} rc={bomb_rc}")
        died_at = int(bomb_out.split("DIED_AT_MB=")[1].split()[0].strip())
        # 512 MiB job total − ~384 MiB held ⇒ bomb dies well under 384 MiB
        # of its own; per-process 512 would have let it pass 384.
        assert died_at <= 384, (
            f"bomb committed {died_at} MB while the job held ~384 of 512 MiB "
            f"— SHARED budget not enforced: {bomb_out!r}")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(holder_path)
        h.kill()
        h.wait()
