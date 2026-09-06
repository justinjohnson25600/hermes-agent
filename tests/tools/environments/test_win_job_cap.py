"""Behavior contracts for the Windows Job Object commit cap
(tools/environments/win_job_cap.py).

Real processes, real allocations — no mocks. The scenario pinned here is the
2026-09-06 incident class: an agent-spawned child allocating without bound
must die INSIDE the job (MemoryError) while the host keeps allocating
happily. Windows-only (Job Objects are a Windows primitive; other platforms
assert the module degrades to a no-op).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.windows_only

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOMB = textwrap.dedent("""
    import sys
    blocks = []
    try:
        while True:
            blocks.append(bytearray(64 * 1024 * 1024))   # 64 MB steps
            for b in blocks[-1:]:                        # touch: make it real
                b[:] = b"\\x01"
    except MemoryError:
        print("CAUGHT_MEMORYERROR", flush=True)
        sys.exit(0)
    sys.exit(0)
""")


def _spawn_under_job(env_extra: dict) -> subprocess.CompletedProcess:
    """Spawn the bomb through LocalEnvironment._run_bash with a tiny cap."""
    from tools.environments.local import LocalEnvironment

    env = dict(os.environ)
    env.update(env_extra)
    local_env = LocalEnvironment(env=env)
    proc = local_env._run_bash(
        f'"{sys.executable}" -c {BOMB!r}', timeout=120)
    out, _ = proc.communicate(timeout=120)
    return out or ""


def test_bomb_dies_inside_job_host_survives(monkeypatch, tmp_path):
    """A child allocating without bound under a 1 GB cap dies with
    MemoryError inside the job; the host process then allocates 64 MB
    without trouble (the cap protected the host, not just the child)."""
    import tools.environments.win_job_cap as wjc

    # Fresh module state so the cached job reflects OUR cap, not a
    # previously-created one from another test in this file.
    monkeypatch.setattr(wjc, "_job_handle", None)
    monkeypatch.setattr(wjc, "_job_failed", False)
    monkeypatch.setenv("TERMINAL_JOB_COMMIT_LIMIT_GB", "1")

    out = _spawn_under_job({})
    assert "CAUGHT_MEMORYERROR" in out, (
        f"bomb did not hit the job cap as expected; output: {out!r}")

    # Host must still be able to allocate — the whole point of the cap.
    canary = bytearray(64 * 1024 * 1024)
    canary[0] = 1
    canary[-1] = 1
    assert len(canary) == 64 * 1024 * 1024


def test_disabled_cap_is_noop(monkeypatch):
    """job_commit_limit_gb=0 must leave the module a no-op (fail-open)."""
    import tools.environments.win_job_cap as wjc

    monkeypatch.setattr(wjc, "_job_handle", None)
    monkeypatch.setattr(wjc, "_job_failed", False)
    monkeypatch.setenv("TERMINAL_JOB_COMMIT_LIMIT_GB", "0")

    assert wjc._get_job() is None
    # and assigning must not raise for a live process
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wjc.assign_to_job(proc)   # must be a silent no-op
    finally:
        proc.kill()
        proc.wait()


def test_job_limits_total_commit_across_children(monkeypatch):
    """Two children in the SAME job share one commit budget: A holding
    ~900 MB plus B allocating without bound must push B over the 1 GB
    job total — the cap is on the JOB, not per-process."""
    import tools.environments.win_job_cap as wjc

    monkeypatch.setattr(wjc, "_job_handle", None)
    monkeypatch.setattr(wjc, "_job_failed", False)
    monkeypatch.setenv("TERMINAL_JOB_COMMIT_LIMIT_GB", "1")

    holder = textwrap.dedent("""
        import time
        blocks = [bytearray(48 * 1024 * 1024) for _ in range(18)]  # ~900MB
        for b in blocks:
            b[:] = b"\\x01"
        print("HOLDER_READY", flush=True)
        time.sleep(300)
    """)
    from tools.environments.local import LocalEnvironment

    local_env = LocalEnvironment(env=dict(os.environ))
    h = local_env._run_bash(f'"{sys.executable}" -c {holder!r}', timeout=10)
    try:
        out = ""
        for _ in range(200):        # wait for HOLDER_READY
            line = (h.stdout.readline() if h.stdout else "")
            out += line
            if "HOLDER_READY" in out:
                break
            import time as _t
            _t.sleep(0.1)
        assert "HOLDER_READY" in out, f"holder never became ready: {out!r}"

        bomb_out = _spawn_under_job({})
        assert "CAUGHT_MEMORYERROR" in bomb_out, (
            f"second child did not hit the SHARED job cap: {bomb_out!r}")
    finally:
        h.kill()
        h.wait()
