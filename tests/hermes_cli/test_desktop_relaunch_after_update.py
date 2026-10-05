"""An update run outside the Desktop must give back the window it closed.

`hermes update` from a terminal, a scheduled task or a fleet orchestrator stops a
running Desktop so the packed exe can replace it (Windows exe lock). Only the
Desktop's own update hand-off used to reopen the app, so every other update left
the user's window closed until they noticed and reopened it by hand.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from hermes_cli import main_desktop


@pytest.fixture(autouse=True)
def _no_leftover_launches():
    main_desktop._stopped_desktop_launches.clear()
    yield
    main_desktop._stopped_desktop_launches.clear()


def _release_exe(tmp_path: Path) -> tuple[Path, Path]:
    desktop_dir = tmp_path / "apps" / "desktop"
    exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    exe.parent.mkdir(parents=True)
    return desktop_dir, exe


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.platforms("windows")
def test_update_reopens_the_desktop_it_closed_to_free_the_build(tmp_path, monkeypatch):
    """Real processes end to end: the stop frees the exe, the relaunch restores the
    same command line, working directory and environment in the same session, once.
    The environment is the stopped app's own (an Explorer launch), never the update
    interpreter's activation (PYTHONPATH, __HERMES_ACTIVATED) leaking into the app and
    the backend it spawns."""
    import psutil

    desktop_dir, exe = _release_exe(tmp_path)
    # A real, long-lived GUI-session process standing in for Hermes.exe.
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", exe)
    argv = [str(exe), "-n", "300", "127.0.0.1"]
    launch_cwd = tmp_path / "launched-from"
    launch_cwd.mkdir()
    app_env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "__HERMES_ACTIVATED")}
    app_env["HERMES_RELAUNCH_PROBE"] = "explorer-env"
    original = subprocess.Popen(argv, cwd=launch_cwd, env=app_env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait_for(lambda: psutil.pid_exists(original.pid))
        assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == [original.pid]
        original.wait(timeout=10)
        # The update's completion interpreter is activated; none of that may leak.
        monkeypatch.setenv("PYTHONPATH", "update-activation")
        monkeypatch.setenv("__HERMES_ACTIVATED", "update-activation")

        reopened = main_desktop.relaunch_desktops_stopped_for_update()

        assert len(reopened) == 1
        proc = psutil.Process(reopened[0])
        assert Path(proc.exe()).resolve() == exe.resolve()
        assert proc.cmdline()[1:] == argv[1:]
        assert Path(proc.cwd()).resolve() == launch_cwd.resolve()
        env = {k.upper(): v for k, v in proc.environ().items()}
        assert env.get("HERMES_RELAUNCH_PROBE") == "explorer-env"
        assert "PYTHONPATH" not in env and "__HERMES_ACTIVATED" not in env
        # Drained: a second finish (retry, takeover re-entry) never opens a second window.
        assert main_desktop.relaunch_desktops_stopped_for_update() == []
    finally:
        for proc in psutil.process_iter(["exe"]):
            with __import__("contextlib").suppress(Exception):
                if proc.info["exe"] and Path(proc.info["exe"]).resolve() == exe.resolve():
                    proc.kill()


class _Proc:
    def __init__(self, pid, exe, argv, ppid):
        self.pid, self._argv, self._ppid = pid, argv, ppid
        self.info = {"pid": pid, "exe": exe}
        self.terminated = False

    def cmdline(self):
        return self._argv

    def ppid(self):
        return self._ppid

    def terminate(self):
        self.terminated = True


def _fake_psutil(monkeypatch, procs):
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(
        Process=lambda pid: types.SimpleNamespace(parents=lambda: []),
        process_iter=lambda attrs: list(procs),
        wait_procs=lambda victims, timeout=5: ([], []),
    ))


def test_only_the_main_desktop_process_is_relaunched(tmp_path, monkeypatch):
    """Electron's renderer/GPU helpers run the same exe; relaunching them would be
    garbage. Only the main process (not a child of another victim) is recorded.
    A main process whose environment cannot be read falls back to this process's
    environment minus the update interpreter's activation."""
    desktop_dir, exe = _release_exe(tmp_path)
    exe.write_text("app", encoding="utf-8")
    main = _Proc(10, str(exe), [str(exe), "--profile", "work"], ppid=1)
    renderer = _Proc(11, str(exe), [str(exe), "--type=renderer"], ppid=10)
    # A utility child carries no --type= on Windows; its parent gives it away.
    utility = _Proc(12, str(exe), [str(exe)], ppid=10)
    _fake_psutil(monkeypatch, [main, renderer, utility])
    monkeypatch.setattr(main_desktop, "_windows_session_id", lambda pid: 1)
    monkeypatch.setattr(main_desktop.sys, "platform", "win32")
    monkeypatch.setenv("PYTHONPATH", "update-activation")
    monkeypatch.setenv("__HERMES_ACTIVATED", "update-activation")
    spawned = []
    monkeypatch.setattr(main_desktop, "_spawn_detached_desktop",
                        lambda argv, *, cwd, env=None: spawned.append((argv, cwd, env)) or types.SimpleNamespace(pid=99))

    assert main_desktop._stop_desktop_processes_locking_build(desktop_dir) == [10, 11, 12]
    assert main_desktop.relaunch_desktops_stopped_for_update() == [99]
    [(argv, cwd, env)] = spawned
    assert (argv, cwd) == ([str(exe), "--profile", "work"], exe.parent)
    assert env is not None
    assert not {"PYTHONPATH", "__HERMES_ACTIVATED"} & {k.upper() for k in env}


def test_pack_and_swap_stops_of_the_same_window_reopen_it_once(tmp_path, monkeypatch):
    """The pack-time stop and the swap-time stop can both catch the same app (the user
    reopened it mid-build); the update still gives back one window, not two."""
    desktop_dir, exe = _release_exe(tmp_path)
    exe.write_text("app", encoding="utf-8")
    monkeypatch.setattr(main_desktop, "_windows_session_id", lambda pid: 1)
    monkeypatch.setattr(main_desktop.sys, "platform", "win32")
    spawned = []
    monkeypatch.setattr(main_desktop, "_spawn_detached_desktop",
                        lambda argv, *, cwd, env=None: spawned.append(argv) or types.SimpleNamespace(pid=7))
    for pid in (20, 30):
        _fake_psutil(monkeypatch, [_Proc(pid, str(exe), [str(exe)], ppid=1)])
        main_desktop._stop_desktop_processes_locking_build(desktop_dir, also_posix=True)

    assert main_desktop.relaunch_desktops_stopped_for_update() == [7]
    assert spawned == [[str(exe)]]


def test_a_desktop_in_another_windows_session_is_not_reopened_here(tmp_path, monkeypatch, capsys):
    """An update driven over SSH runs in a different Windows session than the user's
    desktop. Launching there would start an invisible app that holds the
    single-instance lock; tell the user instead."""
    desktop_dir, exe = _release_exe(tmp_path)
    exe.write_text("app", encoding="utf-8")
    _fake_psutil(monkeypatch, [_Proc(40, str(exe), [str(exe)], ppid=1)])
    monkeypatch.setattr(main_desktop, "_windows_session_id", lambda pid: 2 if pid == 40 else 0)
    monkeypatch.setattr(main_desktop.sys, "platform", "win32")
    monkeypatch.setattr(main_desktop, "_spawn_detached_desktop",
                        lambda *a, **k: pytest.fail("must not launch into the wrong session"))

    main_desktop._stop_desktop_processes_locking_build(desktop_dir)

    assert main_desktop.relaunch_desktops_stopped_for_update() == []
    assert "reopen" in capsys.readouterr().out.lower()


def test_update_reopens_the_desktop_even_when_the_build_fails(tmp_path, monkeypatch):
    """A failed pack keeps the previous app; the window the build closed comes back either way."""
    from hermes_cli import source_build as source_build_mod

    root = tmp_path / "checkout"
    (root / "apps" / "desktop").mkdir(parents=True)
    events = []

    def failing_build(desktop_dir, *, source_mode, npm, env, icons=None):
        events.append("build")
        raise subprocess.CalledProcessError(1, ["npm", "run", "builder"])

    for name, value in {
        "source_frontends": lambda _root: ("ui-tui",),
        "source_build_env": lambda **_kw: {"PATH": os.environ.get("PATH", "")},
        "prepare_source_dependencies": lambda *_a, **_kw: None,
        "source_product_current": lambda *_a, **_kw: True,
    }.items():
        monkeypatch.setattr(source_build_mod, name, value)
    monkeypatch.setattr("hermes_cli.update_stage.publish_stage", lambda _s: None, raising=False)
    monkeypatch.setattr(main_desktop, "_packaged_desktop_current_for_head", lambda *_a: False)
    monkeypatch.setattr(main_desktop, "build_prepared_desktop", failing_build)
    monkeypatch.setattr(main_desktop, "relaunch_desktops_stopped_for_update",
                        lambda: events.append("relaunch") or [])

    with pytest.raises(source_build_mod.ProductBuildError):
        source_build_mod.build_update_products(root, desktop=True)

    assert events == ["build", "relaunch"]
