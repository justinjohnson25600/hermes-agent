"""Windows Job Object commit cap for agent-spawned local processes.

Why this exists (2026-09-06, NucBox_M8): a coding-session test suite ran an
unbounded ``list()`` over an infinite generator and allocated ~45 GB of commit
in ~2 minutes. System commit hit 100%, DWM restart-looped, every app's
allocations failed, and the host hard-hung until a manual power reset — three
times in one evening, because the session auto-resumed after each reboot and
re-ran the same command. A Job Object carrying ``JOB_OBJECT_LIMIT_JOB_MEMORY``
caps the TOTAL commit of every process in the job (spawned child + all its
descendants): a runaway allocator gets ``MemoryError`` / allocation failures
INSIDE the child and dies alone; the host keeps running.

Scope: the local backend only — ``LocalEnvironment._run_bash`` assigns each
spawned process to the per-Hermes-process job. Container backends cap memory
through their own ``container_memory`` settings; SSH children run on another
host. ``KILL_ON_JOB_CLOSE`` is deliberately NOT set: background terminal
processes are contracted to outlive the spawning turn/gateway, and the job
handle intentionally stays open for this process's lifetime (one handle).

Configuration: ``terminal.job_commit_limit_gb`` in config.yaml (bridged to
``TERMINAL_JOB_COMMIT_LIMIT_GB``, scope-aware like every TERMINAL_* var).
Default 8 GB; ``0`` disables the cap entirely. Fail-open: if the Job Object
cannot be created or a process cannot be assigned (e.g. a parent job
disallows nesting on a pre-Win8 host), the command still runs — a monitoring
cap must never be the reason a terminal breaks.
"""

import ctypes
import logging
import os
import threading

logger = logging.getLogger("tools.environments.win_job_cap")

_IS_WINDOWS = os.name == "nt"

_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200

_DEFAULT_LIMIT_GB = 8


def _limit_bytes() -> int:
    """Configured job commit cap in bytes, or 0 when disabled."""
    if not _IS_WINDOWS:
        return 0
    try:
        from tools.terminal_tool_config import _tenv

        raw = _tenv("TERMINAL_JOB_COMMIT_LIMIT_GB", str(_DEFAULT_LIMIT_GB))
        gb = float(raw)
    except Exception:
        return _DEFAULT_LIMIT_GB * 1024**3
    if gb <= 0:
        return 0
    return int(gb * 1024**3)


if _IS_WINDOWS:
    from ctypes import wintypes

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JobObjectExtendedLimitInformation = 9
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                      wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,
                                                   wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

_job_handle = None
_job_lock = threading.Lock()
_job_failed = False


def _create_job(limit_bytes: int):
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_JOB_MEMORY
    info.JobMemoryLimit = limit_bytes
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    if not _kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
            ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return job


def _get_job():
    """Create-once job handle for this Hermes process (None when unusable)."""
    global _job_handle, _job_failed
    if not _IS_WINDOWS or _job_failed:
        return None
    with _job_lock:
        if _job_failed:
            return None
        if _job_handle is not None:
            return _job_handle
        limit = _limit_bytes()
        if limit <= 0:
            logger.debug("job commit cap disabled (job_commit_limit_gb<=0)")
            _job_failed = True  # cached: disabled for this process
            return None
        try:
            _job_handle = _create_job(limit)
            logger.info("terminal job commit cap active: %.1f GB",
                        limit / 1024**3)
        except OSError as exc:
            # Fail-open: no cap, but terminals keep working.
            logger.warning("job commit cap unavailable (%s); "
                           "agent processes run uncapped", exc)
            _job_failed = True
            return None
        return _job_handle


def assign_to_job(proc) -> None:
    """Assign a just-spawned *proc* (subprocess.Popen) to the commit-capped
    job. Best-effort: never raises, never blocks — a cap we cannot apply must
    not break the spawn it was meant to guard."""
    job = _get_job()
    if job is None:
        return
    handle = _kernel32.OpenProcess(
        _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid)
    if not handle:
        logger.debug("job cap: OpenProcess(%d) failed: %s", proc.pid,
                     ctypes.WinError(ctypes.get_last_error()))
        return
    try:
        if not _kernel32.AssignProcessToJobObject(job, handle):
            # Nested jobs are supported Win8+; a failure here means an
            # ancestor job forbids it — run uncapped rather than fail.
            logger.debug("job cap: AssignProcessToJobObject(%d) failed: %s",
                         proc.pid, ctypes.WinError(ctypes.get_last_error()))
    finally:
        _kernel32.CloseHandle(handle)
