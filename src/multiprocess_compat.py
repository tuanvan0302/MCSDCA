from __future__ import annotations

import os
import threading
from typing import Any


def patch_multiprocess_resource_tracker() -> None:
    """Patch a Windows/Python 3.12 shutdown bug in multiprocess.

    multiprocess 0.70.19 assumes ``threading.RLock`` exposes
    ``_recursion_count()`` in ``ResourceTracker._stop_locked``. On the Windows
    Python 3.12 build used here the lock is ``_thread.RLock`` and does not have
    that private method, which causes an ignored exception during interpreter
    shutdown after an otherwise successful run.
    """

    try:
        import multiprocess.resource_tracker as resource_tracker
    except ImportError:
        return

    lock = threading.RLock()
    if hasattr(lock, "_recursion_count"):
        return

    tracker_cls: Any = resource_tracker.ResourceTracker
    if getattr(tracker_cls, "_mcsdca_windows_rlock_patch", False):
        return

    def _stop_locked(
        self: Any,
        close: Any = os.close,
        waitpid: Any = os.waitpid,
        waitstatus_to_exitcode: Any = os.waitstatus_to_exitcode,
    ) -> Any:
        recursion_count = getattr(self._lock, "_recursion_count", None)
        if recursion_count is not None and recursion_count() > 1:
            return self._reentrant_call_error()
        if self._fd is None:
            return None
        if self._pid is None:
            return None

        close(self._fd)
        self._fd = None

        waitpid(self._pid, 0)
        self._pid = None
        return None

    tracker_cls._stop_locked = _stop_locked
    tracker_cls._mcsdca_windows_rlock_patch = True
