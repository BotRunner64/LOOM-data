"""Task claims on the shared filesystem: how workers divide a plan without a coordinator.

A worker owns a task while `claims/<index>` exists with its name inside. The
file is created with O_EXCL, so exactly one worker gets it. While working, the
owner keeps the file's mtime fresh; a claim whose mtime stops moving belongs
to a dead worker, and any other worker may take it over. Ages are measured
against the filesystem's own clock, never a node's.
"""
from contextlib import contextmanager
import os
import socket
import threading
import uuid

HEARTBEAT_S = 60     # how often an owner refreshes its claim
STALE_S = 600        # a claim this old without a heartbeat is abandoned
POLL_S = 15          # how often a worker rechecks tasks other workers hold


class Claims:
    def __init__(self, directory):
        self.directory = directory
        self.owner = f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        self.clock = directory / f'.clock-{self.owner}'

    def path(self, index):
        return self.directory / str(index)

    def now(self):
        """Current time as the shared filesystem sees it."""
        with open(self.clock, 'w'):
            pass
        os.utime(self.clock)
        return self.clock.stat().st_mtime

    def acquire(self, index):
        try:
            fd = os.open(self.path(index), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, 'w') as handle:
            handle.write(self.owner)
        return True

    def take_over(self, index):
        """Claim a task whose owner stopped its heartbeat. At most one worker succeeds.

        Contenders for the same abandoned claim all saw the same last heartbeat,
        so they compete for one lock named after it; the winner then replaces the
        claim atomically. The lock stays until the next prepare: a contender that
        read the old heartbeat late finds it taken and gives up.
        """
        path = self.path(index)
        try:
            seen = path.stat().st_mtime_ns
            if self.now() - seen / 1e9 < STALE_S:
                return False
        except FileNotFoundError:
            return self.acquire(index)  # released meanwhile
        # Written before competing, so that a full disk cannot strand the lock.
        temporary = self.directory / f'.claim-{index}-{self.owner}'
        temporary.write_text(self.owner)
        lock = self.directory / f'.takeover-{index}-{seen}'
        try:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            except FileExistsError:
                # The winner replaces the claim at once. A lock this old next to an unchanged
                # claim means the winner died first: restart the heartbeat clock on the dead
                # claim, so that it goes stale again under a new lock name.
                if self.now() - lock.stat().st_mtime >= STALE_S and path.stat().st_mtime_ns == seen:
                    os.utime(path)
                return False
            if path.stat().st_mtime_ns != seen:
                return False  # the owner is alive after all, or someone else holds it now
            os.replace(temporary, path)
            return True
        except FileNotFoundError:
            return False  # released or cleaned up meanwhile; the next poll acquires it normally
        finally:
            temporary.unlink(missing_ok=True)

    def owned(self, index):
        try:
            return self.path(index).read_text() == self.owner
        except OSError:
            return False

    def release(self, index):
        if self.owned(index):
            self.path(index).unlink(missing_ok=True)

    @contextmanager
    def held(self, index):
        """Keep the claim alive from a background thread while the task runs."""
        stop = threading.Event()

        def beat():
            while not stop.wait(HEARTBEAT_S):
                try:
                    if self.owned(index):
                        os.utime(self.path(index))
                except OSError:
                    pass

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()
            self.release(index)

    def close(self):
        self.clock.unlink(missing_ok=True)

    def active(self):
        """[(task index, owner, seconds since last heartbeat)] of current claims."""
        now, result = self.now(), []
        for name in os.listdir(self.directory):
            if name.isdecimal():
                path = self.path(name)
                try:
                    result.append((int(name), path.read_text(), now - path.stat().st_mtime))
                except OSError:
                    pass
        return sorted(result)
