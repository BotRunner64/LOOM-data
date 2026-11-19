"""Workers dividing one plan through claims on the shared filesystem."""
import errno
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from support import manifest
from loom_data import paths, resumable
from loom_data.convert import claims as claim_files, job as parallel
from loom_data.convert.claims import Claims
from loom_data.convert.config import make_config
from loom_data.sources.hifi_umi import HifiUmi
from test_convert_hifi_umi import make_source as hifi_source
from test_convert_job import add_hifi_episode

CONVERT = HifiUmi.episode


def age(path, seconds):
    """Make a claim look as if its last heartbeat was `seconds` ago."""
    past = time.time() - seconds
    os.utime(path, (past, past))


def wait_until(condition, timeout=20):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError('condition not reached')
        time.sleep(.005)


class ClaimTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_only_one_of_many_contenders_acquires(self):
        contenders = [Claims(self.directory) for _ in range(16)]
        self.assertEqual(len({c.owner for c in contenders}), 16)
        for index in range(20):
            barrier, won = threading.Barrier(len(contenders)), []

            def contend(claims):
                barrier.wait()
                if claims.acquire(index):
                    won.append(claims)

            threads = [threading.Thread(target=contend, args=(c,)) for c in contenders]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(won), 1)
            self.assertEqual([c.owned(index) for c in contenders].count(True), 1)
            self.assertEqual(self.directory.joinpath(str(index)).read_text(), won[0].owner)

    def test_release_and_ownership_belong_to_the_owner_only(self):
        mine, other = Claims(self.directory), Claims(self.directory)
        self.assertTrue(mine.acquire(3))
        self.assertFalse(mine.acquire(3))
        self.assertFalse(other.acquire(3))
        self.assertEqual((mine.owned(3), other.owned(3), other.owned(4)), (True, False, False))
        other.release(3)
        self.assertTrue(mine.owned(3))
        mine.release(3)
        self.assertFalse(self.directory.joinpath('3').exists())
        mine.release(3)  # releasing twice is harmless
        self.assertTrue(other.acquire(3))

    def test_take_over_needs_a_stale_claim_and_leaves_only_its_lock(self):
        dead, first, second = Claims(self.directory), Claims(self.directory), Claims(self.directory)
        self.assertTrue(dead.acquire(0))
        path = self.directory / '0'
        self.assertFalse(first.take_over(0))                # fresh: its owner is alive
        age(path, claim_files.STALE_S - 30)
        self.assertFalse(first.take_over(0))                # quiet, but not for long enough
        self.assertTrue(dead.owned(0))
        self.assertEqual([p.name for p in self.directory.iterdir() if p.name.startswith('.takeover')], [])
        age(path, claim_files.STALE_S + 30)
        heartbeat = path.stat().st_mtime_ns
        self.assertTrue(first.take_over(0))
        self.assertEqual((first.owned(0), dead.owned(0)), (True, False))
        self.assertLess(first.now() - path.stat().st_mtime, 30)  # the new owner's claim starts fresh
        self.assertFalse(second.take_over(0))
        dead.release(0)                                     # the previous owner cannot release it
        self.assertTrue(first.owned(0))
        self.assertTrue(second.take_over(7))                # nothing to take over: a plain acquire
        for claims in (dead, first, second):
            claims.close()
        # The lock named after the dead owner's last heartbeat stays until the next prepare.
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), [f'.takeover-0-{heartbeat}', '0', '7'])
        # The new owner dies too: its claim has another last heartbeat, hence another lock.
        age(path, claim_files.STALE_S + 60)
        self.assertTrue(second.take_over(0))
        self.assertTrue(second.owned(0))
        self.assertEqual(len(list(self.directory.glob('.takeover-0-*'))), 2)

    def test_only_one_of_many_contenders_takes_over_a_stale_claim(self):
        contenders = [Claims(self.directory) for _ in range(8)]
        for index in range(150):
            path = self.directory / str(index)
            path.write_text('gone-node-4242-0123abcd')
            age(path, claim_files.STALE_S + 30)
            barrier, won = threading.Barrier(len(contenders)), []

            def contend(claims):
                barrier.wait()
                if claims.take_over(index):
                    won.append(claims)

            threads = [threading.Thread(target=contend, args=(c,)) for c in contenders]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(won), 1, index)
            self.assertEqual(path.read_text(), won[0].owner)
            self.assertEqual([c.owned(index) for c in contenders].count(True), 1)
        self.assertEqual(list(self.directory.glob('.claim-*')), [])

    def interleave(self, contender, action):
        """Run `action` between the contender's reading of the last heartbeat and its attempt to take over."""
        now = contender.now

        def after_reading_the_heartbeat():
            contender.now = now
            action()
            return now()
        contender.now = after_reading_the_heartbeat

    def test_late_contender_loses_to_a_take_over_completed_after_it_read_the_heartbeat(self):
        first, late = Claims(self.directory), Claims(self.directory)
        path = self.directory / '0'
        path.write_text('gone-node-4242-0123abcd')
        age(path, claim_files.STALE_S + 30)
        done = []
        self.interleave(late, lambda: done.append(first.take_over(0)))
        self.assertFalse(late.take_over(0))
        self.assertEqual(done, [True])
        self.assertEqual((first.owned(0), late.owned(0)), (True, False))
        self.assertEqual(path.read_text(), first.owner)
        # The same when the winner has already finished and released the task: the stale reading
        # earns nothing, and the next poll finds the task free.
        first.release(0)
        path.write_text('gone-node-4242-0123abcd')
        age(path, claim_files.STALE_S + 60)
        self.interleave(late, lambda: (first.take_over(0), first.release(0)))
        self.assertFalse(late.take_over(0))
        self.assertFalse(path.exists())
        self.assertTrue(late.take_over(0))
        self.assertTrue(late.owned(0))

    def test_contender_gives_up_when_the_owner_heartbeats_after_it_read_the_claim(self):
        owner, contender = Claims(self.directory), Claims(self.directory)
        self.assertTrue(owner.acquire(0))
        path = self.directory / '0'
        age(path, claim_files.STALE_S + 30)                 # stalled, not dead
        stale = path.stat().st_mtime_ns
        self.interleave(contender, lambda: os.utime(path))
        self.assertFalse(contender.take_over(0))
        self.assertTrue(owner.owned(0))
        self.assertNotEqual(path.stat().st_mtime_ns, stale)
        self.assertFalse(contender.take_over(0))            # and the claim is fresh again
        # If the owner then really dies, the lock spent on the old heartbeat does not block the take-over.
        self.assertEqual([p.name for p in self.directory.glob('.takeover-*')], [f'.takeover-0-{stale}'])
        age(path, claim_files.STALE_S + 60)
        self.assertTrue(contender.take_over(0))
        self.assertEqual((owner.owned(0), contender.owned(0)), (False, True))

    def stale_claim(self, index=0):
        path = self.directory / str(index)
        path.write_text('gone-node-4242-0123abcd')
        age(path, claim_files.STALE_S + 30)
        return path, path.stat().st_mtime_ns

    def hidden(self, prefix):
        return sorted(p.name for p in self.directory.glob(f'{prefix}*'))

    def test_full_disk_during_take_over_strands_no_lock(self):
        path, heartbeat = self.stale_claim()
        unlucky, later = Claims(self.directory), Claims(self.directory)
        with patch.object(Path, 'write_text', side_effect=OSError(errno.ENOSPC, 'No space left on device')):
            with self.assertRaises(OSError):
                unlucky.take_over(0)
        self.assertEqual(self.hidden('.takeover'), [])
        self.assertEqual(path.stat().st_mtime_ns, heartbeat)
        self.assertTrue(later.take_over(0))
        self.assertTrue(later.owned(0))
        self.assertEqual(self.hidden('.claim'), [])

    def test_winner_that_died_between_lock_and_replace_does_not_strand_the_task(self):
        path, heartbeat = self.stale_claim()
        lock = self.directory / f'.takeover-0-{heartbeat}'
        lock.touch()                                        # its winner never replaced the claim
        first, second = Claims(self.directory), Claims(self.directory)
        # While the lock is recent its winner may still be about to replace the claim: hands off.
        for _ in range(3):
            self.assertFalse(first.take_over(0))
        self.assertEqual((path.stat().st_mtime_ns, path.read_text()), (heartbeat, 'gone-node-4242-0123abcd'))
        # A lock that old beside an untouched claim has no winner behind it. Nobody may take the claim
        # under that lock, but the heartbeat clock is restarted ...
        age(lock, claim_files.STALE_S + 30)
        self.assertFalse(first.take_over(0))
        self.assertEqual(path.read_text(), 'gone-node-4242-0123abcd')
        self.assertLess(first.now() - path.stat().st_mtime, 30)
        self.assertFalse(second.take_over(0))
        # ... so the claim goes stale once more, under a new lock that someone can win.
        age(path, claim_files.STALE_S + 60)
        self.assertEqual([c.take_over(0) for c in (first, second)], [True, False])
        self.assertTrue(first.owned(0))
        self.assertEqual(len(self.hidden('.takeover')), 2)
        self.assertEqual(self.hidden('.claim'), [])

    def test_old_lock_is_ignored_once_its_winner_did_replace_the_claim(self):
        path, heartbeat = self.stale_claim()
        winner, late = Claims(self.directory), Claims(self.directory)
        self.assertTrue(winner.take_over(0))
        age(self.directory / f'.takeover-0-{heartbeat}', claim_files.STALE_S + 30)
        taken = path.stat().st_mtime_ns
        # A contender that read the heartbeat before the take-over must not refresh or disturb the new claim.
        self.interleave_reading(late, heartbeat)
        self.assertFalse(late.take_over(0))
        self.assertEqual((path.stat().st_mtime_ns, winner.owned(0)), (taken, True))
        self.assertEqual(self.hidden('.claim'), [])

    def interleave_reading(self, contender, heartbeat):
        """Make the contender believe the claim's last heartbeat is still `heartbeat` when it first looks."""
        path = self.directory / '0'
        now, current = contender.now, path.stat().st_mtime_ns
        os.utime(path, ns=(heartbeat, heartbeat))

        def after_reading_the_heartbeat():
            contender.now = now
            os.utime(path, ns=(current, current))
            return now()
        contender.now = after_reading_the_heartbeat

    def test_staleness_is_measured_against_the_filesystem_clock(self):
        claims = Claims(self.directory)
        self.assertAlmostEqual(claims.now(), time.time(), delta=5)
        self.assertTrue(Claims(self.directory).acquire(0))
        age(self.directory / '0', claim_files.STALE_S + 30)
        # A node whose own clock is far behind still sees the claim as stale.
        with patch.object(time, 'time', return_value=0.):
            self.assertTrue(claims.take_over(0))

    def test_heartbeat_keeps_a_held_claim_fresh_and_releases_it(self):
        claims, path = Claims(self.directory), self.directory / '5'
        self.assertTrue(claims.acquire(5))
        with patch.object(claim_files, 'HEARTBEAT_S', .01), claims.held(5):
            for _ in range(3):
                age(path, claim_files.STALE_S + 30)
                wait_until(lambda: claims.now() - path.stat().st_mtime < claim_files.STALE_S)
            self.assertFalse(Claims(self.directory).take_over(5))
        self.assertFalse(path.exists())

    def test_heartbeat_stops_refreshing_a_claim_it_lost(self):
        claims, path = Claims(self.directory), self.directory / '5'
        self.assertTrue(claims.acquire(5))
        with patch.object(claim_files, 'HEARTBEAT_S', .01), claims.held(5):
            path.write_text('another-worker-1-abcdef01')
            age(path, 1000)
            stamp = path.stat().st_mtime
            time.sleep(.1)
            self.assertEqual(path.stat().st_mtime, stamp)
        # Leaving the task does not release a claim that now belongs to someone else.
        self.assertEqual(path.read_text(), 'another-worker-1-abcdef01')

    def test_active_lists_claims_with_owner_and_age(self):
        first, second = Claims(self.directory), Claims(self.directory)
        self.assertEqual(first.active(), [])
        self.assertTrue(first.acquire(12))
        self.assertTrue(second.acquire(3))
        age(self.directory / '3', 100)
        (self.directory / '.stale-9-someone').write_text('x')
        listed = second.active()
        self.assertEqual([(index, owner) for index, owner, _ in listed], [(3, second.owner), (12, first.owner)])
        self.assertAlmostEqual(listed[0][2], 100, delta=5)
        self.assertAlmostEqual(listed[1][2], 0, delta=5)


class WorkerClaimTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.output = self.root / 'raw', self.root / 'export'
        hifi_source(self.source)
        self.config = make_config('hifi_umi', self.source, left_eef_offset=[0., 0., 0.], right_eef_offset=[0., 0., 0.])
        # Nothing here may wait for the real intervals.
        for name, value in (('POLL_S', .02), ('HEARTBEAT_S', .02)):
            patcher = patch.object(claim_files, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, episodes):
        for index in range(8, 7 + episodes):
            add_hifi_episode(self.source, index)
        self.store = parallel.prepare(self.config, self.output)
        self.assertEqual(len(self.store.plan['tasks']), episodes)
        self.claims = self.store.work / 'claims'
        return self.store

    def start(self, rank=0, count=1):
        """Run a worker in a thread; .code is its exit status once it is done."""
        def run():
            try:
                thread.code = parallel.worker(self.output, rank, count, report=thread.log.append)
            except BaseException as exc:
                thread.code = exc
        thread = threading.Thread(target=run, daemon=True)
        thread.log, thread.code = [], None
        thread.start()
        return thread

    def finish(self, *threads):
        for thread in threads:
            thread.join(60)
            self.assertFalse(thread.is_alive())
        return [thread.code for thread in threads]

    def counted(self, behaviour=None):
        """Patch the episode conversion, recording (episode id, thread) of every attempt."""
        calls = []

        def episode(adapter, task, destination):
            calls.append((destination.name, threading.current_thread()))
            if behaviour:
                behaviour(destination.name)
            return CONVERT(adapter, task, destination)
        patcher = patch.object(HifiUmi, 'episode', episode)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def errors(self):
        return sorted(p.name for p in (self.store.work / 'errors').iterdir())

    def test_concurrent_workers_convert_every_episode_exactly_once(self):
        store = self.prepare(8)
        calls = self.counted()
        workers = [self.start(rank, 3) for rank in range(3)]
        self.assertEqual(self.finish(*workers), [0, 0, 0])
        self.assertEqual(sorted(name for name, _ in calls), sorted(store.ids))
        self.assertEqual(list(self.claims.iterdir()), [])
        self.assertEqual(list((store.work / 'staging').iterdir()), [])
        parallel.finalize(self.output)
        self.assertEqual([r['episode_id'] for r in manifest(self.output)], store.ids)

    def test_later_worker_takes_the_remaining_tasks_and_waits_for_the_held_one(self):
        store = self.prepare(4)
        entered, proceed = threading.Event(), threading.Event()

        def hold_first(episode_id):
            if episode_id == store.ids[0]:
                entered.set()
                self.assertTrue(proceed.wait(60))
        calls = self.counted(hold_first)
        early = self.start()
        self.assertTrue(entered.wait(60))
        late = self.start()
        try:
            # The late worker converts everything the early one has not reached ...
            wait_until(lambda: len(calls) == 4)
            self.assertEqual({name for name, thread in calls if thread is late}, set(store.ids[1:]))
            # ... and then stays, rechecking the task the early worker still holds.
            wait_until(lambda: sum(m.startswith('Task') for m in late.log) == 3 and not early.log[-1].startswith('Claim'))
            time.sleep(.1)
            self.assertTrue(late.is_alive())
            self.assertEqual(len(calls), 4)
            self.assertEqual([index for index, _, _ in Claims(self.claims).active()], [0])
        finally:
            proceed.set()
        self.assertEqual(self.finish(early, late), [0, 0])
        self.assertEqual([(name, thread is early) for name, thread in calls if name == store.ids[0]],
                         [(store.ids[0], True)])
        self.assertEqual(len(calls), 4)
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 4)

    def test_dead_workers_claim_is_taken_over_but_a_live_one_is_waited_on(self):
        store = self.prepare(2)
        calls = self.counted()
        foreign = self.claims / '0'
        foreign.write_text('gone-node-4242-0123abcd')
        polls = []
        take_over = Claims.take_over

        def polled(claims, index):
            polls.append(index)
            return take_over(claims, index)

        with patch.object(Claims, 'take_over', polled):
            worker = self.start()
            # A claim with a recent heartbeat is left alone however often it is rechecked.
            wait_until(lambda: len(polls) >= 5)
            self.assertTrue(worker.is_alive())
            self.assertEqual([name for name, _ in calls], [store.ids[1]])
            self.assertEqual(foreign.read_text(), 'gone-node-4242-0123abcd')
            self.assertIsNone(store.state(store.ids[0]))
            # Its owner died: no heartbeat for STALE_S.
            age(foreign, claim_files.STALE_S + 30)
            self.assertEqual(self.finish(worker), [0])
        self.assertEqual(set(polls), {0})
        self.assertEqual([name for name, _ in calls], [store.ids[1], store.ids[0]])
        # The claim is released; only the take-over lock stays, until the next prepare.
        lock, = self.claims.iterdir()
        self.assertTrue(lock.name.startswith('.takeover-0-'))
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 2)

    def test_first_pass_never_takes_over_so_stale_claims_wait_one_poll(self):
        store = self.prepare(1)
        self.counted()
        (self.claims / '0').write_text('gone-node-4242-0123abcd')
        age(self.claims / '0', claim_files.STALE_S + 30)
        with patch.object(parallel.time, 'sleep', wraps=time.sleep) as sleep:
            self.assertEqual(parallel.worker(self.output), 0)
        sleep.assert_called_once_with(claim_files.POLL_S)
        self.assertIsNotNone(store.state(store.ids[0]))

    def test_heartbeat_protects_a_task_that_runs_longer_than_the_stale_limit(self):
        store = self.prepare(1)
        calls = self.counted(lambda episode_id: time.sleep(1.2))
        with patch.object(claim_files, 'STALE_S', .5):
            slow = self.start()
            wait_until(lambda: calls)
            waiting = self.start()
            self.assertEqual(self.finish(slow, waiting), [0, 0])
        self.assertEqual([(name, thread is slow) for name, thread in calls], [(store.ids[0], True)])
        self.assertEqual(self.errors(), [])

    def test_without_a_heartbeat_the_same_task_is_taken_over(self):
        # The counterpart of the test above: what the heartbeat prevents, and how the loser ends.
        store = self.prepare(1)
        entered, proceed = threading.Event(), threading.Event()

        def stall(episode_id):
            if not entered.is_set():
                entered.set()
                self.assertTrue(proceed.wait(60))
        calls = self.counted(stall)
        with patch.object(claim_files, 'STALE_S', .2), patch.object(claim_files, 'HEARTBEAT_S', 3600):
            stalled = self.start()
            self.assertTrue(entered.wait(60))
            rescuer = self.start()
            self.assertEqual(self.finish(rescuer), [0])
            self.assertIsNotNone(store.state(store.ids[0]))
            committed = {p: p.stat().st_mtime_ns for p in (store.dataset / 'episodes').rglob('*') if p.is_file()}
            proceed.set()
            self.assertEqual(self.finish(stalled), [0])
        self.assertEqual([thread is stalled for _, thread in calls], [True, False])
        self.assertTrue(any('Task 0: finished by another worker' in message for message in stalled.log))
        self.assertEqual(self.errors(), [])
        self.assertEqual({p: p.stat().st_mtime_ns for p in (store.dataset / 'episodes').rglob('*') if p.is_file()},
                         committed)
        self.assertEqual(list((store.work / 'staging').iterdir()), [])
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 1)

    def take_over_during_conversion(self, store, *, keep_claim):
        """While the worker converts episode 0, another worker commits it; optionally it also holds the claim."""
        rival = Claims(self.claims)
        self.addCleanup(rival.close)
        attempt = store.work / 'staging' / 'rival-attempt'

        def episode(adapter, task, destination):
            if destination.name == store.ids[0] and not attempt.exists():
                if not keep_claim:
                    (self.claims / '0').write_text(rival.owner)
                other = attempt / destination.name
                adapter.commit(parallel.Store.load(self.output), destination.name, other, CONVERT(adapter, task, other))
            return CONVERT(adapter, task, destination)
        return rival, patch.object(HifiUmi, 'episode', episode)

    def episode_files(self):
        return {p: (p.stat().st_mtime_ns, p.stat().st_size, p.stat().st_ino)
                for p in (self.store.dataset / 'episodes').rglob('*') if p.is_file()}

    def test_worker_that_lost_its_claim_mid_task_discards_its_result_and_continues(self):
        store = self.prepare(2)
        rival, patched = self.take_over_during_conversion(store, keep_claim=False)
        log = []
        with patched:
            self.assertEqual(parallel.worker(self.output, report=log.append), 0)
        finished, = [message for message in log if 'finished by another worker' in message]
        self.assertIn('Task 0', finished)
        self.assertIn('already committed', finished)
        self.assertEqual(self.errors(), [])                      # not a failure of the episode
        self.assertTrue(rival.owned(0))                          # the new owner's claim is not released for it
        self.assertEqual([index for index, _, _ in rival.active()], [0])
        self.assertIsNotNone(store.state(store.ids[1]))          # the worker went on to the next task
        self.assertEqual(sorted(p.name for p in (store.dataset / 'episodes').iterdir()), store.ids)
        self.assertEqual([p.name for p in (store.work / 'staging').iterdir()], ['rival-attempt'])
        rival.release(0)
        parallel.prepare(self.config, self.output)
        with patch.object(HifiUmi, 'episode', side_effect=AssertionError('both are committed')):
            self.assertEqual(parallel.worker(self.output), 0)
        parallel.finalize(self.output)
        self.assertEqual([r['episode_id'] for r in manifest(self.output)], store.ids)

    def test_episode_finished_by_another_worker_is_benign_even_while_owning_the_claim(self):
        store = self.prepare(2)
        _, patched = self.take_over_during_conversion(store, keep_claim=True)
        log = []
        with patched:
            self.assertEqual(parallel.worker(self.output, report=log.append), 0)
        self.assertEqual(sum('Task 0: finished by another worker' in message for message in log), 1)
        self.assertEqual(self.errors(), [])
        self.assertEqual(list(self.claims.iterdir()), [])
        self.assertEqual(len(store.audit()[0]), 2)

    def damaged_during_conversion(self, store, change_owner):
        def episode(adapter, task, destination):
            if destination.name == store.ids[0]:
                (store.work / 'records' / f'{destination.name}.json').write_text('{')
                if change_owner:
                    (self.claims / '0').write_text('another-node-1-0123abcd')
            return CONVERT(adapter, task, destination)
        return patch.object(HifiUmi, 'episode', episode)

    def test_damaged_state_is_still_an_error_while_owning_the_claim(self):
        store = self.prepare(2)
        with self.damaged_during_conversion(store, False), self.assertRaisesRegex(parallel.StateError, 'Cannot read'):
            parallel.worker(self.output)
        self.assertEqual(self.errors(), [])                      # never recorded as an episode failure
        self.assertEqual(list(self.claims.iterdir()), [])
        self.assertEqual(list((store.work / 'staging').iterdir()), [])
        self.assertEqual(list((store.dataset / 'episodes').iterdir()), [])  # the worker stopped at once
        with self.assertRaisesRegex(parallel.StateError, 'Cannot read'):
            parallel.prepare(self.config, self.output)

    def test_damaged_state_seen_after_losing_the_claim_is_left_to_the_new_owner(self):
        store = self.prepare(2)
        log = []
        with self.damaged_during_conversion(store, True):
            self.assertEqual(parallel.worker(self.output, report=log.append), 0)
        lost, = [message for message in log if 'lost its claim to another worker' in message]
        self.assertIn('Cannot read', lost)
        self.assertEqual(self.errors(), [])
        self.assertEqual((self.claims / '0').read_text(), 'another-node-1-0123abcd')
        self.assertEqual([p.name for p in (store.dataset / 'episodes').iterdir()], [store.ids[1]])

    def test_rescuer_whose_commit_comes_second_keeps_the_stalled_workers_result(self):
        store = self.prepare(1)
        episode_id = store.ids[0]
        stalled_entered, resume, rescuer_entered = threading.Event(), threading.Event(), threading.Event()

        def order(_):
            if not stalled_entered.is_set():
                stalled_entered.set()
                self.assertTrue(resume.wait(60))
            else:
                # The rescuer is past the pending check; it converts only after the stalled worker commits.
                rescuer_entered.set()
                wait_until(lambda: (store.dataset / 'episodes' / episode_id).exists(), 60)
        calls = self.counted(order)
        commits = []
        commit = parallel.Store.commit

        def committed(store, *args):
            commit(store, *args)
            commits.append((threading.current_thread(), self.episode_files()))
        with patch.object(claim_files, 'STALE_S', .2), patch.object(claim_files, 'HEARTBEAT_S', 3600), \
                patch.object(parallel.Store, 'commit', committed):
            stalled = self.start()
            self.assertTrue(stalled_entered.wait(60))
            rescuer = self.start()
            self.assertTrue(rescuer_entered.wait(60))
            owner = (self.claims / '0').read_text()
            resume.set()
            self.assertEqual(self.finish(stalled, rescuer), [0, 0])
        self.assertEqual([thread is stalled for _, thread in calls], [True, False])
        (thread, files), = commits                               # one commit, by the stalled worker
        self.assertIs(thread, stalled)
        self.assertEqual(len(files), 4)
        self.assertEqual(self.episode_files(), files)            # the rescuer changed nothing
        self.assertNotIn(owner, ' '.join(stalled.log))           # ... although it owned the claim
        self.assertIn(owner, ' '.join(rescuer.log))
        self.assertTrue(any(f'Task 0: finished by another worker: {episode_id}: already committed' in m
                            for m in rescuer.log))
        self.assertFalse(any('finished by another' in m or 'lost its claim' in m for m in stalled.log))
        self.assertEqual(self.errors(), [])
        self.assertEqual(list((store.work / 'staging').iterdir()), [])
        self.assertEqual([p.name for p in self.claims.iterdir() if not p.name.startswith('.takeover-')], [])
        self.assertEqual([r['episode_id'] for r in store.audit()[0]], [episode_id])
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 1)

    def test_two_workers_committing_one_episode_at_the_same_time_leave_one_consistent_commit(self):
        store = self.prepare(1)
        episode_id = store.ids[0]
        entered, resume = threading.Event(), threading.Event()
        calls, sizes = [], {}

        def episode(adapter, task, destination):
            first = not calls
            calls.append(threading.current_thread())
            if first:
                entered.set()
                self.assertTrue(resume.wait(60))
            record = CONVERT(adapter, task, destination)
            if not first:
                # The two attempts do not produce the same bytes (another node, another codec build).
                with open(destination / 'top.mp4', 'ab') as handle:
                    handle.write(bytes(8))
            sizes[threading.current_thread()] = {p.name: p.stat().st_size for p in destination.iterdir()}
            return record
        # Both have validated their attempt and reach the commit lock at the same moment.
        together = threading.Barrier(2, timeout=60)
        os_open = os.open

        def at_the_commit_lock(path, *args, **kwargs):
            if Path(path).parent == store.work / 'commits':
                together.wait()
            return os_open(path, *args, **kwargs)
        with patch.object(claim_files, 'STALE_S', .2), patch.object(claim_files, 'HEARTBEAT_S', 3600), \
                patch.object(HifiUmi, 'episode', episode), patch.object(os, 'open', at_the_commit_lock), \
                patch('sys.stdout', new_callable=io.StringIO) as out:
            stalled = self.start()
            self.assertTrue(entered.wait(60))
            rescuer = self.start()
            wait_until(lambda: len(calls) == 2, 60)
            resume.set()
            self.assertEqual(self.finish(stalled, rescuer), [0, 0])
        self.assertEqual(together.n_waiting, 0)
        self.assertFalse(together.broken)                        # both really went as far as the lock
        self.assertNotEqual(sizes[stalled], sizes[rescuer])
        self.assertEqual(out.getvalue().count(f'Committed {episode_id}'), 1)
        lost = [w for w in (stalled, rescuer)
                if any(f'Task 0: finished by another worker: {episode_id}: being committed' in m for m in w.log)]
        self.assertEqual(len(lost), 1)
        winner = rescuer if lost[0] is stalled else stalled
        self.assertFalse(any('failed' in m or 'lost its claim' in m for m in stalled.log + rescuer.log))
        self.assertEqual(self.errors(), [])
        self.assertEqual([p.name for p in (store.dataset / 'episodes').iterdir()], [episode_id])
        self.assertEqual([p.name for p in (store.work / 'commits').iterdir()], [episode_id])
        self.assertEqual(list((store.work / 'staging').iterdir()), [])
        # Record and files are the winner's, whichever of the two that was.
        directory = store.dataset / 'episodes' / episode_id
        on_disk = {p.name: p.stat().st_size for p in directory.iterdir()}
        self.assertEqual(on_disk, sizes[winner])
        self.assertEqual(store.state(episode_id)['files'], on_disk)
        self.assertEqual([r['episode_id'] for r in store.audit()[0]], [episode_id])
        parallel.prepare(self.config, self.output)
        self.assertEqual(list((store.work / 'commits').iterdir()), [])
        with patch.object(HifiUmi, 'episode', side_effect=AssertionError('committed')):
            self.assertEqual(parallel.worker(self.output), 0)
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 1)

    def test_commit_that_fails_under_the_lock_releases_it_and_is_retried_next_round(self):
        for step in ('record', 'rename'):
            with self.subTest(step=step):
                output = self.root / step
                store = parallel.prepare(self.config, output)
                episode_id = store.ids[0]
                atomic_json, rename = parallel.atomic_json, Path.rename
                full = OSError(errno.ENOSPC, 'No space left on device')

                def record(path, value):
                    if step == 'record' and path.parent.name == 'records':
                        self.assertEqual(os.listdir(store.work / 'commits'), [episode_id])
                        raise full
                    atomic_json(path, value)

                def move(path, target):
                    if step == 'rename' and Path(target).parent == store.dataset / 'episodes':
                        self.assertEqual(os.listdir(store.work / 'commits'), [episode_id])
                        raise full
                    return rename(path, target)
                log = []
                with patch.object(parallel, 'atomic_json', record), patch.object(Path, 'rename', move):
                    self.assertEqual(parallel.worker(output, report=log.append), 1)
                self.assertEqual(os.listdir(store.work / 'commits'), [])
                self.assertTrue(any(f'Episode {episode_id} failed: OSError' in m for m in log))
                self.assertIn('No space left on device', (store.work / 'errors' / f'{episode_id}.err').read_text())
                self.assertEqual(os.listdir(store.dataset / 'episodes'), [])
                self.assertEqual(os.listdir(store.work / 'records'), [f'{episode_id}.json'] * (step == 'rename'))
                self.assertIsNone(store.state(episode_id))
                self.assertEqual(os.listdir(store.work / 'claims'), [])
                # An ordinary failure: left alone for the rest of the round, converted in the next.
                with patch.object(HifiUmi, 'episode', side_effect=AssertionError('one attempt per round')):
                    self.assertEqual(parallel.worker(output), 0)
                parallel.prepare(self.config, output)
                self.assertEqual(parallel.worker(output), 0)
                self.assertEqual(os.listdir(store.work / 'errors'), [])
                self.assertEqual(os.listdir(store.work / 'commits'), [episode_id])
                parallel.finalize(output)
                self.assertEqual(len(manifest(output)), 1)

    def test_commit_lock_of_a_worker_killed_inside_commit_waits_for_the_next_prepare(self):
        for killed in ('before_record', 'after_record'):
            with self.subTest(killed=killed):
                output = self.root / killed
                store = parallel.prepare(self.config, output)
                episode_id = store.ids[0]
                # Die inside commit, as SIGKILL would: no handler runs, lock and claim stay behind.
                atomic_json = parallel.atomic_json

                class Killed(BaseException):
                    pass

                def record(path, value):
                    if path.parent.name == 'records':
                        if killed == 'after_record':
                            atomic_json(path, value)
                        kept = {directory: {p.name: p.read_bytes() for p in (store.work / directory).iterdir()}
                                for directory in ('commits', 'claims')}
                        raise Killed(kept)
                    atomic_json(path, value)
                with patch.object(parallel, 'atomic_json', record), self.assertRaises(Killed) as death:
                    parallel.worker(output)
                for directory, files in death.exception.args[0].items():
                    for name, content in files.items():
                        (store.work / directory / name).write_bytes(content)
                self.assertEqual(os.listdir(store.work / 'commits'), [episode_id])
                dead = store.work / 'claims' / '0'
                self.assertTrue(dead.is_file())
                age(dead, claim_files.STALE_S + 30)
                # A rescuer of the same round converts the task but may not commit: it cannot know
                # how far the dead worker got. That is not a failure of the episode.
                log = []
                with patch('sys.stdout', new_callable=io.StringIO) as out:
                    self.assertEqual(parallel.worker(output, report=log.append), 0)
                self.assertNotIn('Committed', out.getvalue())
                self.assertTrue(any(f'Task 0: finished by another worker: {episode_id}: being committed' in m
                                    for m in log))
                self.assertEqual(os.listdir(store.work / 'errors'), [])
                self.assertIsNone(store.state(episode_id))
                self.assertEqual(os.listdir(store.dataset / 'episodes'), [])
                self.assertEqual([n for n in os.listdir(store.work / 'claims') if not n.startswith('.takeover-')], [])
                with self.assertRaisesRegex(parallel.StateError, '1 unfinished episodes'):
                    parallel.finalize(output)
                self.assertFalse(output.exists())
                # The next prepare runs when no worker is left, so the lock is certainly dead.
                with patch('sys.stdout', new_callable=io.StringIO) as out:
                    parallel.prepare(self.config, output)
                self.assertIn('0 exported, 0 skipped, 0 failed, 1 unfinished', out.getvalue())
                self.assertEqual(os.listdir(store.work / 'commits'), [])
                self.assertEqual(os.listdir(store.work / 'claims'), [])
                self.assertEqual(parallel.worker(output), 0)
                self.assertIsNotNone(store.state(episode_id))
                parallel.finalize(output)
                self.assertEqual(len(manifest(output)), 1)

    def test_failure_is_attempted_once_per_round_and_retried_after_prepare(self):
        store = self.prepare(3)
        bad = store.ids[1]
        broken = [True]

        def fail(episode_id):
            if episode_id == bad and broken[0]:
                raise ValueError('injected conversion error')
        calls = self.counted(fail)
        attempts = lambda: [name for name, _ in calls].count(bad)
        # Concurrent workers: one of them meets the failure and reports it.
        self.assertEqual(sorted(self.finish(*[self.start(rank, 3) for rank in range(3)])), [0, 0, 1])
        self.assertEqual(attempts(), 1)
        self.assertEqual(sorted(name for name, _ in calls), sorted(store.ids))
        self.assertEqual(self.errors(), [f'{bad}.err'])
        # Workers arriving later in the same round leave it alone and have nothing to report.
        for rank in range(2):
            self.assertEqual(parallel.worker(self.output, rank, 2), 0)
        self.assertEqual(len(calls), 3)
        # The next prepare starts a new round; a failure that persists is again tried once.
        parallel.prepare(self.config, self.output)
        self.assertEqual(sorted(self.finish(self.start(0, 2), self.start(1, 2))), [0, 1])
        self.assertEqual((attempts(), len(calls)), (2, 4))
        self.assertEqual(parallel.worker(self.output), 0)
        self.assertEqual(attempts(), 2)
        broken[0] = False
        self.assertEqual(parallel.worker(self.output), 0)        # still the same round
        self.assertEqual(attempts(), 2)
        parallel.prepare(self.config, self.output)
        self.assertEqual(parallel.worker(self.output), 0)
        self.assertEqual((attempts(), len(calls)), (3, 5))
        self.assertEqual(self.errors(), [])
        parallel.finalize(self.output)
        self.assertEqual(len(manifest(self.output)), 3)

    def test_prepare_clears_leftover_claims(self):
        store = self.prepare(2)
        dead = Claims(self.claims)
        self.assertTrue(dead.acquire(1))
        dead.now()
        (self.claims / '0').write_text('gone-node-1-00000000')
        age(self.claims / '0', claim_files.STALE_S + 30)
        self.assertTrue(Claims(self.claims).take_over(0))
        (self.claims / '.claim-1-gone-node-1-00000000').write_text('gone-node-1-00000000')
        self.assertEqual({p.name.split('-')[0] for p in self.claims.iterdir()},
                         {'.claim', '.clock', '.takeover', '0', '1'})
        (store.work / 'commits' / store.ids[1]).touch()
        before = (store.work / 'round').stat().st_mtime_ns
        parallel.prepare(self.config, self.output)
        self.assertEqual(list(self.claims.iterdir()), [])
        self.assertEqual(list((store.work / 'commits').iterdir()), [])
        self.assertGreater((store.work / 'round').stat().st_mtime_ns, before)
        # With the dead worker's claim gone a new worker does not wait for it to go stale.
        with patch.object(parallel.time, 'sleep', side_effect=AssertionError('nothing to wait for')):
            self.assertEqual(parallel.worker(self.output), 0)
        self.assertEqual(len(store.audit()[0]), 2)

    def test_worker_requires_a_round_started_by_prepare(self):
        store = self.prepare(1)
        (store.work / 'round').unlink()
        with self.assertRaisesRegex(parallel.StateError, 'Run prepare'):
            parallel.worker(self.output)
        self.assertEqual(list(self.claims.iterdir()), [])

    def status(self):
        with patch('sys.stdout', new_callable=io.StringIO) as out:
            self.assertEqual(parallel.main(['status', '--output-dir', str(self.output)]), 0)
        return out.getvalue().splitlines()

    def test_status_counts_records_failures_and_claims(self):
        store = self.prepare(4)
        self.assertEqual(self.status(), ['0/4 episodes recorded (0.0%), 0 failed, 4 not done; 0 tasks in progress'])

        def fail(episode_id):
            if episode_id == store.ids[2]:
                raise ValueError('injected conversion error')
        self.counted(fail)
        # Two workers on one node and one on another hold tasks; one of them stopped beating.
        (self.claims / '0').write_text('node-a-17-0123abcd')
        (self.claims / '1').write_text('node-a-18-4567cdef')
        (self.claims / '3').write_text('node-b-17-89ab0123')
        with patch.object(claim_files, 'POLL_S', 0), patch.object(Claims, 'take_over', return_value=False), \
                patch.object(parallel.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                parallel.worker(self.output)    # fails the one free task, then would wait for the others
        age(self.claims / '3', claim_files.STALE_S + 30)
        self.assertEqual(self.status(), [
            '0/4 episodes recorded (0.0%), 1 failed, 3 not done; 3 tasks in progress',
            '  node-a: 2 task(s)', '  node-b: 1 task(s)',
            f'  1 claim(s) without a heartbeat for {claim_files.STALE_S}s; a running worker will take them over'])
        self.assertEqual(sorted(p.name for p in self.claims.iterdir()), ['0', '1', '3'])  # status leaves no trace
        for name in '01':
            (self.claims / name).unlink()   # those workers finished without converting; the third one died
        self.assertEqual(parallel.worker(self.output), 0)
        self.assertEqual(self.status(), ['3/4 episodes recorded (75.0%), 1 failed, 0 not done; 0 tasks in progress'])
        parallel.finalize(self.output)
        self.assertEqual(self.status(), [f'Published: {self.output.resolve()}'])

    def test_cli_processes_auto_uses_the_cpu_limit_and_reports_any_failure(self):
        self.prepare(1)
        arguments = ['worker', '--output-dir', str(self.output), '--processes']
        with patch.object(parallel, 'available_cpus', return_value=3) as cpus, \
                patch.object(parallel, 'run_workers', return_value=[0, 0, 0]) as spawn:
            self.assertEqual(parallel.main([*arguments, 'auto']), 0)
            (command, count), = [call.args for call in spawn.call_args_list]
            self.assertEqual(count, 3)
            self.assertEqual(command[1:], ['-m', 'loom_data', 'convert', 'worker', '--output-dir',
                                           str(self.output.resolve())])
            spawn.return_value = [0, 1, 0]
            self.assertEqual(parallel.main([*arguments, '3']), 1)
            cpus.assert_called_once()
            spawn.reset_mock()
            with patch('sys.stderr', new_callable=io.StringIO):
                for bad in ('0', '-2', 'many'):
                    self.assertEqual(parallel.main([*arguments, bad]), 1)
            spawn.assert_not_called()
        self.assertEqual(list((self.store.work / 'logs').iterdir()), [])  # the launcher itself is not a worker

    def test_cli_processes_runs_real_local_workers(self):
        store = self.prepare(5)
        result = subprocess.run(
            [sys.executable, '-m', 'loom_data', 'convert', 'worker', '--output-dir', str(self.output), '--processes', '2'],
            cwd=paths.ROOT, capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(sorted(line for line in result.stdout.splitlines() if line.startswith('Committed')),
                         [f'Committed {episode_id}' for episode_id in store.ids])
        logs = [path.read_text() for path in (store.work / 'logs').glob('worker-*.err')]
        self.assertEqual(sorted(text[text.index('Worker ') + 7:][:3] for text in logs), ['0/2', '1/2'])
        owners = {line.split('Claim owner: ')[1] for text in logs for line in text.splitlines() if 'Claim owner: ' in line}
        self.assertEqual(len(owners), 2)
        self.assertTrue(all('Worker exit code: 0' in text for text in logs))
        self.assertEqual(list(self.claims.iterdir()), [])
        parallel.finalize(self.output)
        self.assertEqual([r['episode_id'] for r in manifest(self.output)], store.ids)


class AvailableCpuTests(unittest.TestCase):
    def cpus(self, limit):
        def read(path, *args, **kwargs):
            self.assertEqual(str(path), '/sys/fs/cgroup/cpu.max')
            if isinstance(limit, Exception):
                raise limit
            return limit
        with patch.object(Path, 'read_text', read), \
                patch.object(os, 'sched_getaffinity', return_value=set(range(96)), create=True):
            return resumable.available_cpus()

    def test_cgroup_limit_wins_over_the_cores_of_the_host(self):
        for limit, expected in (('800000 100000\n', 8), ('250000 100000\n', 2), ('100000 100000\n', 1),
                                ('50000 100000\n', 1),          # a fraction of a CPU still runs one worker
                                ('max 100000\n', 96),           # unlimited: the cores this process may use
                                (FileNotFoundError(), 96),      # no cgroup v2
                                ('', 96), ('garbage here\n', 96)):
            with self.subTest(limit=limit):
                self.assertEqual(self.cpus(limit), expected)

    def test_hosts_without_cpu_affinity_fall_back_to_the_core_count(self):
        affinity = getattr(os, 'sched_getaffinity', None)
        if affinity:
            del os.sched_getaffinity
            self.addCleanup(setattr, os, 'sched_getaffinity', affinity)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError), \
                patch.object(os, 'cpu_count', return_value=12) as count:
            self.assertEqual(resumable.available_cpus(), 12)
            count.return_value = None
            self.assertEqual(resumable.available_cpus(), 1)

    def test_worker_rank_defaults_to_a_lone_worker_outside_slurm(self):
        with patch.dict(os.environ):
            for key in ('SLURM_PROCID', 'SLURM_NTASKS'):
                os.environ.pop(key, None)
            self.assertEqual(resumable.worker_rank(None, None), (0, 1))
            self.assertEqual(resumable.worker_rank(2, 5), (2, 5))
            os.environ.update(SLURM_PROCID='3', SLURM_NTASKS='4')
            self.assertEqual(resumable.worker_rank(None, None), (3, 4))
            self.assertEqual(resumable.worker_rank(1, None), (1, 4))


if __name__ == '__main__':
    unittest.main()
