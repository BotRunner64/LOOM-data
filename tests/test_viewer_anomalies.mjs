import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../src/loom_data/viewer/anomalies.js', import.meta.url), 'utf8');
const {containsTime, filterRecords, frameNs} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('frame ranges cover their frames and exclude their end', () => {
  // Frame times as the server sends them: k / 30 s rounded to whole nanoseconds.
  assert.deepEqual([0, 1, 2, 3, 30].map(frameNs), [0, 33333333, 66666667, 100000000, 1000000000]);
  const interval = {start_frame: 10, end_frame: 12};
  assert.equal(containsTime(interval, frameNs(9)), false);
  assert.equal(containsTime(interval, frameNs(10)), true);
  assert.equal(containsTime(interval, frameNs(11)), true);
  assert.equal(containsTime(interval, frameNs(12)), false);
});

test('filters compose without interpreting reasons, reordering or deduplicating', () => {
  const a = {episode_id: 'a', stream: 'left_eef', reason: '<future rule>'};
  const b = {episode_id: 'b', stream: 'right_gripper', reason: 'custom'};
  const records = [b, a, {...a}];
  assert.deepEqual(filterRecords(records, {}), records);
  assert.deepEqual(filterRecords(records, {reason: '<future rule>'}), [a, a]);
  assert.deepEqual(filterRecords(records, {episode: 'b', stream: 'right_gripper', reason: 'custom'}), [b]);
  assert.deepEqual(filterRecords(records, {episode: 'a', stream: 'right_gripper'}), []);
});

test('records without a stream are filtered as the whole episode', () => {
  const idle = {episode_id: 'a', stream: null, reason: 'boundary_idle'};
  const spike = {episode_id: 'a', stream: 'left_eef', reason: 'eef_spike'};
  assert.deepEqual(filterRecords([idle, spike], {stream: 'episode'}), [idle]);
  assert.deepEqual(filterRecords([idle, spike], {stream: 'left_eef'}), [spike]);
  assert.deepEqual(filterRecords([idle, spike], {}), [idle, spike]);
});
