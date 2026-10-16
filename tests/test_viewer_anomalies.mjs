import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../scripts/viewer/anomalies.js', import.meta.url), 'utf8');
const {containsTime, filterRecords} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('intervals exclude their end; points include exactly their timestamp', () => {
  const interval = {start_ns: 10, end_ns: 20};
  assert.equal(containsTime(interval, 9), false);
  assert.equal(containsTime(interval, 10), true);
  assert.equal(containsTime(interval, 19), true);
  assert.equal(containsTime(interval, 20), false);
  const point = {start_ns: 20, end_ns: 20};
  assert.equal(containsTime(point, 19), false);
  assert.equal(containsTime(point, 20), true);
  assert.equal(containsTime(point, 21), false);
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
