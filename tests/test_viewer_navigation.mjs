import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../scripts/viewer/navigation.js', import.meta.url), 'utf8');
const {adjacentSampleTime, sampleIndex} = await import(
  'data:text/javascript;base64,' + Buffer.from(source).toString('base64')
);

test('sample lookup uses the latest native timestamp at or before the target', () => {
  const times = [10, 30, 31, 90];
  assert.equal(sampleIndex(times, 9), -1);
  assert.equal(sampleIndex(times, 10), 0);
  assert.equal(sampleIndex(times, 30), 1);
  assert.equal(sampleIndex(times, 89), 2);
  assert.equal(sampleIndex(times, 100), 3);
});

test('arrow stepping follows only the focused asynchronous stream', () => {
  const cameraTimes = [10, 30, 90];
  const stateTimes = [5, 20, 21, 70, 120];
  assert.equal(adjacentSampleTime(cameraTimes, 30, 1), 90);
  assert.equal(adjacentSampleTime(cameraTimes, 30, -1), 10);
  assert.equal(adjacentSampleTime(stateTimes, 30, 1), 70);
  assert.equal(adjacentSampleTime(stateTimes, 30, -1), 20);
  assert.equal(adjacentSampleTime(cameraTimes, 0, 1), 10);
  assert.equal(adjacentSampleTime(cameraTimes, 0, -1), 0);
  assert.equal(adjacentSampleTime(cameraTimes, 100, 1), 90);
});
