import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../scripts/viewer/state_curves.js', import.meta.url), 'utf8');
const {buildCurveTraces} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

test('curve traces contain only the exported EEF and gripper streams', () => {
  const pose = [[1, 2, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1], [4, 5, 6, 0, 1, 0, 1, 0, 0, 0, 0, -1]];
  const streams = {
    left_eef: {times: [0, 10], values: pose},
    right_eef: {times: [0, 10], values: pose},
    left_gripper: {times: [0, 10], values: [0.1, 0.8]},
    right_gripper: {times: [0, 10], values: [0.2, 0.9]},
    ignored_joint_state: {times: [0], values: [[42]]},
  };
  const traces = buildCurveTraces(streams);
  assert.equal(traces.length, 20);
  assert.deepEqual(new Set(traces.map(trace => trace.streamName)), new Set([
    'left_eef', 'right_eef', 'left_gripper', 'right_gripper',
  ]));
  assert.equal(traces.filter(trace => trace.enabled).length, 8);
  assert.deepEqual(traces.filter(trace => trace.enabled).map(trace => trace.component),
    ['x', 'y', 'z', 'x', 'y', 'z', 'opening', 'opening']);
  assert.deepEqual(traces.find(trace => trace.id === 'left_eef:x').values, [1, 4]);
  assert.deepEqual(traces.find(trace => trace.id === 'right_gripper:opening').values, [0.2, 0.9]);
});

test('curve trace ranges ignore non-finite values', () => {
  const pose = [[Number.NaN, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]];
  const traces = buildCurveTraces({
    left_eef: {times: [0, 1], values: pose},
    left_gripper: {times: [0, 1], values: [Number.NaN, Number.NaN]},
  });
  assert.deepEqual(
    [traces.find(trace => trace.id === 'left_eef:x').min,
      traces.find(trace => trace.id === 'left_eef:x').max],
    [2, 2],
  );
  assert.deepEqual(
    [traces.find(trace => trace.id === 'left_gripper:opening').min,
      traces.find(trace => trace.id === 'left_gripper:opening').max],
    [0, 0],
  );
});
