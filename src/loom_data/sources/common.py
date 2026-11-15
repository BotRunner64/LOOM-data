"""Small value mappings shared by several sources."""
import math


def normalized_gripper(value, closed, opened):
    if not all(math.isfinite(v) for v in (value, closed, opened)) or closed == opened:
        raise ValueError('Nonfinite gripper value or invalid open/closed range')
    return min(1., max(0., (float(value) - closed) / (opened - closed)))


def task_instructions(task_indices, tasks, times):
    if not times or times[-1] <= times[0]:
        raise ValueError('Empty or zero-duration episode')
    transitions = {}
    previous = None
    for index, timestamp in zip(task_indices, times):
        if type(index) is not int or index not in tasks:
            raise ValueError(f'Unknown task_index: {index!r}')
        text = tasks[index]
        if text != previous:
            transitions[timestamp] = text
            previous = text
    starts = sorted(transitions)
    return [{'start_ns': start - times[0], 'end_ns': end - times[0],
             'text': transitions[start]}
            for start, end in zip(starts, starts[1:] + [times[-1]]) if end > start]
