"""Explicit source-data rejections; operational/programming errors remain failures."""
from __future__ import annotations

import numpy as np


class SourceDataError(ValueError):
    def __init__(self, reason, detail, *, reasons=None):
        super().__init__(detail)
        self.reason = reason
        self.reasons = reasons or [reason]


def source_value(reason, function, *args, **kwargs):
    """Classify only a known source-value validation boundary."""
    try:
        return function(*args, **kwargs)
    except SourceDataError:
        raise
    except ValueError as exc:
        raise SourceDataError(reason, str(exc)) from exc


def stationary_base(position, orientation, commands):
    """Recover zero-pose markers under the same rule as AgiBot World 2026.

    Call only after validating the position/orientation shapes and finite values.
    Measured poses are retained. No moving-base interpolation is performed.
    """
    missing = np.all(position == 0, axis=1) & np.all(orientation == 0, axis=1)
    if not missing.any():
        return position, orientation, []
    if (commands is None or commands.shape != (len(position), 2) or commands.dtype.kind not in 'fiub'
            or not np.isfinite(commands).all()):
        raise SourceDataError('unverified_base_pose', 'Missing base pose without complete finite base velocity commands')
    if np.any(np.abs(commands) > 1e-6):
        raise SourceDataError('moving_base_without_pose', 'Missing base pose with nonzero base velocity commands')
    position, orientation = position.copy(), orientation.copy()
    if missing.all():
        orientation[:] = (0., 0., 0., 1.)
        return position, orientation, ['stationary_base_identity']
    reference = int(np.flatnonzero(~missing)[0])
    for index, absent in enumerate(missing):
        if absent:
            position[index], orientation[index] = position[reference], orientation[reference]
        else:
            reference = index
    return position, orientation, ['stationary_base_fill']
