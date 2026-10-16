/* Timestamp navigation shared by asynchronous video and state streams. */

export function sampleIndex(times, target) {
  let low = 0, high = times.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (times[middle] <= target) low = middle + 1; else high = middle;
  }
  return low - 1;
}

export function adjacentSampleTime(times, current, direction) {
  if (!times.length) return current;
  const currentIndex = sampleIndex(times, current);
  if (currentIndex < 0 && direction < 0) return current;
  const targetIndex = direction < 0
    ? Math.max(0, currentIndex - 1)
    : Math.min(times.length - 1, currentIndex + 1);
  return times[targetIndex];
}
