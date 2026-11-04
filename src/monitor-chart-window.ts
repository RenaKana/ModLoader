import { seriesSegments, type HistoryPoint } from './resource-metrics'

type DrawPoint = { time: number; value: number }

/** Keep real in-window samples and one separate real point needed to clip the left edge. */
export function retainChartWindow<T extends HistoryPoint>(samples: readonly T[], start: number) {
  const points: T[] = []
  let boundaryPoint: T | undefined
  for (const point of samples) {
    if (point.time < start) boundaryPoint = point
    else points.push(point)
  }
  return { points, boundaryPoint }
}

/** Clip existing continuous line segments, without inventing samples or bridging missing data. */
export function windowSegments(points: readonly HistoryPoint[], start: number, end: number): DrawPoint[][] {
  const at = (before: DrawPoint, after: DrawPoint, time: number): DrawPoint => ({ time,
    value: before.value + (after.value - before.value) * (time - before.time) / (after.time - before.time) })
  const result: DrawPoint[][] = []
  for (const segment of seriesSegments([...points], 'current')) {
    if (segment.at(-1)!.time < start || segment[0].time > end) continue
    const visible = segment.filter(point => point.time >= start && point.time <= end)
    const first = segment.findIndex(point => point.time >= start)
    if (first > 0 && segment[first].time > start) visible.unshift(at(segment[first - 1], segment[first], start))
    const after = segment.findIndex(point => point.time > end)
    if (after > 0 && segment[after - 1].time < end) visible.push(at(segment[after - 1], segment[after], end))
    if (visible.length) result.push(visible)
  }
  return result
}
