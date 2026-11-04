import { expect, test } from '@playwright/test'
import { usageSlots, nearestUsageSlot, usageHitRange, usagePaths, nextUsageBoundary, usagePlotValue, usageValue } from '../src/monitor-usage'
import type { RequestHistoryUsageBucket } from '../src/types'

test('用量桶固定等距槽：最后真实桶在最右，单桶居中，命中区由相邻中点划分', () => {
  const slots = usageSlots(5, 64, 664)
  expect(slots).toEqual([64, 214, 364, 514, 664])
  expect(usageSlots(1, 64, 664)).toEqual([364])
  expect(usageSlots(0, 64, 664)).toEqual([])
  expect(usageHitRange(slots, 0, 64, 664)).toEqual([64, 139])
  expect(usageHitRange(slots, 2, 64, 664)).toEqual([289, 439])
  expect(usageHitRange(slots, 4, 64, 664)).toEqual([589, 664])
  expect(nearestUsageSlot(140, 5, 64, 664)).toBe(1)
  expect(nearestUsageSlot(663, 5, 64, 664)).toBe(4)
  expect(nearestUsageSlot(-20, 1, 64, 664)).toBe(0)
})

test('已知桶连续连接但不补统计、不加end尾点；少于两个已知点不伪造线', () => {
  const points = [{ x: 64, value: 10 }, { x: 214, value: null }, { x: 364, value: 20 }, { x: 514, value: 5 }, { x: 664, value: 30 }]
  const saved = structuredClone(points)
  const paths = usagePaths(points, value => 190 - value, 190)
  expect(paths.line).toMatch(/^M64,180/)
  expect(paths.line).toMatch(/664,160$/)
  expect(paths.line).not.toContain('214,')
  expect(paths.area).toContain('664,190')
  expect(points).toEqual(saved)
  expect(usagePaths([{ x: 364, value: 10 }], value => value, 190)).toEqual({ line: '', area: '' })
  expect(usagePaths([{ x: 64, value: null }, { x: 664, value: 10 }], value => value, 190)).toEqual({ line: '', area: '' })
})

test('开始新活动只预约下个真实桶边界，不用过期next_refresh_at立即刷新', () => {
  expect(nextUsageBoundary(1800, 300, 1780)).toBe(1800)
  expect(nextUsageBoundary(1800, 300, 1800)).toBe(2100)
  expect(nextUsageBoundary(1800, 300, 2500)).toBe(2700)
})

test('空桶命中率仅绘图归零，真实ratio与有请求缺测不变，单桶仍无静态path', () => {
  const empty = { request_count: 0, cache_ratio: null, total: { value: 0, reported: 0 } } as RequestHistoryUsageBucket
  const missing = { ...empty, request_count: 2 }
  const known = { ...missing, cache_ratio: .5 }
  const buckets = [known, empty, missing, known, empty], saved = structuredClone(buckets)
  expect(buckets.map(bucket => usagePlotValue(bucket, 'hit'))).toEqual([50, 0, null, 50, 0])
  expect(buckets.map(bucket => usageValue(bucket, 'hit'))).toEqual([50, null, null, 50, null])
  expect(usagePlotValue(empty, 'total')).toBe(0)
  const slots = usageSlots(buckets.length, 64, 664)
  const paths = usagePaths(buckets.map((bucket, index) => ({ x: slots[index], value: usagePlotValue(bucket, 'hit') })), value => 190 - value, 190)
  expect(paths.line).toContain('214,190'); expect(paths.line).toMatch(/664,190$/)
  expect(buckets).toEqual(saved)
  expect(usagePaths([{ x: 364, value: usagePlotValue(empty, 'hit') }], value => 190 - value, 190)).toEqual({ line: '', area: '' })
})
