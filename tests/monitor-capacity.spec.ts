import { expect, test } from '@playwright/test'
import type { ChatGenerationActivity, RuntimeInstance, Telemetry } from '../src/types'
import { generationView, monitorGenerationView, rememberGeneration, reportedContextCapacity } from '../src/monitor-generation'

const instance = { id: 'loaded-model', state: 'ready', active_requests: 1 } as RuntimeInstance
function payload(capacity: number | null, request = false): Telemetry {
  return { instance_id: instance.id, backend: 'ninfer', state: 'ready', updated_at: 100, unavailable: [], groups: [],
    live_generation: { request_key: request ? 'request-five' : null, client_request_id: request ? 'turn-five' : null,
      scope: 'request', source: 'fixture', backend_running: request ? 1 : 0, backend_queued: 0, tokens_per_second: null,
      window_s: null, available_for_chat: false, reason: null, phase: request ? 'thinking' : 'idle', context_capacity: capacity } }
}
function turn(extra: Partial<ChatGenerationActivity> = {}): ChatGenerationActivity {
  return { instanceId: instance.id, messageId: 'turn-five', phase: 'generating', requests: Array.from({ length: 5 }, (_, index) => ({
    id: `call-${index + 1}`, index: index + 1, kind: 'generation', status: index === 4 ? 'running' : 'completed', startedAt: index * 1000,
    labels: [], usage: { prompt_tokens: 57_000, completion_tokens: 851 },
  })), ...extra }
}

test('运行中从聊天选中本轮时保留同一已加载实例先前报告的容量', () => {
  const known = payload(200_000), memory = rememberGeneration(undefined, known)
  // A later request may not report capacity yet; the loaded instance is unchanged.
  const current = payload(null, true), retained = rememberGeneration(memory, current)
  const view = monitorGenerationView(current, retained, instance, false, [], turn())
  expect(view).toMatchObject({ historical: true, status: '正在思考', used: 57_851, capacity: 200_000 })
  expect(view.requests).toHaveLength(5)
  expect(reportedContextCapacity(known)).toBe(200_000) // No request identity is required for instance capacity.
})

test('历史容量优先使用消息快照，原实例已卸载仍可独立查看', () => {
  const current = payload(200_000, true), memory = rememberGeneration(undefined, current)
  const historical = turn({ instanceId: 'unloaded-model', phase: 'completed', contextCapacity: 131_072 })
  const view = monitorGenerationView(current, memory, instance, false, [], historical)
  expect(view).toMatchObject({ capacity: 131_072, status: '已完成', used: 57_851, historical: true })
  expect(monitorGenerationView(null, undefined, undefined, false, [], historical)).toMatchObject({ capacity: 131_072, used: 57_851 })
})

test('旧实例或无实例归属的未知容量不借用同模型当前实例的遥测', () => {
  const current = payload(200_000, true), memory = rememberGeneration(undefined, current)
  for (const instanceId of ['unloaded-model', '']) {
    const view = monitorGenerationView(current, memory, instance, false, [], turn({ instanceId, phase: 'completed' }))
    expect(view.capacity).toBeNull()
    expect(view.requests).toHaveLength(5)
  }
  expect(monitorGenerationView(current, memory, instance, false, [], turn({ contextCapacity: 65_536 })).capacity).toBe(65_536)
})

test('自动/零和无来源不能成为容量，真实槽位容量与已知实例容量仍可用', () => {
  for (const capacity of [null, 0, -1, Number.NaN]) {
    const sample = payload(capacity), memory = rememberGeneration(undefined, sample)
    expect(reportedContextCapacity(sample)).toBeNull()
    expect(generationView(sample, memory, true, 0).capacity).toBeNull()
    expect(monitorGenerationView(sample, memory, instance, false, [], turn()).capacity).toBeNull()
  }
  const slot = { ...payload(0), backend: 'llama_cpp', groups: [{ id: 'llama_slot_0', title: '槽位', metrics: [
    { id: 'n_ctx', label: '容量', value: 8192, kind: 'gauge' as const },
  ] }] }
  expect(reportedContextCapacity(slot)).toBe(8192)
  const known = rememberGeneration(undefined, payload(200_000))
  expect(generationView(payload(0), rememberGeneration(known, payload(0)), true, 0).capacity).toBe(200_000)
})
