import type { RequestHistoryDates } from './types'

export function historyBackend(instanceId: string | null | undefined, backend: string | undefined,
  instances: readonly { id: string; backend: string }[], catalog: RequestHistoryDates | null) {
  return backend || (instanceId ? instances.find(instance => instance.id === instanceId && instance.backend)?.backend
    || catalog?.dates.flatMap(day => day.instances).find(instance => instance.id === instanceId && instance.backend)?.backend : '') || ''
}

export type HistoryDateMode = 'today' | 'yesterday' | 'date' | 'range'
export const localDate = (date = new Date()) => `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
export function previousDate(today: string) {
  const date = new Date(`${today}T12:00:00`)
  date.setDate(date.getDate() - 1)
  return localDate(date)
}
export function historyDateRange(mode: HistoryDateMode, today: string, from: string, to: string) {
  const start = mode === 'today' ? today : mode === 'yesterday' ? previousDate(today) : from
  const end = mode === 'range' ? to : start
  return { from: start, to: end, error: !start || !end ? '请选择日期' : start > end ? '结束日期不能早于开始日期' : '' }
}
export type HistoryInstance = RequestHistoryDates['dates'][number]['instances'][number]
export const historyInstanceIdentity = (instance: Pick<HistoryInstance, 'run_id' | 'id' | 'source' | 'conversation_id'>) => instance.source === 'internal'
  ? JSON.stringify(['internal', instance.id, instance.id ? null : instance.conversation_id]) : JSON.stringify([instance.run_id, instance.id])
export function historyInstanceMatches(filter: HistoryInstance | null, candidate: HistoryInstance | undefined) {
  return !filter || !!candidate && historyInstanceIdentity(filter) === historyInstanceIdentity(candidate)
}
export function historyInstances(catalog: RequestHistoryDates | null, from: string, to: string) {
  const instances = new Map<string, HistoryInstance & { date: string }>()
  for (const day of catalog?.dates ?? []) {
    if (day.date < from || day.date > to) continue
    for (const instance of day.instances) {
      const key = historyInstanceIdentity(instance)
      if (!instances.has(key)) instances.set(key, { ...instance, date: day.date })
    }
  }
  return [...instances.values()].map(instance => {
    const duplicate = [...instances.values()].some(other => other !== instance && other.alias === instance.alias && other.backend === instance.backend)
    const run = instance.started_at != null ? new Date(instance.started_at * 1000).toLocaleString('zh-CN', { hour12: false }) : `${instance.date} · ${(instance.run_id ?? instance.conversation_id ?? '').slice(-8)}`
    return { ...instance, value: historyInstanceIdentity(instance), label: `${instance.alias} · ${instance.backend ? `${instance.backend} · ` : ''}${instance.source === 'internal' ? '内置' : '外部'}${duplicate ? ` · ${run}` : ''}` }
  })
}

