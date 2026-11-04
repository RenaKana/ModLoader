import type { RequestHistoryPage } from './types'

export function requireGroupedHistoryPage(page: RequestHistoryPage) {
  if (!page?.summary || typeof page.summary !== 'object' || !Array.isArray(page.groups)
    || !Array.isArray(page.internal_groups) || typeof page.group_total !== 'number') {
    throw new Error('运行监控接口尚未更新，请退出并重新打开开发版')
  }
  return page
}

// Older running gateways ignore the new selector. Read their retained inventory
// once per refresh, rather than treating a grouped representative as a request page.
export async function readMonitorHistoryPage(query: URLSearchParams, read: (query: URLSearchParams) => Promise<RequestHistoryPage>,
  requestMode: { current: boolean | undefined }) {
  const requested = new URLSearchParams(query)
  if (requestMode.current !== false) requested.set('external_group_by', 'request')
  const page = requireGroupedHistoryPage(await read(requested))
  requestMode.current = page.external_group_by === 'request'
  if (requestMode.current || query.get('source') === 'internal') return page
  if (!page.total) return { ...page, external_group_by: 'request' as const }

  const offset = Number(query.get('offset') || 0), limit = Number(query.get('limit') || 50)
  const allQuery = new URLSearchParams(query)
  allQuery.set('offset', '0'); allQuery.set('limit', String(Math.max(1, page.total)))
  const all = page.offset === 0 && page.limit >= page.group_total! ? page : requireGroupedHistoryPage(await read(allQuery))
  const rawQuery = new URLSearchParams(allQuery)
  rawQuery.delete('group_by'); rawQuery.delete('source')
  const external = await read(rawQuery)
  const internal = all.internal_groups ?? []
  const entries = [
    ...external.items.map(record => ({ startedAt: record.started_at ?? 0, record })),
    ...internal.map(group => ({ startedAt: group.started_at, group })),
  ].sort((a, b) => b.startedAt - a.startedAt)
  const selected = entries.slice(offset, offset + limit)
  return { ...all, external_group_by: 'request' as const, offset, limit, groups: [],
    items: selected.flatMap(entry => 'record' in entry ? [entry.record] : []),
    internal_groups: selected.flatMap(entry => 'group' in entry ? [entry.group] : []),
    total: all.total, group_total: entries.length,
    issues: [...new Set([...all.issues, ...external.issues])] }
}
