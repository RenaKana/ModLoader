import type { Message, ToolCall } from './chat-history'
import { toolMetricLabel } from './chat-turn-metrics'

export type ChatToolStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type ChatActivityEvent =
  | { id: string; kind: 'reasoning'; text: string }
  | { id: string; kind: 'text'; text: string }
  | { id: string; kind: 'tool'; call: ToolCall; status: ChatToolStatus; result?: string; requestId?: string; startedAt?: number; endedAt?: number; elapsed?: number }

const toolStatuses = new Set<ChatToolStatus>(['pending', 'running', 'completed', 'failed', 'cancelled'])
function object(value: unknown): value is Record<string, unknown> { return value !== null && typeof value === 'object' && !Array.isArray(value) }
function parsed(value: string | undefined): unknown {
  if (value === undefined) return undefined
  try { return JSON.parse(value) } catch { return value }
}
function string(value: unknown) { return typeof value === 'string' ? value : '' }

/** Display history is separate from the final model message and keeps event order. */
export function normalizeChatActivity(value: unknown): ChatActivityEvent[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((event): ChatActivityEvent[] => {
    if (!object(event) || typeof event.id !== 'string' || !event.id) return []
    if ((event.kind === 'reasoning' || event.kind === 'text') && typeof event.text === 'string') return [{ id: event.id, kind: event.kind, text: event.text }]
    if (event.kind !== 'tool' || !toolStatuses.has(event.status as ChatToolStatus) || !object(event.call) || !object(event.call.function)) return []
    const fn = event.call.function
    if (typeof fn.name !== 'string' || typeof fn.arguments !== 'string') return []
    return [{ id: event.id, kind: 'tool', status: event.status as ChatToolStatus,
      call: { ...(typeof event.call.id === 'string' ? { id: event.call.id } : {}), ...(typeof event.call.type === 'string' ? { type: event.call.type } : {}), function: { name: fn.name, arguments: fn.arguments } },
      ...(typeof event.result === 'string' ? { result: event.result } : {}),
      ...(typeof event.requestId === 'string' ? { requestId: event.requestId } : {}),
      ...Object.fromEntries(['startedAt', 'endedAt', 'elapsed'].flatMap(key => typeof event[key] === 'number' && Number.isFinite(event[key]) && event[key] >= 0 ? [[key, event[key]]] : [])),
    }]
  })
}

export function toolResultStatus(result: string): ChatToolStatus {
  const value = parsed(result)
  if (object(value)) {
    if (value.written === false || value.cancelled === true) return 'cancelled'
    if (value.error) return 'failed'
  }
  return 'completed'
}

export function messageActivity(message: Message): ChatActivityEvent[] {
  if (message.activity?.length) return message.activity
  // Old records did not capture positions between reasoning and tool calls.
  const events: ChatActivityEvent[] = []
  const reasoning = [message.reasoningHistory, message.reasoning].filter(Boolean).join('\n\n')
  if (reasoning) events.push({ id: 'legacy-reasoning', kind: 'reasoning', text: reasoning })
  for (const [index, result] of (message.toolResults || []).entries()) {
    events.push({ id: `legacy-result-${index}`, kind: 'tool', call: result.call, status: toolResultStatus(result.content), result: result.content })
  }
  if (message.tools?.length && message.text) events.push({ id: 'legacy-tool-text', kind: 'text', text: message.text })
  for (const [index, call] of (message.tools || []).entries()) {
    const executed = message.toolResults?.some(result => result.call.id === call.id && result.call.function.name === call.function.name && result.call.function.arguments === call.function.arguments)
    const status = message.status === 'failed' ? 'failed' : message.status === 'stopped' || message.status === 'interrupted' ? 'cancelled' : 'pending'
    if (!executed) events.push({ id: `legacy-request-${index}`, kind: 'tool', call, status })
  }
  return events
}

export type ToolActivityDescription = { action: string; target: string; outcome: string; error: string; status: ChatToolStatus }

function resultError(value: unknown) {
  if (!object(value)) return ''
  if (typeof value.error === 'string') return value.error
  return object(value.error) ? string(value.error.message) : ''
}

function lineRange(value: Record<string, unknown>) {
  if (typeof value.start_line !== 'number' || typeof value.end_line !== 'number') return ''
  if (value.end_line < value.start_line) return '空内容'
  return value.start_line === value.end_line ? `第 ${value.start_line} 行` : `第 ${value.start_line}–${value.end_line} 行`
}

export function describeToolActivity(event: Extract<ChatActivityEvent, { kind: 'tool' }>): ToolActivityDescription {
  const input = parsed(event.call.function.arguments), result = parsed(event.result)
  const args = object(input) ? input : {}, output = object(result) ? result : {}
  const name = event.call.function.name
  const action = toolMetricLabel(name, result) || '工具调用'
  const target = name === 'current_time' ? string(args.time_zone)
    : name === 'reverse_image_search' ? string(output.image_name)
    : name === 'list_directory' ? string(args.path) || '.'
      : ['query', 'path', 'expression', 'url'].map(key => string(args[key])).find(Boolean) || ''
  let status = event.status, outcome = '', error = resultError(result)
  if (event.result !== undefined) {
    const resultStatus = toolResultStatus(event.result)
    if (resultStatus !== 'completed') status = resultStatus
  }
  if (status === 'cancelled') {
    outcome = output.written === false ? error.includes('用户拒绝') ? '已拒绝' : '未写入' : '已取消'
    if (error.includes('用户拒绝')) error = ''
  } else if (status === 'failed') {
    if (!error && typeof result === 'string') error = result
    outcome = error ? '' : '失败'
  } else if (status === 'completed') {
    if (name === 'search_web' && Array.isArray(output.sources)) outcome = `${output.sources.length} 条结果`
    else if (name === 'search_images' && Array.isArray(output.images)) outcome = `${output.images.length} 张图片`
    else if (name === 'reverse_image_search' && Array.isArray(output.matches)) outcome = `${output.matches.length} 条匹配`
    else if (name === 'list_directory' && Array.isArray(output.entries)) outcome = `${output.entries.length} 项${typeof output.total_entries === 'number' && output.total_entries !== output.entries.length ? ` · 共 ${output.total_entries} 项` : ''}`
    else if (name === 'read_file' || name === 'read_page') outcome = output.kind === 'summary' ? '阅读摘要' : lineRange(output)
    else if (name === 'write_file' && output.written === true) outcome = '已写入'
    else if (name === 'calculate' && (typeof output.result === 'number' || typeof output.result === 'string')) outcome = `= ${output.result}`
    else if (name === 'current_time') outcome = string(output.local) || string(output.utc)
    if (object(output.routing) && Array.isArray(output.routing.attempts) && output.routing.attempts.length > 1) {
      const route = output.routing.attempts.filter(object).map(attempt => string(attempt.label)).filter(Boolean).join(' → ')
      if (route) outcome = [outcome, route].filter(Boolean).join(' · ')
    }
  }
  return { action, target, outcome, error, status }
}

export function formatToolDetail(value: string) {
  try { return JSON.stringify(JSON.parse(value), null, 2) } catch { return value }
}
