import type { ChatActivityEvent } from './chat-activity'
import type { ToolCall } from './chat-history'

const copyCall = (call: ToolCall): ToolCall => ({ ...call, function: { ...call.function } })

/** Records display order only; model request history is assembled separately. */
export function createActivityRecorder() {
  let events: ChatActivityEvent[] = []
  let sequence = 0
  let lastReasoning = '', recordedText = ''
  let passTools = new Map<number, string>()
  let appendableTextId = ''
  let requestId: string | undefined
  function nextId() { return `activity-${++sequence}` }
  function appendText(kind: 'reasoning' | 'text', text: string) {
    if (!text) return
    const last = events.at(-1)
    if (last?.id === appendableTextId && last.kind === kind) events = [...events.slice(0, -1), { ...last, text: last.text + text }]
    else { appendableTextId = nextId(); events = [...events, { id: appendableTextId, kind, text }] }
  }
  function updateTool(id: string, change: Partial<Extract<ChatActivityEvent, { kind: 'tool' }>>) {
    events = events.map(event => event.id === id && event.kind === 'tool' ? { ...event, ...change } : event)
  }
  function findTool(call: ToolCall) {
    return [...events].reverse().find(event => event.kind === 'tool' && event.call.id === call.id && ['pending', 'running'].includes(event.status))
  }
  function toolEvent(call: ToolCall) {
    const existing = findTool(call)
    if (existing) return existing.id
    const id = nextId()
    events = [...events, { id, kind: 'tool', call: copyCall(call), status: 'pending', ...(requestId ? { requestId } : {}) }]
    appendableTextId = ''
    return id
  }
  return {
    startPass(id?: string) { requestId = id; lastReasoning = ''; recordedText = ''; passTools = new Map(); appendableTextId = '' },
    capture(reasoning: string, text: string, calls: ToolCall[]) {
      if (reasoning !== lastReasoning) {
        appendText('reasoning', reasoning.startsWith(lastReasoning) ? reasoning.slice(lastReasoning.length) : reasoning)
        lastReasoning = reasoning
      }
      if (calls.length && text !== recordedText) {
        appendText('text', text.startsWith(recordedText) ? text.slice(recordedText.length) : text)
        recordedText = text
      }
      calls.forEach((call, index) => {
        if (!call) return
        let id = passTools.get(index)
        if (!id) { id = nextId(); passTools.set(index, id); events = [...events, { id, kind: 'tool', call: copyCall(call), status: 'pending', ...(requestId ? { requestId } : {}) }]; appendableTextId = '' }
        else updateTool(id, { call: copyCall(call) })
      })
    },
    startTool(call: ToolCall) { const id = toolEvent(call); updateTool(id, { call: copyCall(call), status: 'running', startedAt: Date.now() }); appendableTextId = '' },
    finishTool(call: ToolCall, result: unknown) {
      const value = result !== null && typeof result === 'object' ? result as Record<string, unknown> : {}
      const status = value.written === false ? 'cancelled' : value.error ? 'failed' : 'completed'
      const id = toolEvent(call), event = events.find(item => item.id === id), endedAt = Date.now()
      updateTool(id, { call: copyCall(call), status, result: JSON.stringify(result) ?? 'null', endedAt,
        ...(event?.kind === 'tool' && event.startedAt !== undefined ? { elapsed: Math.max(0, endedAt - event.startedAt) / 1000 } : {}) })
      appendableTextId = ''
    },
    interrupt(cancelled: boolean) {
      const endedAt = Date.now()
      events = events.map(event => event.kind === 'tool' && ['pending', 'running'].includes(event.status) ? { ...event, status: cancelled ? 'cancelled' : 'failed', endedAt,
        ...(event.startedAt === undefined ? {} : { elapsed: Math.max(0, endedAt - event.startedAt) / 1000 }) } : event)
    },
    snapshot(): ChatActivityEvent[] { return [...events] },
  }
}
