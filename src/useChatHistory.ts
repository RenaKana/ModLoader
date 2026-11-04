import { useEffect, useRef, useState } from 'react'
import { reconcileMessageRevisions, validContextCompaction } from './chat-context'
import { apiFetch, errorMessage } from './api'
import { emptyConversation, type Conversation, type ConversationPatch, type ConversationSummary, type Message, type ModelSelection } from './chat-history'
import { validatePresetTools, type ChatPresetCatalog } from './chat-presets'
import { DEFAULT_WEB_SEARCH_SETTINGS } from './web-search'
import { turnTotals } from './chat-turn-metrics'

const SAVE_INTERVAL_MS = 1000
const SAVE_TIMEOUT_MS = 6000
type MetadataPatch = Omit<ConversationPatch, 'messages'>
type Entry = { document: Conversation; persisted: boolean; activate: boolean; metadata: MetadataPatch; messages: Map<string, Message>; timer: ReturnType<typeof setTimeout> | null; inFlight: Promise<void> | null; error: string }
const hasContent = (document: Conversation) => document.messages.length > 0 || Boolean(document.draft.text.trim()) || document.draft.images.length > 0 || Boolean(document.draft.documents?.length)
const dirty = (entry: Entry) => Object.keys(entry.metadata).length > 0 || entry.messages.size > 0
const stopTimer = (entry: Entry) => { if (entry.timer) clearTimeout(entry.timer); entry.timer = null }

export function useChatHistory() {
  const [document, setDocument] = useState<Conversation>(emptyConversation)
  const current = useRef(document)
  const entries = useRef(new Map<string, Entry>())
  const [items, setItems] = useState<ConversationSummary[]>([])
  const [ready, setReady] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [presets, setPresets] = useState<ChatPresetCatalog>({ items: [], default_id: null })
  const [presetsReady, setPresetsReady] = useState(false)
  const [presetError, setPresetError] = useState('')
  const presetCatalog = useRef(presets)
  const presetsAvailable = useRef(false)
  const pendingEmpty = useRef<{ selection: ModelSelection | null } | null>(null)
  const presetWrites = useRef(new Set<Promise<unknown>>())
  const [, render] = useState(0)
  const mounted = useRef(true)
  const latestSave = useRef<(id: string) => Promise<void>>(async () => {})
  function register(value: Conversation, persisted: boolean) {
    const entry: Entry = { document: value, persisted, activate: !persisted, metadata: {}, messages: new Map(), timer: null, inFlight: null, error: '' }
    entries.current.set(value.id, entry)
    return entry
  }
  if (!entries.current.has(document.id)) register(document, false)
  function refresh(entry: Entry) {
    if (!mounted.current) return
    if (entry.document.id === current.current.id) { current.current = entry.document; setDocument(entry.document) }
    render(value => value + 1)
  }
  function schedule(entry: Entry) {
    if (entry.timer || entry.error || !mounted.current) return
    entry.timer = setTimeout(() => { entry.timer = null; void latestSave.current(entry.document.id).catch(() => {}) }, SAVE_INTERVAL_MS)
  }
  function mergeSummary(summary: ConversationSummary) {
    setItems(previous => [summary, ...previous.filter(item => item.id !== summary.id)].sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || (b.sort_order ?? b.last_sent_at ?? 0) - (a.sort_order ?? a.last_sent_at ?? 0) || b.created_at.localeCompare(a.created_at)))
  }
  function adopt(entry: Entry) { current.current = entry.document; setDocument(entry.document) }
  function restore(value: Conversation) {
    const entry = register(value, true)
    // Only disk-loaded requests are interrupted; cached background requests stay alive.
    const messages = value.messages.map(message => {
      if (message.status !== 'generating' && message.turn?.status !== 'running' && !message.requests?.some(request => request.status === 'running')) return message
      const requests = message.requests?.map(request => request.status === 'running' ? { ...request, status: 'incomplete' as const } : request)
      return { ...message, status: message.status === 'generating' ? 'interrupted' as const : message.status, thinking: false, requests,
        ...(message.turn && requests ? { turn: turnTotals(requests, message.turn.startedAt, message.turn.elapsed, 'incomplete') } : {}),
        activity: message.activity?.map(event => event.kind === 'tool' && (event.status === 'pending' || event.status === 'running') ? { ...event, status: 'cancelled' as const } : event) }
    })
    if (messages.some((message, index) => message !== value.messages[index])) update({ messages }, value.id)
    return entry
  }
  function receivePresets(value: ChatPresetCatalog) {
    presetCatalog.current = value
    if (mounted.current) setPresets(value)
  }
  function newConversation(selection: ModelSelection | null = null, catalog = presetCatalog.current) {
    const preset = catalog.items.find(item => item.id === catalog.default_id)
    if (catalog.default_id && !preset) throw new Error('默认对话预设不存在，请重新选择。')
    return emptyConversation(selection, preset)
  }
  function showEmpty(selection: ModelSelection | null = null) {
    const available = presetsAvailable.current
    pendingEmpty.current = available ? null : { selection }
    adopt(register(available ? newConversation(selection) : emptyConversation(selection), false))
    setReady(available)
  }
  async function loadPresets(cancelled: () => boolean = () => false) {
    setPresetError('')
    try {
      const catalog = await timedRequest<ChatPresetCatalog>('GET', '/api/chat-presets')
      if (cancelled() || !mounted.current) return
      receivePresets(catalog)
      presetsAvailable.current = true; setPresetsReady(true)
      // Only an empty, locked placeholder waits for the default. Existing drafts
      // and messages are never reloaded or adopted by a preset retry.
      if (pendingEmpty.current) showEmpty(pendingEmpty.current.selection)
    } catch (issue) {
      if (!cancelled() && mounted.current) {
        presetsAvailable.current = false; setPresetsReady(false)
        setPresetError(`读取对话预设失败：${errorMessage(issue)}`)
      }
    }
  }
  function requirePresets() {
    if (!presetsAvailable.current) throw new Error('请先重新读取对话预设。')
  }
  async function load(cancelled: () => boolean = () => false) {
    setLoadError('')
    try {
      const result = await timedRequest<{ items: ConversationSummary[]; active_id: string | null }>('GET', '/api/conversations')
      const id = result.active_id || result.items[0]?.id
      const value = id ? await timedRequest<Conversation>('GET', `/api/conversations/${id}`) : null
      if (cancelled()) return
      setItems(result.items)
      if (value) { pendingEmpty.current = null; adopt(restore(value)); setReady(true) }
      else showEmpty()
    } catch (issue) { if (!cancelled()) setLoadError(errorMessage(issue)) }
  }
  useEffect(() => {
    mounted.current = true
    let cancelled = false
    void load(() => cancelled)
    void loadPresets(() => cancelled)
    return () => { cancelled = true; mounted.current = false; entries.current.forEach(stopTimer) }
  }, [])

  function update(patch: ConversationPatch, id = current.current.id) {
    const entry = entries.current.get(id)
    if (!entry) return
    const previous = entry.document
    patch = { ...patch }
    if (patch.messages) patch.messages = reconcileMessageRevisions(previous.messages, patch.messages)
    const summary = patch.compaction === undefined ? previous.compaction : patch.compaction
    if (summary && !validContextCompaction(patch.messages || previous.messages, summary)) patch.compaction = null
    const next = { ...previous, ...patch }
    for (const key of ['title', 'selection', 'draft', 'settings', 'compaction'] as const) {
      if (key in patch && patch[key] !== previous[key]) Object.assign(entry.metadata, { [key]: patch[key] })
    }
    if (patch.messages) {
      const old = new Map(previous.messages.map(message => [message.id, message]))
      for (const message of patch.messages) if (old.get(message.id) !== message) entry.messages.set(message.id, message)
      next.message_count = patch.messages.length
      if (!previous.messages.length && previous.title === '新会话') {
        const user = patch.messages.find(message => message.role === 'user')
        if (user) { next.title = user.text.trim().split(/\r?\n/)[0] || user.documents?.[0]?.name || '图片对话'; entry.metadata.title = next.title }
      }
    }
    entry.document = next; refresh(entry)
    if (dirty(entry)) schedule(entry)
  }
  function updateMessages(id: string, change: (messages: Message[]) => Message[]) {
    const entry = entries.current.get(id)
    if (entry) update({ messages: change(entry.document.messages) }, id)
  }
  async function saveOnce(id: string) {
    const entry = entries.current.get(id)
    if (!entry) return
    stopTimer(entry)
    if (entry.inFlight) { await entry.inFlight; if (dirty(entry)) await saveOnce(id); return }
    if (!dirty(entry)) return
    const snapshot = entry.document
    const fields = { ...entry.metadata }
    const messages = [...entry.messages.values()]
    const operation = async () => {
      try {
        if (!hasContent(snapshot)) {
          if (entry.persisted) await timedRequest('DELETE', `/api/conversations/${id}`)
          entry.persisted = false; entry.activate = true
          setItems(previous => previous.filter(item => item.id !== id))
        } else {
          const activate = entry.activate && id === current.current.id
          if (!entry.persisted) {
            await timedRequest('POST', '/api/conversations', { id, title: snapshot.title, selection: snapshot.selection, draft: snapshot.draft, settings: snapshot.settings, ...(snapshot.compaction ? { compaction: snapshot.compaction } : {}) })
            entry.persisted = true
          }
          // Background saves never change the selected conversation.
          const summary = await timedRequest<ConversationSummary>('PATCH', `/api/conversations/${id}`, { ...fields, messages, activate })
          entry.activate = false
          entry.document = { ...entry.document, created_at: summary.created_at, updated_at: summary.updated_at, last_message_at: summary.last_message_at, last_sent_at: summary.last_sent_at, pinned: summary.pinned, sort_order: summary.sort_order }; mergeSummary(summary)
        }
        for (const key of Object.keys(fields) as Array<keyof MetadataPatch>) if (entry.metadata[key] === fields[key]) delete entry.metadata[key]
        for (const message of messages) if (entry.messages.get(message.id) === message) entry.messages.delete(message.id)
        entry.error = ''
      } catch (issue) { entry.error = errorMessage(issue); throw issue }
    }
    const pending = operation(); entry.inFlight = pending; refresh(entry)
    try { await pending } finally {
      entry.inFlight = null; refresh(entry)
      if (dirty(entry) && !entry.error) schedule(entry)
    }
  }
  latestSave.current = saveOnce
  async function flush(id = current.current.id) { await Promise.all([...presetWrites.current]); await saveOnce(id) }
  async function flushAll() { await Promise.all([...entries.current.keys()].map(flush)) }
  async function open(id: string) {
    await flush()
    if (id === current.current.id) return
    const entry = entries.current.get(id) ?? restore(await timedRequest<Conversation>('GET', `/api/conversations/${id}`))
    if (entry.persisted) await timedRequest('PATCH', `/api/conversations/${id}`, { activate: true })
    pendingEmpty.current = null; adopt(entry); setReady(true)
  }
  async function create(selection: ModelSelection | null) {
    await flush()
    requirePresets()
    showEmpty(selection)
  }
  async function replaceMessages(messages: Message[], id = current.current.id, resetTitle = false) {
    await flush(id)
    const entry = entries.current.get(id)
    if (!entry) return
    const title = resetTitle ? { title: '新会话' } : {}
    if (!entry.persisted) { update({ messages, ...title }, id); await flush(id); return }
    messages = reconcileMessageRevisions(entry.document.messages, messages)
    const compaction = validContextCompaction(messages, entry.document.compaction) ? entry.document.compaction : null
    const summary = await timedRequest<ConversationSummary>('PATCH', `/api/conversations/${id}`, { clear_messages: true, messages, compaction, ...title })
    entry.document = { ...entry.document, ...summary, messages, compaction }; mergeSummary(summary); refresh(entry)
    if (!hasContent(entry.document)) { entry.metadata.draft = entry.document.draft; await flush(id) }
  }
  async function clear() { await replaceMessages([], current.current.id, true) }
  async function remove(ids = [current.current.id]) {
    let removedCurrent = false
    const removed = new Set<string>()
    try {
      for (const id of ids) {
        await flush(id)
        const entry = entries.current.get(id)
        if (entry?.persisted || (!entry && items.some(item => item.id === id))) await timedRequest('DELETE', `/api/conversations/${id}`)
        if (entry) stopTimer(entry)
        entries.current.delete(id); removed.add(id); removedCurrent ||= id === current.current.id
        setItems(previous => previous.filter(item => item.id !== id))
      }
    } finally {
      if (removedCurrent) {
        const next = items.find(item => !removed.has(item.id))
        if (next) {
          try { await open(next.id) }
          catch (issue) {
            pendingEmpty.current = null
            adopt(register(emptyConversation(current.current.selection), false)); setReady(false)
            setLoadError(errorMessage(issue))
            throw issue
          }
        } else showEmpty(current.current.selection)
      }
    }
  }
  async function pin(ids: string[], pinned: boolean) {
    // Apply from bottom to top to preserve the selected rows' relative order.
    for (const item of items.filter(item => ids.includes(item.id)).reverse()) {
      await flush(item.id)
      const summary = await timedRequest<ConversationSummary>('PATCH', `/api/conversations/${item.id}`, { pinned })
      const entry = entries.current.get(item.id)
      if (entry) { entry.document = { ...entry.document, ...summary }; refresh(entry) }
      mergeSummary(summary)
    }
  }
  async function reorder(id: string, before: string | null) {
    await flush(id)
    await timedRequest('PATCH', `/api/conversations/${id}`, { move_before: before })
    const result = await timedRequest<{ items: ConversationSummary[] }>('GET', '/api/conversations')
    setItems(result.items)
    for (const summary of result.items) {
      const entry = entries.current.get(summary.id)
      if (entry) { entry.document = { ...entry.document, pinned: summary.pinned, sort_order: summary.sort_order }; refresh(entry) }
    }
  }
  function presetWrite<T>(operation: () => Promise<T>): Promise<T> {
    const pending = operation()
    presetWrites.current.add(pending)
    void pending.then(() => presetWrites.current.delete(pending), () => presetWrites.current.delete(pending))
    return pending
  }
  function applyPreset(identifier: string) {
    requirePresets()
    const settings = current.current.settings
    if (!identifier) {
      const { presetId: _, ...snapshot } = settings
      update({ settings: snapshot })
      return
    }
    const preset = presetCatalog.current.items.find(item => item.id === identifier)
    if (!preset) throw new Error('对话预设不存在，请重新读取。')
    update({ settings: { ...settings, presetId: preset.id, system: preset.system, toolsDraft: preset.toolsDraft, reasoningEffort: preset.reasoningEffort, toolConcurrency: preset.toolConcurrency ?? DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency } })
  }
  function savePreset(name: string, identifier?: string, makeDefault?: boolean) {
    return presetWrite(async () => {
      requirePresets()
      if (!name.trim()) throw new Error('请输入预设名称。')
      const source = current.current
      validatePresetTools(source.settings.toolsDraft)
      const toolConcurrency = source.settings.toolConcurrency ?? DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency
      if (!Number.isInteger(toolConcurrency) || toolConcurrency <= 0) throw new Error('工具并行数必须是正整数。')
      const id = identifier || crypto.randomUUID()
      const payload = { name: name.trim(), system: source.settings.system, toolsDraft: source.settings.toolsDraft, reasoningEffort: source.settings.reasoningEffort ?? null, toolConcurrency, ...(makeDefault === undefined ? {} : { make_default: makeDefault }) }
      const catalog = identifier
        ? await timedRequest<ChatPresetCatalog>('PATCH', `/api/chat-presets/${id}`, payload)
        : await timedRequest<ChatPresetCatalog>('POST', '/api/chat-presets', { id, ...payload })
      receivePresets(catalog)
      const entry = entries.current.get(source.id)
      if (entry) update({ settings: { ...entry.document.settings, presetId: id } }, source.id)
      return id
    })
  }
  const entry = entries.current.get(document.id)!
  const saveError = entry.error
  const saveState = entry.inFlight ? 'saving' : dirty(entry) ? 'pending' : 'saved'
  return { document, current, items, ready, loadError, saveError, saveState, presets, presetsReady, presetError, reloadPresets: () => loadPresets(), applyPreset, savePreset, update, updateMessages, flush, flushAll, open, create, clear, remove, replaceMessages, pin, reorder, reload: () => load() }
}

async function timedRequest<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await apiFetch(path, { method, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(SAVE_TIMEOUT_MS) })
  return response.status === 204 ? undefined as T : response.json()
}
