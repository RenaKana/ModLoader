import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { ArrowDown, ArrowUp, Check, ChevronsDownUp, ChevronDown, ChevronUp, Copy, Download, Eraser, FolderOpen, History, Paperclip, Upload, Loader2, MessageSquare, Pencil, Plus, RotateCcw, Settings2, Sparkles, Square, Trash2, UserRound, Wrench, X } from 'lucide-react'
import type { ChatGenerationActivity, LiveGeneration, PageProps, Telemetry } from '../types'
import { apiFetch, copy, errorMessage, exportMarkdown, GENERATION_REFRESH_INTERVAL_MS, isReady, pickPath, requestProgress } from '../api'
import { Button, ErrorNotice, Field, IconButton, Modal, NumberInput, StateBadge, formatBytes } from '../components'
import NumericParameter, { numberError } from '../NumericParameter'
import { customConfig, runtimeGenerationFields } from '../custom-engine'
import ContextUsage from '../ContextUsage'
import { fullReasoning, reportedNumber, splitThinking } from '../chat-response'
import type { Timings, Usage } from '../chat-response'
import { mergeRequestMetrics, mergeRequestReport, modelRequestLabels, sampleStreamDecode, toolMetricLabel, turnTotals, type BackendRequestReport, type ModelRequestRecord, type PromptProgress, type RequestStatus, type StreamDecodeSample, type StreamDecodeState } from '../chat-turn-metrics'
import { ChatActivity } from '../ChatActivity'
import { createActivityRecorder } from '../chat-activity-recorder'
import { MarkdownContent } from '../MessageContent'
import { GalleryImage, ImageGallery } from '../ImageViewer'
import type { Attachment, DocumentAttachment, Conversation, Message, ToolCall } from '../chat-history'
import ChatDocuments from '../ChatDocuments'
import { reverseImageTool, searchReverseImage } from '../reverse-image-search'
import { useChatHistory } from '../useChatHistory'
import { conversationMarkdown } from '../chat-export'
import ChatHistoryPanel from '../ChatHistoryPanel'
import { ChatTitle } from '../ChatTitle'
import { LiveSpeed, MessageMetrics, MessageTime } from '../ChatMessageMeta'
import MarkdownInput, { type MarkdownInputHandle } from '../MarkdownInput'
import ChatPresetPicker from '../ChatPresetPicker'
import ChatReasoningControl from '../ChatReasoningControl'
import ChatSearchControl from '../ChatSearchControl'
import { useChatModelLoading } from '../chat-model-loading'
import { basicToolDefinitions, fileToolDefinitions, executeBasicTool, toolArguments } from '../chat-tools'
import { useChatFileTools } from '../ChatFileTools'
import { ChatLocalResources } from '../ChatLocalResources'
import { localResourceFromResult } from '../chat-local-resources'
import { buildChatContext, createContextCompaction, validContextCompaction, compactableMessageCount, autoCompactionDecision } from '../chat-context'
import { chatReasoningCapability, reasoningDefaultLabel, reasoningRequestParameters } from '../chat-reasoning'
import { generationDefaultText } from '../parameter-defaults'
import { chatRequestLive, mergeRequestLive, reportedContextCapacity } from '../monitor-generation'
import { DEFAULT_WEB_SEARCH_SETTINGS, type WebReadResult, type WebSearchMode, type WebSearchResult } from '../web-search'
import { runWebChat, type WebCompletion, type WebModelRequest } from '../web-chat'
import WebSearchSources from '../WebSearchSources'
import '../chat-layout.css'

interface ChatContent { content?: string | Array<{ text?: string }>; reasoning_content?: string; reasoning?: string }
interface ChatChunk { prompt_progress?: PromptProgress; choices?: Array<{ delta?: ChatContent & { tool_calls?: Array<{ index?: number; id?: string; type?: string; function?: { name?: string; arguments?: string } }> }; message?: ChatContent & { tool_calls?: ToolCall[] }; finish_reason?: string }>; usage?: Usage; timings?: Timings; modloader_request?: BackendRequestReport; error?: { message?: string } }

function acceptRequestReport(record: ModelRequestRecord, report: BackendRequestReport, controller: AbortController) {
  const next = mergeRequestReport(record, report)
  // A stop from the model manager must cancel the same tool/compaction chain
  // as the chat stop button, including when its receipt is fetched after EOF.
  if (next.status === 'cancelled') controller.abort()
  return next
}

function useChatPanels() {
  const [open, setOpen] = useState({ history: false, options: false })
  const requested = useRef(open)
  const layout = useRef<HTMLDivElement>(null)
  function geometry(node: HTMLElement) {
    const style = getComputedStyle(node)
    return { '--left-width': style.getPropertyValue('--left-width'), '--right-width': style.getPropertyValue('--right-width'),
      opacity: style.opacity, visibility: style.visibility }
  }
  const previous = useRef<Array<{ node: HTMLElement; geometry: ReturnType<typeof geometry> }> | null>(null)
  const animations = useRef<Animation[]>([])
  const contained = useRef<HTMLElement[]>([])
  const settleFrame = useRef<number | null>(null)
  function restoreMessages() {
    if (settleFrame.current !== null) cancelAnimationFrame(settleFrame.current)
    settleFrame.current = null
    if (!contained.current.length) return
    const area = layout.current?.querySelector<HTMLElement>('.chat-messages')
    const viewport = area?.getBoundingClientRect()
    const anchor = viewport && Array.from(area!.querySelectorAll<HTMLElement>('.chat-message'), message => ({ message, rect: message.getBoundingClientRect() }))
      .find(({ rect }) => rect.bottom > viewport.top && rect.top < viewport.bottom)
    contained.current.forEach(node => {
      node.style.removeProperty('content-visibility')
      node.style.removeProperty('contain-intrinsic-block-size')
    })
    contained.current = []
    // Reflowing old messages above the viewport must not move the text the
    // user is reading. The visible message itself keeps its normal layout.
    if (area && anchor) area.scrollTop += anchor.message.getBoundingClientRect().top - anchor.rect.top
  }
  function finish() {
    animations.current.forEach(animation => animation.cancel())
    animations.current = []
    restoreMessages()
  }
  function containOffscreenMessages() {
    const node = layout.current, center = node?.querySelector<HTMLElement>('.chat-panel'), area = node?.querySelector<HTMLElement>('.chat-messages')
    if (!node || !center || !area || contained.current.length) return
    const sideWidth = Math.max(...Array.from(node.querySelectorAll<HTMLElement>('.chat-side-slot'), side => side.offsetWidth + 1))
    // With enough room on both sides, opening either sidebar leaves the
    // transcript geometry unchanged; it needs no temporary containment.
    if (node.clientWidth > center.offsetWidth + 2 * sideWidth) return
    const viewport = area.getBoundingClientRect()
    const messages = Array.from(area.querySelectorAll<HTMLElement>('.chat-message'), message => ({ message, rect: message.getBoundingClientRect() }))
      .filter(({ rect }) => rect.bottom <= viewport.top || rect.top >= viewport.bottom)
    // Keep the visible message's formatting context unchanged: adding
    // containment there would move the browser's reading-position anchor.
    messages.forEach(({ message, rect }) => {
      message.style.containIntrinsicBlockSize = `auto ${rect.height}px`
      message.style.contentVisibility = 'auto'
    })
    contained.current = messages.map(({ message }) => message)
  }
  function change(panel: 'history' | 'options', value: boolean) {
    if (requested.current[panel] === value) return
    requested.current = { ...requested.current, [panel]: value }
    if (settleFrame.current !== null) cancelAnimationFrame(settleFrame.current)
    settleFrame.current = null
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) containOffscreenMessages()
    previous.current = Array.from(layout.current?.children || []).map(node => ({ node: node as HTMLElement, geometry: geometry(node as HTMLElement) }))
    setOpen(requested.current)
  }
  useLayoutEffect(() => {
    const before = previous.current
    previous.current = null
    animations.current.forEach(animation => animation.cancel())
    animations.current = []
    if (!before || !layout.current?.clientWidth || window.matchMedia('(prefers-reduced-motion: reduce)').matches) { restoreMessages(); return }
    const style = getComputedStyle(layout.current)
    const time = style.getPropertyValue('--chat-panel-duration').trim()
    const timing = { duration: parseFloat(time) * (time.endsWith('ms') ? 1 : 1000), easing: style.getPropertyValue('--ease-out').trim() }
    for (const { node, geometry: from } of before) {
      const to = geometry(node)
      if (Object.keys(from).every(key => from[key as keyof typeof from] === to[key as keyof typeof to])) continue
      // One geometry clock drives both the revealed sidebar and the space it
      // takes. These non-inherited properties leave text unscaled.
      animations.current.push(node.animate([from, to], timing))
    }
    const running = animations.current
    if (!running.length) { restoreMessages(); return }
    void Promise.all(running.map(animation => animation.finished)).then(() => {
      if (animations.current !== running) return
      // Present the final motion frame before recalculating the offscreen
      // history once. Keep ordinary layout outside this short transition.
      settleFrame.current = requestAnimationFrame(() => {
        if (animations.current === running) restoreMessages()
      })
    }, () => { /* Reversal and cancellation are handled by the next run. */ })
  }, [open])
  useLayoutEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
    reduced.addEventListener('change', finish)
    window.addEventListener('resize', finish)
    return () => { finish(); reduced.removeEventListener('change', finish); window.removeEventListener('resize', finish) }
  }, [])
  const style = { '--left-width': open.history ? 'var(--history-width)' : '0px', '--right-width': open.options ? 'var(--options-width)' : '0px' } as CSSProperties
  return { ...open, layout, change, finish, style }
}

export default function Chat({ active, selectedInstance, knownContextCapacity, onLockChange, onGenerationActivity, onInspectTurn, ...props }: PageProps & { active: boolean; selectedInstance?: { id: string; request: number }; knownContextCapacity?: (instanceId: string) => number | null; onLockChange: (locked: boolean) => void; onGenerationActivity: (activity: ChatGenerationActivity) => void; onInspectTurn: (message: Message) => void }) {
  const history = useChatHistory()
  const conversation = history.document
  const [instanceId, setInstanceId] = useState('')
  const messages = conversation.messages
  const input = conversation.draft.text
  const attachments = conversation.draft.images
  const documents = conversation.draft.documents || []
  const { system, temperature, maxTokens, stream, toolsDraft, reasoningEffort, basicTools = false, toolDirectory = '', autoCompact = true,
    webSearch = DEFAULT_WEB_SEARCH_SETTINGS.webSearch, webSearchRounds = DEFAULT_WEB_SEARCH_SETTINGS.webSearchRounds, toolConcurrency = DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency,
    webSearchMode = DEFAULT_WEB_SEARCH_SETTINGS.webSearchMode } = conversation.settings
  const setInput = (text: string) => history.update({ draft: { ...history.current.current.draft, text } })
  const setAttachments = (value: Attachment[] | ((previous: Attachment[]) => Attachment[])) => history.update({ draft: { ...history.current.current.draft, images: typeof value === 'function' ? value(history.current.current.draft.images) : value } })
  const changeSetting = (patch: Partial<typeof conversation.settings>) => history.update({ settings: { ...history.current.current.settings, ...patch } })
  const changeToolDirectory = async (path: string) => {
    const id = history.current.current.id
    changeSetting({ toolDirectory: path, ...(path ? { basicTools: true } : {}) })
    await history.flush(id)
  }
  const setSystem = (system: string) => changeSetting({ system })
  const setTemperature = (temperature: number | undefined) => changeSetting({ temperature })
  const setMaxTokens = (maxTokens: number | undefined) => changeSetting({ maxTokens })
  const setStream = (stream: boolean) => changeSetting({ stream })
  const setToolsDraft = (toolsDraft: string) => changeSetting({ toolsDraft })
  const setWebSearchMode = (mode: WebSearchMode | 'off') => changeSetting(mode === 'off' ? { webSearch: false } : { webSearch: true, webSearchMode: mode })
  const generations = useRef(new Map<string, { controller: AbortController; instanceId: string; messageId: string; started?: number; webProgress?: string; runningTools: Set<string>; phase: 'planning' | 'searching' | 'reading' | 'tools' | 'compacting' | 'generating'; done: Promise<void> | null; contextCapacity?: number; streamSample?: StreamDecodeSample; report?: () => void; observeLive?: (live: LiveGeneration) => void }>())
  const contextCapacities = useRef(new Map<string, number>())
  const [, renderGeneration] = useState(0)
  const activeGeneration = generations.current.get(conversation.id)
  const busy = Boolean(activeGeneration)
  const activeMessageId = activeGeneration?.messageId || ''
  const generationStarted = activeGeneration?.started
  const preparing = Boolean(activeGeneration && activeGeneration.phase !== 'generating')
  const phaseLabel = activeGeneration && activeGeneration.runningTools.size > 1 ? `调用工具 · ${activeGeneration.runningTools.size} 项并行` : activeGeneration?.webProgress || (activeGeneration?.phase === 'planning' ? '规划检索' : activeGeneration?.phase === 'searching' ? '正在搜索' : activeGeneration?.phase === 'reading' ? '阅读网页' : activeGeneration?.phase === 'tools' ? '调用工具' : activeGeneration?.phase === 'compacting' ? '压缩上下文' : '')
  const [exporting, setExporting] = useState(false)
  const [transitioning, setTransitioning] = useState(false)
  const [closing, setClosing] = useState(false)
  const panels = useChatPanels()
  const historyOpen = panels.history, setHistoryOpen = (value: boolean) => panels.change('history', value)
  const [dialog, setDialog] = useState<'clear' | 'delete' | 'retry' | null>(null)
  const [deleteIds, setDeleteIds] = useState<string[]>([])
  const [historyReset, setHistoryReset] = useState(0)
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null)
  const editingTextarea = useRef<MarkdownInputHandle>(null)
  const [retryId, setRetryId] = useState('')
  const [renaming, setRenaming] = useState<{ id: string; text: string; originalTitle: string; appliedTitle?: string } | null>(null)
  const renameInput = useRef<HTMLInputElement>(null)
  const [atBottom, setAtBottom] = useState(true)
  const [promptIndex, setPromptIndex] = useState(-1)
  const jumpTarget = useRef<number | null>(null)
  const navigationScroll = useRef<number | null>(null)
  const lockedRef = useRef(false)
  const closeHandler = useRef<() => Promise<void>>(async () => {})
  const modelRequest = useRef<number | undefined>(undefined)
  const attachmentRead = useRef<Promise<void> | null>(null)
  const [liveSample, setLiveSample] = useState<{ messageId: string; live: LiveGeneration | null; error: string } | null>(null)
  const activeRequest = messages.find(message => message.id === activeMessageId)?.requests?.at(-1)
  const liveGeneration = chatRequestLive(liveSample?.messageId === activeMessageId ? liveSample.live : undefined, activeRequest, activeGeneration?.streamSample)
  const liveError = activeGeneration?.streamSample?.requestId === activeRequest?.id && activeGeneration?.streamSample?.tokensPerSecond != null
    ? '' : liveSample?.messageId === activeMessageId ? liveSample.error : ''
  const [error, setError] = useState('')
  const optionsOpen = panels.options, setOptionsOpen = (value: boolean) => panels.change('options', value)
  const [attachmentBusy, setAttachmentBusy] = useState(false)
  const [fileDrag, setFileDrag] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const scrollArea = useRef<HTMLDivElement>(null)
  const scrollContent = useRef<HTMLDivElement>(null)
  const shouldScroll = useRef(true)
  const lastScrollTop = useRef(0)
  const textarea = useRef<MarkdownInputHandle>(null)
  const ready = props.state.instances.filter(instance => isReady(instance.state))
  const loading = props.state.instances.filter(instance => ['starting', 'loading', 'preparing'].includes(instance.state))
  const instance = props.state.instances.find(item => item.id === instanceId)
  const available = instance && isReady(instance.state)
  const modelOptions = ready.map(item => ({ value: item.id, label: item.alias, state: item.state }))
  if (instance && !available) modelOptions.push({ value: instance.id, label: instance.alias, state: instance.state })
  const runtime = props.state.runtimes.find(item => item.id === instance?.runtime_id)
  const profile = props.state.profiles.find(item => item.id === instance?.profile_id)
  const reasoningCapability = chatReasoningCapability(instance, props.state.models.find(item => item.id === instance?.model_id), runtime, profile)
  const generationDefaults = profile || { backend: instance?.backend || '', generation: {}, extra_args: [] }
  const generationFields = runtimeGenerationFields(props.state.backends.find(backend => backend.id === instance?.backend)?.generation_parameters || props.state.generation_parameters || [], runtime, instance?.capabilities)
  const streamUsage = instance?.backend !== 'custom' || (instance.capabilities.stream_usage ?? customConfig(runtime).stream_usage) === true
  const temperatureField = generationFields.find(field => field.key === 'temperature')
  const maxTokensField = generationFields.find(field => field.key === 'max_tokens')
  const inputCapabilities = (instance?.capabilities.input as string[] | undefined) || ['text']
  const supportsImages = inputCapabilities.includes('image')
  const hasImages = attachments.length > 0
  const effectiveMessages = validContextCompaction(messages, conversation.compaction) ? messages.slice(messages.findIndex(message => message.id === conversation.compaction!.throughMessageId) + 1) : messages
  const contextHasImages = effectiveMessages.some(message => !message.error && !message.tools?.length && message.images.length > 0)
  const incompatibleImages = (hasImages || contextHasImages) && !supportsImages
  const lastResponse = [...messages].reverse().find(message => message.role === 'assistant')
  const matchingResponse = lastResponse && (!conversation.compaction || lastResponse.createdAt > conversation.compaction.createdAt) && lastResponse.modelId === instance?.model_id && lastResponse.profileId === instance?.profile_id ? lastResponse : undefined
  const promptTokens = reportedNumber(matchingResponse?.usage?.prompt_tokens)
  const outputTokens = reportedNumber(matchingResponse?.usage?.completion_tokens)
  const completedContext = reportedNumber(matchingResponse?.usage?.total_tokens) ?? (promptTokens !== undefined && outputTokens !== undefined ? promptTokens + outputTokens : undefined)
  const contextUsed = busy ? liveGeneration?.context_tokens : completedContext
  const contextCapacity = busy ? liveGeneration?.context_capacity ?? activeGeneration?.contextCapacity : matchingResponse?.contextCapacity
  const locked = transitioning || attachmentBusy || closing || !history.ready
  const inputLocked = locked || busy
  const fileTools = useChatFileTools(path => props.notify(`已写入 ${path}`))
  const { quickLoadOptions, loadProfile, cancelAutoSelect, unloadInstance, unloadingIds, unloadError } = useChatModelLoading(props.state, props.refresh, selectInstance, conversation.id, instanceId)
  useLayoutEffect(() => { lockedRef.current = locked; onLockChange(locked) }, [locked, onLockChange])
  function selectInstance(id: string) {
    const target = props.state.instances.find(item => item.id === id)
    const previous = history.current.current.selection
    const previousBackend = props.state.profiles.find(item => item.id === previous?.profile_id)?.backend
    const changedModel = previous && target && (previous.model_id !== target.model_id || previousBackend && previousBackend !== target.backend)
    setInstanceId(id); setError('')
    // Unique aliases distinguish parallel instances; runtime IDs stay on requests.
    history.update({ selection: target ? { profile_id: target.profile_id || '', model_id: target.model_id || '', alias: target.alias } : null,
      ...(changedModel ? { settings: { ...history.current.current.settings, reasoningEffort: undefined } } : {}) })
  }

  useEffect(() => {
    if (!history.ready || !selectedInstance || modelRequest.current === selectedInstance.request) return
    modelRequest.current = selectedInstance.request
    if (lockedRef.current || generations.current.has(history.current.current.id)) { props.notify('请先停止生成或等待当前操作完成，再切换模型。', 'info'); return }
    selectInstance(selectedInstance.id)
  }, [selectedInstance, history.ready])
  useEffect(() => {
    if (!history.ready || transitioning || closing) return
    if (activeGeneration) { setInstanceId(activeGeneration.instanceId); return }
    const selection = conversation.selection
    if (ready.length === 1) {
      const target = ready[0]
      setInstanceId(target.id)
      if (!selection || selection.profile_id !== target.profile_id || selection.model_id !== target.model_id || selection.alias !== target.alias) selectInstance(target.id)
    } else if (!ready.length && loading.length === 1) {
      setInstanceId(loading[0].id)
    } else if (selection) {
      const matched = ready.find(item => item.id === selection.instance_id)
        || ready.find(item => item.alias === selection.alias && (!selection.profile_id || item.profile_id === selection.profile_id))
        || (!selection.instance_id ? ready.find(item => item.profile_id === selection.profile_id) : undefined)
      setInstanceId(matched?.id || '')
    } else if (!messages.length && ready.length) selectInstance(ready[0].id)
  }, [history.ready, conversation.id, conversation.selection, props.state.instances, busy, transitioning, closing])
  useEffect(() => () => { generations.current.forEach(task => task.controller.abort()) }, [])
  async function stopGeneration(id: string) {
    const task = generations.current.get(id)
    task?.controller.abort()
    await task?.done
  }
  closeHandler.current = async () => {
    lockedRef.current = true; onLockChange(true); setClosing(true)
    try {
      await Promise.all([...generations.current.keys()].map(stopGeneration))
      await attachmentRead.current
      await fileTools.waitForWrites()
      await history.flushAll()
    } catch (issue) { setClosing(false); throw issue }
  }
  useEffect(() => {
    const unsubscribe = window.modloader?.onBeforeClose(() => closeHandler.current())
    const cancel = window.modloader?.onCloseCancelled(() => setClosing(false))
    return () => { unsubscribe?.(); cancel?.() }
  }, [])
  useEffect(() => {
    function escape(event: KeyboardEvent) { if (event.key === 'Escape' && !document.querySelector('[role="dialog"]:not([popover]), [role="dialog"]:popover-open')) { setHistoryOpen(false); setOptionsOpen(false); setEditing(null) } }
    window.addEventListener('keydown', escape)
    return () => window.removeEventListener('keydown', escape)
  }, [])
  useEffect(() => {
    if (!activeGeneration?.messageId) return
    const generation = activeGeneration
    const { instanceId, messageId } = generation
    const controller = new AbortController()
    let inFlight = false
    async function sample() {
      if (inFlight) return
      inFlight = true
      try {
        const response = await apiFetch(`/api/instances/${instanceId}/telemetry`, { signal: controller.signal })
        const telemetry = await response.json() as Telemetry
        if (!controller.signal.aborted) {
          if (telemetry.instance_id !== instanceId) throw new Error('监控响应与当前实例不匹配')
          const live = telemetry.live_generation
          const matched = live?.client_request_id === messageId ? live : undefined
          if (matched) generation.observeLive?.(matched)
          if (!preparing) setLiveSample({ messageId, live: matched ?? null, error: matched ? '' : live?.reason || '尚无当前请求的有效采样' })
          const capacity = reportedContextCapacity(telemetry) ?? contextCapacities.current.get(instanceId) ?? knownContextCapacity?.(instanceId) ?? null
          if (capacity !== null) {
            contextCapacities.current.set(instanceId, capacity)
            if (generation.contextCapacity !== capacity) {
              generation.contextCapacity = capacity
              generation.report?.()
            }
          }
        }
      } catch (issue) {
        if (!controller.signal.aborted) setLiveSample({ messageId, live: null, error: errorMessage(issue) })
      } finally { inFlight = false }
    }
    void sample()
    const timer = window.setInterval(() => void sample(), GENERATION_REFRESH_INTERVAL_MS)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [conversation.id, activeMessageId, activeGeneration?.instanceId, preparing, generationStarted, knownContextCapacity])
  function followOutput() {
    const node = scrollArea.current
    if (!node || !node.clientHeight) return // The chat stays mounted while another page is visible.
    if (shouldScroll.current) node.scrollTop = node.scrollHeight
    lastScrollTop.current = node.scrollTop
    updateScrollState()
  }
  useLayoutEffect(followOutput, [messages, busy])
  useLayoutEffect(() => { stopNavigationScroll(); shouldScroll.current = true; jumpTarget.current = null; setPromptIndex(-1); followOutput() }, [conversation.id])
  useLayoutEffect(() => { setRenaming(null) }, [conversation.id])
  useLayoutEffect(() => { if (editing) editingTextarea.current?.focus() }, [editing?.id])
  useLayoutEffect(() => { renameInput.current?.select() }, [renaming?.id])
  useEffect(() => {
    const area = scrollArea.current
    if (!area) return
    const settled = () => {
      if (navigationScroll.current !== null && Math.abs(area.scrollTop - navigationScroll.current) < 2) navigationScroll.current = null
    }
    area.addEventListener('scrollend', settled)
    return () => area.removeEventListener('scrollend', settled)
  }, [])
  useLayoutEffect(() => {
    // Reasoning collapse, code highlighting and the finished footer can change
    // the scroll range after the message render has already committed.
    const observer = new ResizeObserver(followOutput)
    if (scrollArea.current) observer.observe(scrollArea.current)
    if (scrollContent.current) observer.observe(scrollContent.current)
    return () => observer.disconnect()
  }, [])

  function copyOriginal(text: string) {
    void copy(text).then(() => props.notify('原文已复制')).catch(issue => props.notify(errorMessage(issue), 'error'))
  }
  function pasteImages(event: React.ClipboardEvent<HTMLElement>) {
    if (!supportsImages || lockedRef.current || busy) return
    const files = Array.from(event.clipboardData.items).filter(item => item.type.startsWith('image/')).map(item => item.getAsFile()).filter((file): file is File => file !== null)
    if (files.length) { event.preventDefault(); void addFiles(files) }
  }

  useEffect(() => {
    if (!active) return
    let depth = 0
    const reset = () => { depth = 0; setFileDrag(false) }
    const hasFiles = (event: DragEvent) => event.dataTransfer?.types.includes('Files')
    const modalOpen = () => Boolean(document.querySelector('[role="dialog"][aria-modal="true"], dialog[open]'))
    const enter = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault(); depth++
      if (!modalOpen()) setFileDrag(true)
    }
    const over = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault()
      event.dataTransfer!.dropEffect = inputLocked || modalOpen() ? 'none' : 'copy'
    }
    const leave = (event: DragEvent) => { if (hasFiles(event) && --depth <= 0) reset() }
    const drop = (event: DragEvent) => {
      if (!hasFiles(event)) return
      event.preventDefault(); event.stopPropagation(); reset()
      if (inputLocked || lockedRef.current || modalOpen()) return
      const items = Array.from(event.dataTransfer!.items)
      if (items.some(item => item.webkitGetAsEntry?.()?.isDirectory)) {
        setError('请拖入文件，不支持文件夹。'); return
      }
      void addFiles(Array.from(event.dataTransfer!.files))
    }
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') reset() }
    window.addEventListener('dragenter', enter)
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop, true)
    window.addEventListener('dragend', reset)
    window.addEventListener('blur', reset)
    window.addEventListener('keydown', escape)
    return () => {
      reset()
      window.removeEventListener('dragenter', enter)
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop, true)
      window.removeEventListener('dragend', reset)
      window.removeEventListener('blur', reset)
      window.removeEventListener('keydown', escape)
    }
  }, [active, inputLocked, supportsImages, conversation.id])

  async function addFiles(files: File[]) {
    if (!files.length) return
    if (lockedRef.current || generations.current.has(history.current.current.id)) return
    lockedRef.current = true; onLockChange(true)
    let finished!: () => void
    attachmentRead.current = new Promise(resolve => { finished = resolve })
    setAttachmentBusy(true); setError('')
    try {
      const images: Attachment[] = [], documents: DocumentAttachment[] = [], failures: string[] = []
      for (const file of files) {
        try {
          if (file.type.startsWith('image/')) {
            if (!supportsImages) throw new Error('当前模型未启用图片输入')
            images.push(await new Promise<Attachment>((resolve, reject) => {
              const reader = new FileReader()
              reader.onload = () => resolve({ id: crypto.randomUUID(), name: file.name || '剪贴板图片', size: file.size, url: String(reader.result) })
              reader.onerror = reader.onabort = () => reject(new Error('无法读取图片'))
              reader.readAsDataURL(file)
            }))
          } else {
            const signal = AbortSignal.timeout(120_000)
            try {
              const response = await apiFetch(`/api/chat/attachments/text?name=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file, signal })
              documents.push({ ...await response.json(), id: crypto.randomUUID() })
            } catch (issue) { if (signal.aborted) throw new Error('文档读取超时，请重试'); throw issue }
          }
        } catch (issue) { failures.push(`${file.name}：${errorMessage(issue)}`) }
      }
      const draft = history.current.current.draft
      history.update({ draft: { ...draft, images: [...draft.images, ...images], documents: [...(draft.documents || []), ...documents] } })
      if (failures.length) setError(failures.join('\n'))
    } catch (issue) { setError(errorMessage(issue)) }
    finally { setAttachmentBusy(false); attachmentRead.current = null; finished() }
  }

  async function compressContext(source: Conversation, controller: AbortController, retainRecentTurns: number, generate?: (body: Record<string, unknown>) => Promise<ChatChunk>) {
    if (!instance || !available) throw new Error('请先加载模型。')
    const result = await createContextCompaction({
      messages: source.messages, system: source.settings.system, compaction: source.compaction, retainRecentTurns, signal: controller.signal,
      generate: async (context, signal) => {
        const body = {
          model: instance.alias, instance_id: instance.id, messages: context, stream: false,
          ...reasoningRequestParameters(reasoningCapability, reasoningEffort, instance.backend),
          ...(instance.backend === 'ninfer' && context.some(message => message.reasoning_content) ? { preserve_thinking: true } : {}),
        }
        const chunk = generate ? await generate(body) : await apiFetch('/api/chat', { method: 'POST', signal, body: JSON.stringify(body) }).then(response => response.json() as Promise<ChatChunk>)
        if (chunk.error) throw new Error(chunk.error.message || '压缩失败')
        const choice = chunk.choices?.[0]
        if (choice?.finish_reason === 'length') throw new Error('摘要达到输出上限，未替换原上下文。')
        if (choice?.message?.tool_calls?.length) throw new Error('压缩需要摘要正文，模型却返回了工具调用。')
        const content = choice?.message?.content
        const text = typeof content === 'string' ? content : Array.isArray(content) ? content.map(part => part.text || '').join('') : ''
        return splitThinking(text, true).text
      },
    })
    controller.signal.throwIfAborted()
    history.update({ compaction: result }, source.id)
    await history.flush(source.id)
    return result
  }
  function launchCompaction() {
    if (lockedRef.current || busy || !available || !messages.length) return
    const source = history.current.current, controller = new AbortController()
    const operationId = crypto.randomUUID(), started = performance.now(), startedAt = Date.now()
    let requests: ModelRequestRecord[] = [], status: RequestStatus = 'running'
    const publish = (ended = false) => onGenerationActivity({ instanceId: instance.id, messageId: operationId,
      phase: ended ? status === 'cancelled' ? 'cancelled' : status === 'completed' ? 'completed' : 'failed' : 'compacting', requests,
      turn: turnTotals(requests, startedAt, (performance.now() - started) / 1000, status, ended ? Date.now() : undefined) })
    const generation = { controller, instanceId: instance.id, messageId: operationId, started, runningTools: new Set<string>(), phase: 'compacting' as const, done: null as Promise<void> | null }
    generations.current.set(source.id, generation); setError(''); renderGeneration(value => value + 1)
    generation.done = (async () => {
      try {
        await history.flush(source.id)
        await compressContext(source, controller, 0, async body => {
          const clock = performance.now()
          let record: ModelRequestRecord = { id: crypto.randomUUID(), index: 1, kind: 'compaction', status: 'running', startedAt: Date.now(), labels: ['手动压缩', ...modelRequestLabels(body.messages as Record<string, unknown>[])] }
          let reported = false
          requests = [record]; publish()
          try {
            const response = await apiFetch('/api/chat', { method: 'POST', signal: controller.signal, headers: { 'X-Modloader-Request-Id': operationId, 'X-Modloader-Call-Id': record.id }, body: JSON.stringify(body) })
            const chunk = await response.json() as ChatChunk
            record = { ...mergeRequestMetrics(record, { usage: chunk.usage ?? {}, timings: chunk.timings ?? {}, prompt_progress: chunk.prompt_progress }, Number(body.n ?? 1) > 1), status: chunk.error ? 'failed' : 'completed' }
            if (chunk.modloader_request) { record = acceptRequestReport(record, chunk.modloader_request, controller); reported = true }
            controller.signal.throwIfAborted()
            return chunk
          } catch (issue) {
            if (!reported) {
              try {
                const report = await apiFetch(`/api/instances/${instance.id}/request-metrics/${record.id}`, { signal: AbortSignal.timeout(3000) }).then(response => response.json() as Promise<BackendRequestReport>)
                record = acceptRequestReport(record, report, controller); reported = true
              } catch { /* No reported usage is different from zero usage. */ }
            }
            record = { ...record, status: reported && record.status !== 'running' ? record.status : controller.signal.aborted ? 'cancelled' : 'failed' }; throw issue
          } finally { record = { ...record, elapsed: record.elapsed ?? (performance.now() - clock) / 1000, endedAt: record.endedAt ?? Date.now() }; requests = [record]; publish() }
        })
        status = 'completed'
      }
      catch (issue) { status = controller.signal.aborted ? 'cancelled' : 'failed'; if (!controller.signal.aborted && history.current.current.id === source.id) setError(`压缩失败：${errorMessage(issue)}`) }
      finally { publish(true); if (generations.current.get(source.id) === generation) generations.current.delete(source.id); renderGeneration(value => value + 1) }
    })()
  }

  function launchSend() {
    if (lockedRef.current || generations.current.has(history.current.current.id)) return
    const task = send()
    const generation = generations.current.get(history.current.current.id)
    if (generation) generation.done = task
  }
  async function send(replacement?: { context: Message[]; user: Message }) {
    const conversationId = conversation.id
    const setMessages = (change: (previous: Message[]) => Message[]) => history.updateMessages(conversationId, change)
    const sourceMessages = replacement?.context ?? messages
    const sourceText = replacement?.user.text ?? input.trim()
    const sourceImages = replacement?.user.images ?? attachments
    const sourceDocuments = replacement ? replacement.user.documents || [] : documents
    const imageReferences = webSearch && webSearchMode === 'agent'
    const searchableImages = imageReferences ? [...new Map([...sourceMessages.filter(message => message.role === 'user').flatMap(message => message.images), ...sourceImages].map(image => [image.id, image])).values()] : []
    const builtInDefinitions = [...(basicTools ? [...basicToolDefinitions, ...(toolDirectory.trim() ? fileToolDefinitions : [])] : []), ...(imageReferences ? [reverseImageTool()] : [])]
    const effectiveSource = validContextCompaction(sourceMessages, conversation.compaction) ? sourceMessages.slice(sourceMessages.findIndex(message => message.id === conversation.compaction!.throughMessageId) + 1) : sourceMessages
    const imageConflict = !supportsImages && (sourceImages.length > 0 || effectiveSource.some(message => !message.error && !message.tools?.length && message.images.length > 0))
    if ((!replacement && (lockedRef.current || generations.current.has(conversationId))) || attachmentBusy || !available || (!sourceText.trim() && !sourceImages.length && !sourceDocuments.length) || imageConflict) {
      if (replacement) setError(!available ? '模型未就绪，无法重新生成。' : imageConflict ? '当前模型不支持这段对话中的图片。' : '请输入消息内容。')
      return
    }
    const numericError = (temperatureField && numberError(temperatureField, temperature)) || (maxTokensField && numberError(maxTokensField, maxTokens))
    if (numericError) { setError(numericError); setOptionsOpen(true); return }
    if ((webSearch || basicTools) && (!Number.isSafeInteger(webSearchRounds) || webSearchRounds <= 0)) { setError('工具调用预算必须是正整数。'); setOptionsOpen(true); return }
    if ((webSearch || basicTools) && (!Number.isInteger(toolConcurrency) || toolConcurrency <= 0)) { setError('工具并行数必须是正整数。'); setOptionsOpen(true); return }
    let tools: unknown[] | undefined
    if (toolsDraft.trim()) {
      try { tools = JSON.parse(toolsDraft); if (!Array.isArray(tools)) throw new Error('工具定义须为 JSON 数组。') }
      catch (issue) { setError(errorMessage(issue)); return }
    }
    setError(''); shouldScroll.current = true
    const controller = new AbortController()
    const turnStarted = performance.now(), turnStartedAt = Date.now()
    const knownCapacity = contextCapacities.current.get(instance.id) ?? knownContextCapacity?.(instance.id) ?? [...sourceMessages].reverse().find(message => message.instanceId === instance.id && (reportedNumber(message.contextCapacity) ?? 0) > 0)?.contextCapacity
    const assistant: Message = { id: crypto.randomUUID(), role: 'assistant', text: '', images: [], model: instance.alias, backend: instance.backend, profileId: instance.profile_id, modelId: instance.model_id, instanceId: instance.id, createdAt: turnStartedAt, status: 'generating', contextCapacity: knownCapacity }
    const generation = { controller, instanceId: instance.id, messageId: assistant.id, started: turnStarted, webProgress: '', runningTools: new Set<string>(), phase: (webSearch && webSearchMode === 'retrieve' ? 'planning' : 'generating') as 'planning' | 'searching' | 'reading' | 'tools' | 'compacting' | 'generating', done: null as Promise<void> | null, contextCapacity: knownCapacity, streamSample: undefined as StreamDecodeSample | undefined, report: undefined as (() => void) | undefined, observeLive: undefined as ((live: LiveGeneration) => void) | undefined }
    generations.current.set(conversationId, generation); renderGeneration(value => value + 1)
    const release = () => { if (generations.current.get(conversationId) === generation) generations.current.delete(conversationId); renderGeneration(value => value + 1) }
    // Save the draft first: a storage failure must not consume input or invoke a model.
    try { await history.flush(conversationId) } catch { release(); return }
    if (controller.signal.aborted) { release(); return }
    const user: Message = replacement?.user ?? { id: crypto.randomUUID(), role: 'user', text: sourceText, images: sourceImages, ...(sourceDocuments.length ? { documents: sourceDocuments } : {}), createdAt: turnStartedAt, status: 'completed' }
    let requests: ModelRequestRecord[] = []
    let turnStatus: RequestStatus = 'running'
    let turnEndedAt: number | undefined
    let turnElapsed: number | undefined
    let sent = false
    let currentRequestId: string | undefined
    let responseStream = stream
    let started: number | undefined
    let text = '', reasoning = '', reasoningHistory = ''
    const activity = createActivityRecorder()
    let thinking = false, finished = false
    let firstOutput: number | undefined, responseAt: number | undefined, finishReason: string | undefined
    let toolCalls: ToolCall[] = []
    let usage: Usage | undefined, timings: Timings | undefined
    let pendingRequest: Promise<ChatChunk> | undefined
    const totals = () => turnTotals(requests, turnStartedAt, turnElapsed ?? (performance.now() - turnStarted) / 1000, turnStatus, turnEndedAt)
    const publishPhase = (phase: ChatGenerationActivity['phase']) => onGenerationActivity({ instanceId: instance.id, messageId: assistant.id, phase, model: instance.alias, contextCapacity: generation.contextCapacity,
      requests, turn: totals(), stream: generation.streamSample, toolActivity: activity.snapshot().filter((event): event is Extract<ReturnType<typeof activity.snapshot>[number], { kind: 'tool' }> => event.kind === 'tool'),
      toolCount: activity.snapshot().filter(event => event.kind === 'tool' && event.startedAt !== undefined).length })
    const saveMetrics = () => {
      const turn = totals()
      assistant.contextCapacity = generation.contextCapacity
      setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, requests, turn, contextCapacity: generation.contextCapacity } : message))
      publishPhase(turnStatus === 'running' ? generation.phase : turnStatus === 'incomplete' ? 'failed' : turnStatus)
    }
    generation.report = saveMetrics
    const update = () => {
      const parsed = splitThinking(text, finished)
      activity.capture(reasoning || parsed.reasoning || '', parsed.text, toolCalls)
      setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, text: parsed.text, reasoning: reasoning || parsed.reasoning, reasoningHistory: reasoningHistory || undefined, activity: activity.snapshot(), thinking: !finished && (thinking || parsed.thinking), tools: toolCalls.length ? toolCalls.filter(Boolean).map(call => ({ ...call, function: { ...call.function } })) : undefined, usage, timings, firstOutput, responseAt, finishReason } : message))
    }
    const consume = (chunk: ChatChunk) => {
      if (chunk.usage) usage = { ...usage, ...chunk.usage }
      if (chunk.timings) timings = { ...timings, ...chunk.timings }
      const choice = chunk.choices?.[0], part = choice?.delta ?? choice?.message
      const thought = part?.reasoning_content ?? part?.reasoning, content = part?.content
      const addition = typeof content === 'string' ? content : Array.isArray(content) ? content.map(part => part.text || '').join('') : ''
      if (typeof thought === 'string' && thought) { reasoning += thought; thinking = true }
      if (addition) { text += addition; thinking = false }
      if (addition || (typeof thought === 'string' && thought) || choice?.delta?.tool_calls?.length || choice?.message?.tool_calls?.length) {
        responseAt ??= Date.now()
        if (responseStream && started !== undefined && firstOutput === undefined) firstOutput = (performance.now() - started) / 1000
      }
      if (choice?.finish_reason) { finishReason = choice.finish_reason; finished = true }
      if (choice?.message?.tool_calls) toolCalls = choice.message.tool_calls
      for (const call of choice?.delta?.tool_calls || []) {
        const index = call.index ?? 0
        if (!toolCalls[index]) toolCalls[index] = { id: call.id, type: call.type || 'function', function: { name: '', arguments: '' } }
        if (call.id) toolCalls[index].id = call.id
        if (call.function?.name) toolCalls[index].function.name += call.function.name
        if (call.function?.arguments) toolCalls[index].function.arguments += call.function.arguments
      }
      update()
    }
    // The final report travels with the response, so even a sub-second call is
    // retained. Paired SSE counters update Decode immediately; final-only SSE
    // backends also contribute their separately attributed live observations.
    const executeModelRequest = async (body: Record<string, unknown>, kind: ModelRequestRecord['kind'], receive?: (chunk: ChatChunk) => void): Promise<ChatChunk> => {
      controller.signal.throwIfAborted()
      const clock = performance.now()
      let record: ModelRequestRecord = { id: crypto.randomUUID(), index: requests.length + 1, kind, status: 'running', startedAt: Date.now(), labels: modelRequestLabels(body.messages as Record<string, unknown>[]) }
      let decodeState: StreamDecodeState | undefined
      generation.streamSample = undefined
      currentRequestId = record.id
      requests = [...requests, record]
      activity.startPass(record.id)
      const store = () => { requests = requests.map(item => item.id === record.id ? record : item); saveMetrics() }
      const observeLive = (live: LiveGeneration) => {
        const next = mergeRequestLive(record, live)
        if (next !== record) { record = next; store() }
      }
      generation.observeLive = observeLive
      store()
      let lastChunk: ChatChunk = {}, reported = false
      const observe = (chunk: ChatChunk) => {
        const choice = chunk.choices?.[0], part = choice?.delta ?? choice?.message
        const hasOutput = Boolean(part?.content || part?.reasoning_content || part?.reasoning || choice?.delta?.tool_calls?.length || choice?.message?.tool_calls?.length)
        const firstOutputChanged = Boolean(body.stream && hasOutput && record.firstOutput === undefined)
        record = { ...mergeRequestMetrics(record, { usage: chunk.usage ?? {}, timings: chunk.timings ?? {}, prompt_progress: chunk.prompt_progress }, Number(body.n ?? 1) > 1),
          ...(firstOutputChanged ? { firstOutput: (performance.now() - clock) / 1000, firstOutputBasis: 'client' as const } : {}) }
        if (chunk.modloader_request) { record = acceptRequestReport(record, chunk.modloader_request, controller); reported = true }
        if (body.stream && (body.n === undefined || body.n === 1)) {
          decodeState = sampleStreamDecode(decodeState, record.id, chunk, Date.now())
          generation.streamSample = decodeState?.sample
        }
        if (chunk.choices?.length) lastChunk = chunk
        receive?.(chunk)
        const calls = choice?.message?.tool_calls ?? (kind === 'generation' ? toolCalls.filter(Boolean) : [])
        const labels = calls.map(call => toolMetricLabel(call.function.name)).filter(Boolean)
        const labelsChanged = labels.some(label => !record.labels.includes(label))
        if (labelsChanged) record = { ...record, labels: [...new Set([...record.labels, ...labels])] }
        if (chunk.usage || chunk.timings || chunk.prompt_progress || chunk.modloader_request || labelsChanged || firstOutputChanged) store()
        if (chunk.error) throw new Error(chunk.error.message || '生成失败')
        controller.signal.throwIfAborted()
      }
      try {
        const response = await apiFetch('/api/chat', { method: 'POST', headers: { 'X-Modloader-Request-Id': assistant.id, 'X-Modloader-Call-Id': record.id }, body: JSON.stringify(body), signal: controller.signal })
        if (response.headers.get('content-type')?.includes('text/event-stream')) {
          const reader = response.body?.getReader()
          if (!reader) throw new Error('没有收到响应流。')
          const decoder = new TextDecoder()
          let buffer = '', doneEvent = false, finishEvent = false
          const consumeEvents = (final = false) => {
            const segments = buffer.replace(/\r\n/g, '\n').split('\n\n')
            buffer = final ? '' : segments.pop() || ''
            for (const event of segments) {
              const data = event.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
              if (!data) continue
              if (data.trim() === '[DONE]') { doneEvent = true; continue }
              const chunk = JSON.parse(data) as ChatChunk
              if (chunk.choices?.some(choice => choice.finish_reason)) finishEvent = true
              observe(chunk)
            }
          }
          try {
            while (!doneEvent) {
              const result = await reader.read()
              if (result.done) { buffer += decoder.decode(); consumeEvents(true); break }
              buffer += decoder.decode(result.value, { stream: true }); consumeEvents()
            }
          } finally { await reader.cancel().catch(() => undefined); reader.releaseLock() }
          if (!doneEvent && !finishEvent) throw new Error('响应流提前结束，已保留收到的内容。')
        } else observe(await response.json())
        record = { ...record, status: record.status === 'running' ? 'completed' : record.status }
        return lastChunk
      } catch (issue) {
        // A disconnected client cannot receive the terminal SSE frame. The
        // existing gateway slot can still return this exact call's known usage.
        if (!reported) {
          try {
            const report = await apiFetch(`/api/instances/${instance.id}/request-metrics/${record.id}`, { signal: AbortSignal.timeout(3000) }).then(response => response.json() as Promise<BackendRequestReport>)
            record = acceptRequestReport(record, report, controller)
            reported = true
          } catch { /* Keep already-reported usage; unavailable is not zero. */ }
        }
        record = { ...record, status: reported && record.status !== 'running' ? record.status : controller.signal.aborted ? 'cancelled' : record.status === 'incomplete' ? 'incomplete' : 'failed' }
        throw issue
      } finally {
        if (generation.observeLive === observeLive) generation.observeLive = undefined
        record = { ...record, endedAt: record.endedAt ?? Date.now(), elapsed: reported && record.elapsed !== undefined ? record.elapsed : (performance.now() - clock) / 1000 }
        // Legacy fields continue to describe the last individual model call.
        usage = record.usage; timings = record.timings; firstOutput = record.firstOutput
        setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, usage, timings, firstOutput, elapsed: record.elapsed } : message))
        store()
      }
    }
    const executeRequest = (body: Record<string, unknown>, kind: ModelRequestRecord['kind'], receive?: (chunk: ChatChunk) => void) => {
      pendingRequest = executeModelRequest(body, kind, receive)
      return pendingRequest
    }
    let compaction = validContextCompaction(sourceMessages, conversation.compaction)
    let apiMessages = buildChatContext([...sourceMessages, user], system, compaction, { imageReferences })
    const body = { model: instance.alias, instance_id: instance.id, ...reasoningRequestParameters(reasoningCapability, reasoningEffort, instance.backend), ...(temperatureField && temperature !== undefined ? { temperature } : {}) }
    const generate = async (request: WebModelRequest): Promise<WebCompletion> => {
      await history.flush(conversationId)
      controller.signal.throwIfAborted()
      const planning = request.phase === 'planning'
      responseStream = request.stream ?? stream
      const requestBody = { ...body, messages: request.messages, stream: responseStream,
        ...(instance.backend === 'ninfer' && request.messages.some(message => message.reasoning_content) ? { preserve_thinking: true } : {}),
        ...(responseStream && streamUsage ? { stream_options: { include_usage: true } } : {}),
        ...(!planning && maxTokensField && maxTokens !== undefined ? { max_tokens: maxTokens } : {}), ...(request.tools ? { tools: request.tools } : {}) }
      if (planning) {
        const chunk = await executeRequest(requestBody, 'planning')
        const choice = chunk.choices?.[0], message = choice?.message
        const raw = typeof message?.content === 'string' ? message.content : Array.isArray(message?.content) ? message.content.map(part => part.text || '').join('') : ''
        const parsed = splitThinking(raw, true)
        activity.capture(message?.reasoning_content ?? message?.reasoning ?? parsed.reasoning ?? '', '', [])
        setMessages(previous => previous.map(item => item.id === assistant.id ? { ...item, activity: activity.snapshot() } : item))
        return { message: { role: 'assistant', content: parsed.text || null, reasoning_content: message?.reasoning_content ?? message?.reasoning ?? parsed.reasoning, tool_calls: message?.tool_calls }, finishReason: choice?.finish_reason }
      }
      reasoningHistory = fullReasoning({ reasoningHistory, reasoning: reasoning || splitThinking(text, true).reasoning })
      text = ''; reasoning = ''; thinking = false; finished = false
      firstOutput = undefined; responseAt = undefined; finishReason = undefined
      toolCalls = []; usage = undefined; timings = undefined
      setLiveSample(null)
      setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, elapsed: undefined } : message))
      update(); started = performance.now()
      await executeRequest(requestBody, 'generation', consume)
      finished = true; update()
      if (!text && !reasoning && !toolCalls.length) throw new Error('模型返回了空响应，请检查加载配置与运行日志。')
      return { message: { role: 'assistant', content: text || null, ...(reasoning ? { reasoning_content: reasoning } : {}), ...(toolCalls.length ? { tool_calls: toolCalls } : {}) }, finishReason }
    }
    try {
      publishPhase(generation.phase)
      if (autoCompact && compactableMessageCount(sourceMessages) > (compaction?.messageRevisions.length || 0)) {
        let capacity = reportedNumber(matchingResponse?.contextCapacity) || (instance.profile_revision === profile?.revision ? reportedNumber(profile?.parameters.context_length) : undefined)
        const telemetry = await apiFetch(`/api/instances/${instance.id}/telemetry`, { signal: controller.signal }).then(response => response.json() as Promise<Telemetry>).catch(issue => { if (controller.signal.aborted) throw issue; return null })
        const observedCapacity = telemetry?.instance_id === instance.id ? reportedContextCapacity(telemetry) : null
        if (observedCapacity !== null) {
          generation.contextCapacity = observedCapacity; contextCapacities.current.set(instance.id, observedCapacity)
          assistant.contextCapacity = observedCapacity
        }
        capacity = observedCapacity ?? capacity
        const decision = autoCompactionDecision({ messages: sourceMessages, system, compaction, imageReferences, enabled: autoCompact, capacity, outputBudget: maxTokens ?? reportedNumber(profile?.generation.max_tokens),
          usageMessage: [...sourceMessages].reverse().find(message => message.role === 'assistant' && message.modelId === instance.model_id && message.profileId === instance.profile_id),
          nextMessages: buildChatContext([user], '', undefined, { imageReferences }), tools: [...(tools || []), ...builtInDefinitions] })
        if (decision.shouldCompact) {
          generation.phase = 'compacting'; publishPhase('compacting'); renderGeneration(value => value + 1)
          try { compaction = await compressContext({ ...conversation, messages: sourceMessages, compaction }, controller, 1, request => executeRequest(request, 'compaction')) }
          catch (issue) { if (controller.signal.aborted) throw issue; throw new Error(`自动压缩失败：${errorMessage(issue)}`) }
          apiMessages = buildChatContext([...sourceMessages, user], system, compaction, { imageReferences })
        }
      }
      controller.signal.throwIfAborted()
      // Pre-send compaction keeps the existing draft/error contract. Only an
      // accepted send installs the in-memory ledger on an assistant message.
      assistant.requests = requests; assistant.turn = totals()
      if (replacement) {
        await history.replaceMessages([...sourceMessages, user, assistant], conversationId)
        if (history.current.current.id === conversationId) { setEditing(null); setDialog(null) }
      } else {
        history.update({ messages: [...sourceMessages, user, assistant], draft: { text: '', images: [] } }, conversationId)
        if (history.current.current.id === conversationId) textarea.current?.resetSize()
      }
      sent = true
      generation.phase = webSearch && webSearchMode === 'retrieve' ? 'planning' : 'generating'
      publishPhase(generation.phase); renderGeneration(value => value + 1)
      const webRequests = new Map<symbol, string>()
      const updateWebProgress = () => {
        if (controller.signal.aborted) return
        generation.webProgress = generation.runningTools.size <= 1 && webRequests.size === 1 ? webRequests.values().next().value! : ''
        renderGeneration(value => value + 1)
      }
      const requestWeb = async <T,>(path: string, body: unknown) => {
        const id = Symbol()
        webRequests.set(id, '')
        updateWebProgress()
        try {
          return await requestProgress<T>('POST', path, body, { signal: controller.signal, onProgress: progress => {
            if (controller.signal.aborted || !webRequests.has(id) || progress.stage !== 'web_fallback') return
            webRequests.set(id, progress.message); updateWebProgress()
          } })
        } finally { webRequests.delete(id); updateWebProgress() }
      }
      if (webSearch || basicTools) await runWebChat({
        mode: webSearchMode, webEnabled: webSearch, messages: apiMessages, tools, readHistory: sourceMessages,
        ...(builtInDefinitions.length ? { builtInTools: { definitions: builtInDefinitions, execute: async (call: ToolCall) => {
          if (call.function.name === 'reverse_image_search') {
            const args = toolArguments(call)
            const image = searchableImages.find(image => image.id === args?.image_id)
            if (!image) throw new Error('未找到指定的对话图片。')
            const result = await searchReverseImage(image, controller.signal)
            return { ...result, image_name: image.name }
          }
          if (basicToolDefinitions.some(tool => tool.function.name === call.function.name)) return executeBasicTool(call)
          // New references are tied to the persisted conversation grant, not a renderer path.
          if (call.function.name === 'attach_file') await history.flush(conversationId)
          return fileTools.execute(call, toolDirectory, controller.signal, conversationId)
        } } } : {}),
        onToolStart: call => { generation.runningTools.add(call.id!); updateWebProgress(); activity.startTool(call); setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, activity: activity.snapshot() } : message)); saveMetrics() },
        onToolResult: (call, result) => {
          generation.runningTools.delete(call.id!); updateWebProgress()
          activity.finishTool(call, result)
          const label = toolMetricLabel(call.function.name, result)
          requests = requests.map(request => request.id === currentRequestId ? { ...request, labels: [...new Set([...request.labels.filter(value => call.function.name !== 'attach_file' || value !== '引用本地资源'), label])] } : request)
          const local = !['search_web', 'read_page', 'search_images'].includes(call.function.name)
          const resource = call.function.name === 'attach_file' ? localResourceFromResult(result) : undefined
          setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, activity: activity.snapshot(),
            ...(local ? { toolResults: [...(message.toolResults || []), { call, content: JSON.stringify(result) }] } : {}),
            ...(resource ? { resources: [...(message.resources || []).filter(item => item.id !== resource.id), resource] } : {}),
          } : message)); saveMetrics()
        }, maxCalls: webSearchRounds, toolConcurrency, signal: controller.signal, generate,
        search: query => requestWeb<WebSearchResult>('/api/web-search', { query }),
        searchImages: query => requestWeb<WebSearchResult>('/api/web-search', { query, category: 'images' }),
        readPage: args => requestWeb<WebReadResult>('/api/web-fetch', args),
        onPhase: phase => { generation.phase = phase; updateWebProgress(); publishPhase(phase) },
        onEvidence: result => setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, webSearch: result } : message)),
      })
      else await generate({ messages: apiMessages, tools, phase: 'answer' })
      turnStatus = 'completed'
      setMessages(previous => previous.map(message => message.id === assistant.id ? { ...message, status: 'completed' } : message))
    } catch (issue) {
      // Tool orchestration races model promises against abort for responsiveness.
      // Finish that model call's bounded receipt before freezing the turn ledger.
      await pendingRequest?.catch(() => undefined)
      const stopped = controller.signal.aborted
      if (sent) { activity.interrupt(stopped); finished = true; update() }
      turnStatus = stopped ? 'cancelled' : 'failed'
      const message = stopped ? '' : errorMessage(issue)
      if (sent) setMessages(previous => previous.map(item => item.id === assistant.id ? { ...item, status: stopped ? 'stopped' : 'failed', stopped, error: message || undefined } : item))
      if (!stopped && history.current.current.id === conversationId) setError(message)
    } finally {
      turnEndedAt = Date.now(); turnElapsed = (performance.now() - turnStarted) / 1000
      saveMetrics()
      try { await history.flush(conversationId) } catch { /* The unsaved state and manual retry remain visible. */ }
      release(); void props.refresh(); if (!replacement && history.current.current.id === conversationId) textarea.current?.focus()
    }
  }

  function rewrite(userId: string, text?: string) {
    if (transitioning || closing || attachmentBusy || !history.ready) return
    const conversationId = conversation.id
    lockedRef.current = true; onLockChange(true); setTransitioning(true)
    void (async () => {
      await stopGeneration(conversationId)
      const current = history.current.current.messages
      const index = current.findIndex(message => message.id === userId && message.role === 'user')
      if (index < 0) { setTransitioning(false); return }
      const task = send({ context: current.slice(0, index), user: { ...current[index], text: text === undefined ? current[index].text : text.trim(), ...(text === undefined ? {} : { createdAt: Date.now() }) } })
      const generation = generations.current.get(conversationId)
      if (generation) generation.done = task
      // The replacement now owns its snapshot; navigation can continue during generation.
      setTransitioning(false)
      await task
    })().catch(issue => { if (history.current.current.id === conversationId) setError(errorMessage(issue)); setTransitioning(false) })
  }

  function retry(messageId: string, confirmed = false) {
    const index = messages.findIndex(message => message.id === messageId)
    const user = messages.slice(0, index).reverse().find(message => message.role === 'user')
    if (!user) return
    if (!confirmed && index < messages.length - 1) { setRetryId(messageId); setDialog('retry'); return }
    rewrite(user.id)
  }
  function askDelete(ids: string[]) { setDeleteIds(ids); setDialog('delete') }

  async function exportConversation() {
    if (exporting || !history.ready || !history.current.current.messages.length) return
    const current = history.current.current
    setExporting(true)
    try {
      const result = await exportMarkdown(conversationMarkdown(current), current.title)
      if (!result.canceled) props.notify('已导出 Markdown')
    } catch (issue) { props.notify(`导出失败：${errorMessage(issue)}`, 'error') }
    finally { setExporting(false) }
  }

  function cancelRename() {
    if (renaming?.appliedTitle !== undefined && renaming.id === history.current.current.id && history.current.current.title === renaming.appliedTitle) history.update({ title: renaming.originalTitle })
    if (renaming?.appliedTitle !== undefined) setError('')
    setRenaming(null)
  }
  async function saveName() {
    if (!renaming || renaming.id !== history.current.current.id || !renaming.text.trim() || lockedRef.current) return
    const title = renaming.text.trim()
    lockedRef.current = true; onLockChange(true); setTransitioning(true)
    history.update({ title }); setRenaming({ ...renaming, appliedTitle: title })
    try { await history.flush(); setRenaming(null); setError('') }
    catch (issue) { setError(errorMessage(issue)) }
    finally { setTransitioning(false) }
  }

  async function transition(action: () => Promise<void>) {
    if (lockedRef.current) return
    cancelRename()
    lockedRef.current = true; onLockChange(true); setTransitioning(true)
    try { await action(); setError(''); setDialog(null); setEditing(null); shouldScroll.current = true; jumpTarget.current = null; setPromptIndex(-1) }
    catch (issue) { setError(errorMessage(issue)) }
    finally { setTransitioning(false) }
  }
  function promptNodes() { return Array.from(scrollContent.current?.querySelectorAll<HTMLElement>('.chat-message.user') || []) }
  function updateScrollState() {
    const node = scrollArea.current
    if (!node?.clientHeight) return
    setAtBottom(node.scrollHeight - node.scrollTop - node.clientHeight < 100)
  }
  function stopNavigationScroll() {
    if (navigationScroll.current !== null) {
      const area = scrollArea.current
      if (area) area.scrollTo({ top: area.scrollTop, behavior: 'instant' })
      navigationScroll.current = null
    }
  }
  function manualScroll() { stopNavigationScroll(); jumpTarget.current = null; setPromptIndex(-1) }
  function jumpPrompt(direction: number) {
    panels.finish()
    const nodes = promptNodes(), area = scrollArea.current
    if (!area || !nodes.length) return
    const bounds = area.getBoundingClientRect()
    let target = jumpTarget.current === null ? -1 : jumpTarget.current + direction
    if (jumpTarget.current === null) {
      if (direction < 0) nodes.forEach((node, index) => { if (node.getBoundingClientRect().top < bounds.bottom - 16) target = index })
      else target = nodes.findIndex(node => node.getBoundingClientRect().top > bounds.top + 32)
    }
    const index = Math.max(0, Math.min(nodes.length - 1, target))
    stopNavigationScroll()
    shouldScroll.current = false
    const top = Math.max(0, Math.min(area.scrollHeight - area.clientHeight, area.scrollTop + nodes[index].getBoundingClientRect().top - area.getBoundingClientRect().top - 16))
    const animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches && Math.abs(top - area.scrollTop) > 1
    navigationScroll.current = animate ? top : null
    area.scrollTo({ top, behavior: animate ? 'smooth' : 'instant' })
    lastScrollTop.current = area.scrollTop; jumpTarget.current = index; setPromptIndex(index); updateScrollState()
  }
  function jumpLatest() { panels.finish(); manualScroll(); shouldScroll.current = true; followOutput(); updateScrollState() }

  return <>

    <div ref={panels.layout} className="chat-layout">
      {fileDrag && <div className="chat-file-drop" role="status"><Upload size={28} /><strong>{inputLocked ? busy ? '正在生成，请稍后添加文件' : '请等待当前操作完成' : '松开以添加文件'}</strong></div>}
      <div className="chat-frame" style={panels.style} aria-hidden="true" />
      <div className={`chat-side-slot history-slot ${historyOpen ? 'is-open' : ''}`} style={panels.style} inert={!historyOpen} aria-hidden={!historyOpen}>
        <ChatHistoryPanel items={history.items} currentId={conversation.id} disabled={locked} reset={historyReset} onOpen={id => void transition(() => history.open(id))} onDelete={askDelete} onPin={(ids, pinned) => void transition(() => history.pin(ids, pinned))} onReorder={(id, before) => void transition(() => history.reorder(id, before))} />
      </div>
      <div className="chat-panel" style={panels.style}><div className="chat-toolbar">
        <IconButton icon={History} label="对话历史" aria-expanded={historyOpen} onClick={() => setHistoryOpen(!historyOpen)} />
        <div className="chat-model-state">{instance && <StateBadge state={instance.state} />}</div>
        <div className="inline-tags">{busy && !preparing && <LiveSpeed live={liveGeneration} error={liveError} />}<Button small icon={Plus} disabled={locked || !history.presetsReady} onClick={() => void transition(() => history.create(conversation.selection))}>新建会话</Button><IconButton icon={Settings2} label="对话参数" aria-expanded={optionsOpen} onClick={() => setOptionsOpen(!optionsOpen)} /></div>
      </div>
      <div className="chat-session-bar">{renaming?.id === conversation.id ? <div className="chat-session-rename" onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cancelRename() } }}><input ref={renameInput} autoFocus aria-label="会话名称" value={renaming.text} disabled={locked} onChange={event => setRenaming({ ...renaming, text: event.target.value })} onKeyDown={event => {
        if (event.key === 'Enter' && !event.nativeEvent.isComposing) { event.preventDefault(); void saveName() }
      }} /><IconButton icon={Check} label="保存名称" disabled={locked || !renaming.text.trim()} onClick={() => void saveName()} /><IconButton icon={X} label="取消重命名" disabled={locked} onClick={cancelRename} /></div> : <button className="chat-session-title" aria-label={conversation.title} data-tooltip="双击重命名会话" disabled={locked} onDoubleClick={() => setRenaming({ id: conversation.id, text: conversation.title, originalTitle: conversation.title })} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); setRenaming({ id: conversation.id, text: conversation.title, originalTitle: conversation.title }) } }}><ChatTitle text={conversation.title} /></button>}{(!history.ready || history.saveError || history.saveState !== 'saved' || messages.length > 0 || Boolean(input.trim()) || attachments.length > 0 || documents.length > 0) && <span className={`chat-save-status ${history.saveError ? 'is-error' : ''}`} role="status" data-tooltip={history.saveError || '生成期间约每秒保存当前回复，结束后立即保存。'}>{!history.ready ? history.loadError || history.presetError ? '读取失败' : '正在读取…' : history.saveError ? '未保存' : busy ? '自动保存' : history.saveState === 'saved' ? '已保存' : '保存中…'}</span>}{history.saveError && <button className="chat-retry" disabled={transitioning || closing} onClick={() => void history.flush().catch(() => {})}>重试保存</button>}
        <div className="chat-transcript-actions"><IconButton icon={ChevronUp} label="上一条提问" disabled={!messages.some(message => message.role === 'user') || promptIndex === 0} onClick={() => jumpPrompt(-1)} /><IconButton icon={ChevronDown} label="下一条提问" disabled={!messages.some(message => message.role === 'user') || promptIndex >= messages.filter(message => message.role === 'user').length - 1} onClick={() => jumpPrompt(1)} /><IconButton icon={ArrowDown} label="返回最新" disabled={atBottom} onClick={jumpLatest} /><IconButton icon={Download} label="导出对话为 Markdown" disabled={exporting || transitioning || closing || !history.ready || !messages.length} onClick={() => void exportConversation()} /><IconButton icon={Eraser} label="清空当前对话" disabled={locked || !messages.length} onClick={() => setDialog('clear')} /><IconButton icon={Trash2} label="删除会话" className="danger" disabled={locked} onClick={() => askDelete([conversation.id])} /></div>
      </div>
      {history.loadError && <div className="chat-load-error"><ErrorNotice message={`无法读取对话数据：${history.loadError}`} /><Button small onClick={() => void history.reload()}>重新读取</Button></div>}
      {history.presetError && <div className="chat-load-error"><ErrorNotice message={history.presetError} /><Button small onClick={() => void history.reloadPresets()}>重新读取预设</Button></div>}
        <div className="chat-messages" ref={scrollArea} tabIndex={0} role="region" aria-label="对话内容" onWheel={manualScroll} onPointerDown={manualScroll} onKeyDown={manualScroll} onScroll={event => {
          const node = event.currentTarget
          if (!node.clientHeight) return
          if (navigationScroll.current === null) {
            if (node.scrollHeight - node.scrollTop - node.clientHeight < 100) shouldScroll.current = true
            // A queued scroll from the previous chunk must not disable following
            // just because the next chunk has already extended the content.
            else if (node.scrollTop < lastScrollTop.current) shouldScroll.current = false
          }
          lastScrollTop.current = node.scrollTop
          updateScrollState()
        }}>
          {!messages.length && <div className="chat-empty"><span className="chat-empty-mark"><MessageSquare size={32} strokeWidth={1.5} /></span><h2>{available ? '开始对话' : '先加载一个模型'}</h2>{!available && <Button small icon={ArrowUp} onClick={() => props.navigate('models')}>前往模型库</Button>}{available && <div className="starter-prompts">{['用两句话介绍你自己', '解释一下什么是上下文窗口', '写一个 Python 快速排序'].map(prompt => <button key={prompt} onClick={() => { setInput(prompt); textarea.current?.focus() }}>{prompt}<ArrowUp size={13} /></button>)}</div>}</div>}
          <div className="chat-messages-content" ref={scrollContent}>{messages.map(message => {
              const generating = busy && message.id === activeMessageId
              const reasoning = fullReasoning(message)
              return <div className={`chat-message ${message.role}`} key={message.id} data-message-id={message.id}>
                <div className="message-avatar" aria-hidden="true">{message.role === 'user' ? <UserRound size={17} /> : <Sparkles size={17} />}</div>
                <ImageGallery className="message-body">
                  <div className="message-label"><strong>{message.role === 'user' ? '你' : message.model}</strong><MessageTime message={message} />{message.role === 'assistant' && <span>{generating ? <><Loader2 size={12} className="spin" />{phaseLabel || (message.thinking ? '思考中' : '生成中')}</> : message.status === 'interrupted' ? '已中断' : message.stopped ? '已停止' : message.error ? '生成失败' : ''}</span>}</div>
                  {!!message.images.length && <div className="message-images">{message.images.map(image => <GalleryImage key={image.id} src={image.url} alt={image.name} />)}</div>}
                  <ChatDocuments documents={message.documents} />
                  <ChatActivity message={message} generating={generating} detail={props.state.settings.chat_tool_details || 'compact'} />
                  {editing?.id === message.id ? <div className="message-editor">
                    <MarkdownInput key={editing.id} ref={editingTextarea} autoFocus aria-label="编辑消息内容" previewLabel="编辑消息预览" resizeLabel="调整消息编辑框高度" value={editing.text} onChange={event => setEditing({ ...editing, text: event.target.value })} disabled={transitioning} onKeyDown={event => { if (event.key === 'Escape') { event.stopPropagation(); setEditing(null) } }} />
                    <div className="message-editor-actions"><div className="message-editor-tools">
                      <IconButton icon={Copy} label="复制编辑原文" disabled={!editing.text} onClick={() => copyOriginal(editing.text)} />
                    </div><Button small disabled={transitioning} onClick={() => setEditing(null)}>取消编辑</Button><Button small variant="primary" disabled={transitioning || !available || (!editing.text.trim() && !message.images.length && !message.documents?.length)} onClick={() => rewrite(message.id, editing.text)}>重新发送</Button></div>
                  </div> : (message.text || message.resources?.length) && !message.tools?.length && <div className="message-text"><MarkdownContent text={message.text} streaming={generating} resources={message.resources} directory={toolDirectory} /></div>}
                  {!!message.tools?.length && <ChatLocalResources resources={message.resources} directory={toolDirectory} />}
                  {message.role === 'user' && editing?.id !== message.id && <div className="message-footer user-actions">{message.text && <IconButton icon={Copy} label="复制消息" onClick={() => void copy(message.text).then(() => props.notify('消息已复制')).catch(issue => props.notify(errorMessage(issue), 'error'))} />}<IconButton icon={Pencil} label="编辑消息" disabled={transitioning || closing || attachmentBusy} onClick={() => setEditing({ id: message.id, text: message.text })} /></div>}
                  {generating && !message.text && !reasoning && !message.tools?.length && <span className="thinking-dots"><i /><i /><i /></span>}
                  {!generating && message.finishReason === 'length' && <div className="message-limit">已达到输出 Token 上限{!message.text && reasoning ? '，模型尚未生成最终回答' : ''}。</div>}
                  {message.webSearch && <WebSearchSources result={message.webSearch} />}
                  {message.error && <div className="message-error">{message.error}</div>}
                  {message.role === 'assistant' && <div className="message-footer">
                    {message.text && <IconButton icon={Copy} label="复制模型回复" onClick={() => void copy(message.text).then(() => props.notify('模型回复已复制')).catch(issue => props.notify(errorMessage(issue), 'error'))} />}
                    <IconButton icon={RotateCcw} label="重试回复" disabled={transitioning || closing || attachmentBusy || !available} onClick={() => retry(message.id)} />
                    <MessageMetrics message={message} generating={generating} started={generating ? generationStarted : undefined} live={generating ? liveGeneration : undefined} liveError={liveError} onInspect={() => onInspectTurn(message)} />
                  </div>}
                </ImageGallery>
              </div>
            })}</div>
        </div>
        <div className="chat-composer-area">{activeGeneration?.phase === 'compacting' && <div className="chat-compaction-status" role="status"><Loader2 size={14} className="spin" />正在压缩上下文</div>}{error && <ErrorNotice message={error} />}{incompatibleImages && <div className="warning-note">{contextHasImages ? '这段对话的历史包含图片，请切换到支持图片的模型，或新建纯文本会话。' : '当前模型未启用图片输入，请移除附件或切换到支持图片的加载配置。'}</div>}{attachments.length > 0 && <ImageGallery className="attachment-list">{attachments.map(image => <div className="attachment" key={image.id}><GalleryImage src={image.url} alt={image.name} /><div><strong>{image.name}</strong><span>{formatBytes(image.size)}</span></div><IconButton icon={X} label={`移除 ${image.name}`} disabled={locked} onClick={() => setAttachments(previous => previous.filter(item => item.id !== image.id))} /></div>)}</ImageGallery>}
          <ChatDocuments documents={documents} disabled={locked} onRemove={id => history.update({ draft: { ...history.current.current.draft, documents: documents.filter(document => document.id !== id) } })} />
          {attachmentBusy && <div className="chat-attachment-status" role="status"><Loader2 size={14} className="spin" />正在读取附件…</div>}
          <div className="chat-composer" onPaste={pasteImages}>
            <MarkdownInput key={conversation.id} ref={textarea} aria-label="对话消息" previewLabel="输入内容预览" resizeLabel="调整输入框高度" value={input} placeholder={available ? '输入消息…' : '模型未就绪'} rows={2} disabled={inputLocked} onChange={event => setInput(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); launchSend() } }} />
            <div className={`composer-actions${busy ? ' is-generating' : ''}`}><div className="composer-attachments"><IconButton icon={Paperclip} label="添加附件" disabled={inputLocked} onClick={() => fileInput.current?.click()} />
              <IconButton icon={Copy} label="复制输入原文" disabled={!input} onClick={() => copyOriginal(input)} />
              <ChatSearchControl enabled={webSearch} mode={webSearchMode} disabled={inputLocked} onChange={setWebSearchMode} />
              <button type="button" className="web-search-toggle composer-tool-icon" aria-label="工具调用" aria-pressed={basicTools} data-tooltip={`工具调用 · ${basicTools ? toolDirectory ? '基础工具与本地文件' : '计算与时间' : '关闭'}`} disabled={inputLocked} onClick={() => changeSetting({ basicTools: !basicTools })}><Wrench size={17} /></button>
              {hasImages && <span>{attachments.length} 张图片</span>}</div><div className="composer-context-controls"><button type="button" className="web-search-toggle composer-tool-icon" aria-label="压缩上下文" data-tooltip={conversation.compaction ? '上下文已压缩 · 点击再次压缩' : '压缩上下文'} data-compacted={Boolean(conversation.compaction)} disabled={inputLocked || !available || !messages.some(message => message.text && !message.error)} onClick={launchCompaction}><ChevronsDownUp size={17} /></button><ContextUsage used={contextUsed} capacity={contextCapacity} detail={busy ? liveGeneration?.context_basis || liveError || undefined : completedContext !== undefined ? '上次请求 · 输入 + 输出' : undefined} />
              <div className="chat-model-controls"><ChatReasoningControl capability={reasoningCapability} value={reasoningEffort} defaultLabel={reasoningDefaultLabel(reasoningCapability, profile)} disabled={inputLocked} resetKey={conversation.id} modelId={instanceId} modelName={instance?.alias || '选择或加载模型'} modelOptions={modelOptions} quickLoadOptions={quickLoadOptions} onLoadProfile={(id, parallel) => void loadProfile(id, parallel)} onUnloadModel={id => void unloadInstance(id)} unloadingIds={unloadingIds} unloadError={unloadError} onModelChange={id => { cancelAutoSelect(); selectInstance(id) }} onChange={reasoningEffort => changeSetting({ reasoningEffort })} />
              </div></div>{busy ? <Button icon={Square} onClick={() => activeGeneration?.controller.abort()}>{webSearch || basicTools || activeGeneration?.phase === 'compacting' ? '停止' : '停止生成'}</Button> : <button className="send-button" data-tooltip="发送消息" aria-label="发送消息" disabled={inputLocked || !available || (!input.trim() && !attachments.length && !documents.length) || incompatibleImages} onClick={() => launchSend()}><ArrowUp size={19} /></button>}</div>
          </div><input ref={fileInput} type="file" multiple hidden onChange={event => { if (event.target.files) void addFiles(Array.from(event.target.files)); event.target.value = '' }} /></div>
      </div><div className={`chat-side-slot options-slot ${optionsOpen ? 'is-open' : ''}`} style={panels.style} inert={!optionsOpen} aria-hidden={!optionsOpen}><aside className="chat-options" aria-label="对话参数"><div className="panel-heading"><div><h3>对话参数</h3></div><IconButton icon={X} label="收起对话参数" onClick={() => setOptionsOpen(false)} /></div><div className="chat-options-body"><ChatPresetPicker history={history} disabled={locked} notify={props.notify} /><Field label="系统提示词"><textarea rows={5} value={system} disabled={locked} onChange={event => setSystem(event.target.value)} placeholder="可选" /></Field><>{temperatureField && <NumericParameter field={temperatureField} disabled={locked} value={temperature} onChange={value => { setTemperature(value); setError('') }} placeholder={generationDefaultText(temperatureField, generationDefaults)} />}{maxTokensField && <NumericParameter field={maxTokensField} disabled={locked} value={maxTokens} onChange={value => { setMaxTokens(value); setError('') }} placeholder={generationDefaultText(maxTokensField, generationDefaults)} />}</><label className="check-option"><input type="checkbox" checked={stream} disabled={locked} onChange={event => setStream(event.target.checked)} /><strong>流式输出</strong></label><Field label="工具调用预算"><NumberInput aria-label="工具调用预算" min={1} step={1} value={webSearchRounds} placeholder={`默认 ${DEFAULT_WEB_SEARCH_SETTINGS.webSearchRounds} 次`} disabled={inputLocked} onValueChange={value => changeSetting({ webSearchRounds: value === '' ? DEFAULT_WEB_SEARCH_SETTINGS.webSearchRounds : Number(value) })} /></Field><Field label="工具并行数"><NumberInput aria-label="工具并行数" min={1} step={1} value={toolConcurrency} placeholder={`默认 ${DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency} 路`} disabled={inputLocked} onValueChange={value => changeSetting({ toolConcurrency: value === '' ? DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency : Number(value) })} /></Field><label className="check-option"><input type="checkbox" checked={autoCompact} disabled={inputLocked} onChange={event => changeSetting({ autoCompact: event.target.checked })} /><strong>自动压缩上下文</strong></label><Field label="本地工具目录" help="仅读写此目录，写入前确认。"><div className="input-with-button"><input aria-label="本地工具目录" value={toolDirectory} readOnly placeholder="未授权" /><IconButton icon={FolderOpen} label="选择本地工具目录" disabled={inputLocked} onClick={() => void pickPath('directory').then(path => { if (path) return changeToolDirectory(path) }).catch(issue => setError(errorMessage(issue)))} />{toolDirectory && <IconButton icon={X} label="撤销本地目录授权" disabled={inputLocked} onClick={() => void changeToolDirectory('').catch(issue => setError(errorMessage(issue)))} />}</div></Field>{conversation.compaction && <details className="chat-context-summary"><summary>上下文摘要</summary><p>{conversation.compaction.summary}</p><Button small disabled={inputLocked} onClick={() => history.update({ compaction: null })}>恢复完整上下文</Button></details>}<Field label="工具定义"><textarea rows={5} className="mono" value={toolsDraft} disabled={locked} onChange={event => setToolsDraft(event.target.value)} placeholder="OpenAI tools JSON" /></Field></div></aside></div></div>
    {fileTools.dialog}
    {dialog && <Modal title={dialog === 'clear' ? '清空当前对话？' : dialog === 'retry' ? '从这条回复重新生成？' : deleteIds.length > 1 ? `删除 ${deleteIds.length} 条会话？` : '删除会话？'} description={dialog === 'clear' ? '清除全部消息和标题，保留参数与草稿。此操作不可撤销。' : dialog === 'delete' ? '删除所选会话及其消息、图片和草稿。此操作不可撤销。' : dialog === 'retry' ? '将清除这条回复及其后的消息，使用原提问重新生成。' : undefined} onClose={() => { if (!transitioning) setDialog(null) }} footer={<><Button disabled={transitioning} onClick={() => setDialog(null)}>取消</Button><Button variant={dialog === 'retry' ? 'primary' : 'danger'} busy={transitioning} onClick={() => {
      if (dialog === 'retry') { retry(retryId, true); return }
      void transition(async () => {
        if (dialog === 'clear') { await stopGeneration(conversation.id); await history.clear() }
        else { await Promise.all(deleteIds.map(stopGeneration)); await history.remove(deleteIds); setHistoryReset(value => value + 1) }
      })
    }}>{dialog === 'clear' ? '清空对话' : dialog === 'retry' ? '重新生成' : '删除会话'}</Button></>}>
      {dialog === 'delete' ? <div className="chat-confirm-title">{deleteIds.map(id => <p key={id}>{history.items.find(item => item.id === id)?.title || conversation.title}</p>)}</div> : <p className="chat-confirm-title">{conversation.title}</p>}{error && <ErrorNotice message={error} />}
    </Modal>}
  </>
}

