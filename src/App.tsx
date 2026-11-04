import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertCircle, ArrowUpRight, CheckCircle2, CircleHelp, Cpu, Download, Layers3, MessageSquare, RefreshCw, Search, Settings2, Square, Terminal, Pin, PinOff, X } from 'lucide-react'
import { ApiError, copy, errorMessage, fetchState, isActive, isJobActive, isReady, request, REFRESH_INTERVAL_MS, resetConnection } from './api'
import type { AppState, RuntimeInstance } from './types'
import { Badge, Button, IconButton, Modal, StateBadge, formatBytes } from './components'
import Models from './pages/Models'
import Discovery from './pages/Discovery'
import Runtimes from './pages/Runtimes'
import { modelSizeKnown } from './platforms'
import Chat from './pages/Chat'
import Settings from './pages/Settings'
import Monitor from './pages/Monitor'
import Tooltip from './Tooltip'
import outfitLicense from './assets/fonts/OFL.txt?raw'
import appIcon from '../electron/icon.png'
import './header.css'
import { useHeaderMotion } from './useHeaderMotion'
import { useMonitorState } from './useMonitorState'
import { miniSnapshot, useMiniMonitorPublisher } from './mini-monitor'
import { generationEventIdentity } from './monitor-generation-events'

const initialState: AppState = { models: [], profiles: [], runtimes: [], instances: [], downloads: [], installations: [], settings: {}, backends: [], hardware: {} }
const navigation = [
  { id: 'models', label: '模型库', icon: Layers3 },
  { id: 'discover', label: '发现模型', icon: Search },
  { id: 'runtimes', label: '运行环境', icon: Cpu },
  { id: 'monitor', label: '运行监控', icon: Activity },
  { id: 'chat', label: '测试对话', icon: MessageSquare },
  { id: 'settings', label: '设置', icon: Settings2 },
]
export default function App() {
  const [page, setPage] = useState('models')
  const pageScroll = useRef<Record<string, number>>({})
  const [state, setState] = useState(initialState)
  const [connected, setConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [connectionError, setConnectionError] = useState('')
  const [toast, setToast] = useState<{ message: string; tone: string; id: number } | null>(null)
  const [forceStop, setForceStop] = useState<RuntimeInstance | null>(null)
  const [stopBusy, setStopBusy] = useState(false)
  const [logs, setLogs] = useState<RuntimeInstance | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const [logError, setLogError] = useState('')
  const [chatInstance, setChatInstance] = useState<{ id: string; request: number } | undefined>()
  const chatLocked = useRef(false)
  const chatRequest = useRef(0)
  const onChatLockChange = useCallback((value: boolean) => { chatLocked.current = value }, [])
  const [about, setAbout] = useState(false)
  const [headerPinned, setHeaderPinned] = useState(() => {
    try { return localStorage.getItem('modloader.header.pinned') === 'true' } catch { return false }
  })
  const [headerHovered, setHeaderHovered] = useState(false)
  const [headerFocused, setHeaderFocused] = useState(false)
  const headerOpen = headerPinned || headerHovered || headerFocused
  const shell = useRef<HTMLDivElement>(null)
  const header = useRef<HTMLElement>(null)
  const headerBrand = useRef<HTMLButtonElement>(null)
  const captureHeaderMotion = useHeaderMotion(header, headerOpen)
  const inFlight = useRef<Promise<void> | null>(null)
  const mounted = useRef(true)
  const fontScale = Number(state.settings.font_scale) || 1
  useEffect(() => { document.documentElement.style.setProperty('--font-scale', String(fontScale)) }, [fontScale])
  useEffect(() => {
    try { localStorage.setItem('modloader.header.pinned', String(headerPinned)) } catch { /* The choice still works for this session. */ }
  }, [headerPinned])
  useLayoutEffect(() => {
    window.scrollTo(0, pageScroll.current[page] || 0)
    const rememberScroll = () => { pageScroll.current[page] = window.scrollY }
    window.addEventListener('scroll', rememberScroll, { passive: true })
    return () => window.removeEventListener('scroll', rememberScroll)
  }, [page])
  useLayoutEffect(() => {
    const node = header.current, container = shell.current
    if (!node || !container) return
    const measure = () => {
      const height = headerPinned ? Math.ceil(node.getBoundingClientRect().height) : parseFloat(getComputedStyle(node).getPropertyValue('--header-compact-height'))
      container.style.setProperty('--app-header-height', `${height}px`)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(node)
    return () => observer.disconnect()
  }, [headerPinned])
  function closeHeader() {
    if (headerPinned) return
    captureHeaderMotion()
    headerBrand.current?.focus({ preventScroll: true })
    setHeaderHovered(false)
    setHeaderFocused(false)
  }
  useEffect(() => {
    if (!headerOpen || headerPinned) return
    const outside = (event: PointerEvent) => {
      if (!header.current?.contains(event.target as Node)) { captureHeaderMotion(); setHeaderHovered(false); setHeaderFocused(false) }
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [headerOpen, headerPinned])

  const refresh = useCallback(async () => {
    if (inFlight.current) return inFlight.current
    const task = (async () => {
      try {
        const next = await fetchState()
        if (mounted.current) { setState(next); setConnected(true); setConnectionError('') }
      } catch (error) {
        if (mounted.current) { setConnected(false); setConnectionError(errorMessage(error)) }
      } finally { if (mounted.current) setLoading(false) }
    })()
    inFlight.current = task
    try { await task } finally { inFlight.current = null }
  }, [])

  const monitor = useMonitorState(state, refresh)
  useEffect(() => { if (page === 'chat') monitor.followLatest() }, [page, monitor.followLatest])
  const monitorSnapshot = useMemo(() => miniSnapshot(state, monitor, connected),
    [state, connected, monitor.payload, monitor.sampleError, monitor.instanceId, monitor.device, monitor.memory])
  const latestMonitorSnapshot = useRef(monitorSnapshot)
  latestMonitorSnapshot.current = monitorSnapshot
  useMiniMonitorPublisher(monitorSnapshot, monitor.generationEvents, generationEventIdentity(monitor.liveGeneration.requests.at(-1), monitor.generationEvents), monitor.liveGeneration.requests.at(-1))
  useEffect(() => window.modloader?.onMiniMonitorAction?.(action => {
    const current = latestMonitorSnapshot.current
    if (action.type === 'instance' && current.instances.some(item => item.id === action.id)) monitor.setInstanceId(action.id)
    if (action.type === 'gpu' && current.gpus.some(item => item.id === action.id)) monitor.setDevice(action.id)
  }), [monitor.setInstanceId, monitor.setDevice])

  useEffect(() => {
    mounted.current = true
    void refresh()
    const timer = window.setInterval(() => { void refresh() }, REFRESH_INTERVAL_MS)
    return () => { mounted.current = false; window.clearInterval(timer) }
  }, [refresh])

  const notify = useCallback((message: string, tone: 'success' | 'error' | 'info' = 'success') => setToast({ message, tone, id: Date.now() }), [])
  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), toast.tone === 'error' ? 8000 : 4000)
    return () => window.clearTimeout(timer)
  }, [toast])

  async function stopInstance(instance: RuntimeInstance) {
    try {
      await request('POST', `/api/instances/${instance.id}/stop`, {})
      await refresh()
      notify('已请求停止模型')
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) setForceStop(instance)
      else notify(errorMessage(error), 'error')
    }
  }

  async function confirmStop() {
    if (!forceStop) return
    setStopBusy(true)
    try {
      await request('POST', `/api/instances/${forceStop.id}/stop`, { force: true })
      setForceStop(null); await refresh(); notify('已请求强制停止')
    } catch (error) { notify(errorMessage(error), 'error') }
    finally { setStopBusy(false) }
  }

  useEffect(() => {
    if (!logs) return
    let cancelled = false
    async function load() {
      try {
        const result = await request<{ lines: string[] }>('GET', `/api/instances/${logs!.id}/logs`)
        if (!cancelled) { setLogLines(result.lines || []); setLogError('') }
      } catch (error) { if (!cancelled) setLogError(errorMessage(error)) }
    }
    void load()
    const timer = window.setInterval(() => { void load() }, 2000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [logs])

  const active = state.instances.filter(instance => isActive(instance.state))
  const ready = active.filter(instance => isReady(instance.state))
  const downloadCount = state.downloads.filter(isJobActive).length
  const installationCount = state.installations.filter(isJobActive).length
  const migrationCount = (state.migrations || []).filter(isJobActive).length
  const taskCounts: Record<string, number> = { models: migrationCount, discover: downloadCount, runtimes: installationCount }
  const props = { state, refresh, notify, stopInstance, openLogs: (instance: RuntimeInstance) => { setLogLines([]); setLogs(instance) }, navigate: (target: string, instanceId?: string) => { setPage(target); if (instanceId && target === 'chat') { if (chatLocked.current) notify('请先停止生成或等待当前操作完成，再切换模型。', 'info'); else setChatInstance({ id: instanceId, request: ++chatRequest.current }); } if (instanceId && target === 'monitor') monitor.setInstanceId(instanceId) } }
  const unknownSizeCount = state.models.filter(model => !modelSizeKnown(model)).length
  const totalSize = state.models.reduce((sum, model) => sum + (modelSizeKnown(model) ? model.size_bytes || 0 : 0), 0)
  const modelSizeSummary = unknownSizeCount ? unknownSizeCount === state.models.length ? '模型大小未统计' : `${formatBytes(totalSize)} 已知 · ${unknownSizeCount} 项未统计` : `${formatBytes(totalSize)} 模型文件`
  const serviceStatus = loading ? '连接中' : connected ? '服务在线' : '服务离线'

  return <div ref={shell} className={`app-shell ${fontScale > 1.1 ? 'font-expanded' : ''} ${headerPinned ? 'header-pinned' : 'header-floating'} ${page === 'chat' ? 'chat-active' : ''}`}>
    <div className="topbar-space">
      <header ref={header} className={`app-header ${headerOpen ? 'is-open' : ''}`} onPointerEnter={event => {
        if (event.pointerType === 'mouse' && window.matchMedia('(hover: hover) and (pointer: fine)').matches) { captureHeaderMotion(); setHeaderHovered(true) }
      }} onPointerLeave={event => { if (event.pointerType === 'mouse') { captureHeaderMotion(); setHeaderHovered(false) } }} onFocusCapture={event => {
        if (event.target.matches(':focus-visible')) { captureHeaderMotion(); setHeaderFocused(true) }
      }} onBlurCapture={event => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) { captureHeaderMotion(); setHeaderFocused(false) }
      }} onKeyDown={event => { if (event.key === 'Escape' && !headerPinned) { event.preventDefault(); closeHeader() } }}>
        <div id="header-panel" className="header-panel">
          <div className="topbar-inner">
            <button ref={headerBrand} className="wordmark" aria-label="ModLoader 模型库" title="ModLoader 模型库" onClick={() => { captureHeaderMotion(); setHeaderHovered(true); setPage('models') }}><span className="brand-mark" aria-hidden="true"><img src={appIcon} alt="" draggable={false} /></span><span className="wordmark-text">ModLoader<span className="brand-period">.</span></span></button>
            <nav className="navigation" aria-label="主导航">{navigation.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} aria-label={item.label} title={item.label} aria-current={page === item.id ? 'page' : undefined} onClick={() => setPage(item.id)}><span className="nav-surface" aria-hidden="true" /><span className="nav-icon"><item.icon size={16} /></span><span className="nav-label">{item.label}</span>{taskCounts[item.id] > 0 && <span className="nav-count">{taskCounts[item.id]}</span>}</button>)}</nav>
            <div className="topbar-end"><span className="header-service" role="status" aria-label={serviceStatus} title={serviceStatus}><Badge dot tone={loading ? 'neutral' : connected ? 'green' : 'red'}><span className="service-label">{serviceStatus}</span></Badge></span><span className="header-extra-actions" aria-hidden={!headerOpen} inert={!headerOpen}><IconButton label="刷新状态" icon={RefreshCw} onClick={() => { resetConnection(); void refresh() }} /><IconButton label="关于 ModLoader" icon={CircleHelp} onClick={() => setAbout(true)} /></span><IconButton label={headerPinned ? '取消钉住顶部栏' : '钉住顶部栏'} aria-pressed={headerPinned} icon={headerPinned ? PinOff : Pin} onClick={() => { captureHeaderMotion(); setHeaderPinned(previous => !previous) }} /></div>
          </div>
          <div className="workspace-summary" aria-hidden={!headerOpen} inert={!headerOpen}><div className="workspace-label"><span className="workspace-orbit" />本地工作台</div><div className="summary-metrics"><span><strong>{state.models.length}</strong> 个模型</span><span><strong>{ready.length}</strong> 运行中</span><span>{modelSizeSummary}</span>{downloadCount > 0 && <button onClick={() => setPage('discover')}><Download size={13} />{downloadCount} 项下载</button>}{installationCount > 0 && <button onClick={() => setPage('runtimes')}><Cpu size={13} />{installationCount} 项安装</button>}</div><div className="api-address"><span className={`tiny-dot ${ready.length ? 'green' : ''}`} /><span>127.0.0.1:{state.settings.api_port || 12340}</span><IconButton icon={ArrowUpRight} label="查看 API 接入设置" onClick={() => setPage('settings')} /></div></div>
        </div>
      </header>
    </div>
    {!connected && !loading && <div className="connection-banner"><AlertCircle size={17} /><span>{connectionError}</span><Button small onClick={() => { resetConnection(); void refresh() }}>重新连接</Button></div>}
    <main className="workspace" aria-busy={loading}>
      <section hidden={page !== 'models'}><Models {...props} active={page === 'models'} /></section>
      <section hidden={page !== 'discover'}><Discovery {...props} /></section>
      <section hidden={page !== 'runtimes'}><Runtimes {...props} active={page === 'runtimes'} /></section>
      <section hidden={page !== 'monitor'}><Monitor {...props} monitor={monitor} active={page === 'monitor'} /></section>
      <section hidden={page !== 'chat'} className="chat-section"><Chat {...props} active={page === 'chat'} selectedInstance={chatInstance} knownContextCapacity={monitor.knownContextCapacity} onLockChange={onChatLockChange} onGenerationActivity={monitor.reportChatActivity} onInspectTurn={message => {
        const matching = state.instances.find(instance => instance.id === message.instanceId) ?? state.instances.find(instance => instance.profile_id === message.profileId && instance.model_id === message.modelId)
        monitor.showChatTurn({ instanceId: message.instanceId || '', messageId: message.id, model: message.model, contextCapacity: message.contextCapacity,
          phase: message.status === 'generating' ? 'generating' : message.status === 'completed' ? 'completed' : message.status === 'failed' ? 'failed' : 'cancelled',
          requests: message.requests, turn: message.turn, toolActivity: message.activity?.filter((event): event is Extract<typeof event, { kind: 'tool' }> => event.kind === 'tool'),
          toolCount: message.activity?.filter(event => event.kind === 'tool' && event.startedAt !== undefined).length }, matching?.id)
        setPage('monitor')
      }} /></section>
      <section hidden={page !== 'settings'}><Settings {...props} active={page === 'settings'} /></section>
    </main>
    {toast && <div className={`toast ${toast.tone}`} role={toast.tone === 'error' ? 'alert' : 'status'}>{toast.tone === 'error' ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}<span>{toast.message}</span><IconButton icon={X} label="关闭通知" onClick={() => setToast(null)} /></div>}
    {forceStop && <Modal title="停止正在处理请求的模型？" description={`${forceStop.alias} 仍有请求正在处理，强制停止会中断这些请求。`} onClose={() => { if (!stopBusy) setForceStop(null) }} footer={<><Button onClick={() => setForceStop(null)} disabled={stopBusy}>继续运行</Button><Button variant="danger" icon={Square} busy={stopBusy} onClick={() => void confirmStop()}>中断请求并停止</Button></>}><div className="confirmation-detail"><StateBadge state={forceStop.state} /><span>{forceStop.active_requests} 个活动请求</span></div></Modal>}
    {logs && <Modal title={`${logs.alias} · 运行日志`} onClose={() => setLogs(null)} wide footer={<><Button icon={Terminal} onClick={() => { void copy(logLines.join('\n')).then(() => notify('日志已复制')).catch(error => notify(errorMessage(error), 'error')) }}>复制日志</Button><Button onClick={() => setLogs(null)}>关闭</Button></>}><div className="log-status"><StateBadge state={state.instances.find(instance => instance.id === logs.id)?.state || logs.state} /><span>PID {logs.pid || '—'}<span className="text-divider" />端口 {logs.port}</span></div>{logError && <div className="error-notice">{logError}</div>}<pre className="log-view">{logLines.length ? logLines.join('\n') : '暂无日志，等待运行环境输出。'}</pre></Modal>}
    {about && <Modal title="ModLoader" description="本地模型工作台" onClose={() => setAbout(false)} footer={<Button onClick={() => setAbout(false)}>关闭</Button>}><div className="about-layout"><span className="about-brand" aria-hidden="true"><img src={appIcon} alt="" draggable={false} /></span><div><p>管理模型文件、加载配置和运行环境，在统一的本地 API 中使用已加载的模型。</p><p className="muted">界面字体 Outfit · SIL Open Font License 1.1</p></div></div><details className="font-license"><summary>字体许可</summary><pre>{outfitLicense}</pre></details></Modal>}
    <Tooltip />
  </div>
}
