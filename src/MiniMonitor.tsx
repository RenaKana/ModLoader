import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent } from 'react'
import { ArrowLeft, Check, ChevronDown, GripVertical, Minimize2, Pin, PinOff, SlidersHorizontal, Thermometer, X } from 'lucide-react'
import { defaultMiniMetrics, miniMetrics, MINI_PREFERENCE_KEY, moveMiniMetric, readMiniSelection,
  type MiniMetricId, type MiniMonitorSnapshot, type MiniReadout } from './mini-monitor'
import { reorderLayoutRect, useReorderMotion } from './useReorderMotion'
import './mini-monitor.css'

type Drag = { id: MiniMetricId; x: number; y: number; offsetX: number; offsetY: number; width: number; height: number; order: MiniMetricId[] }
type Press = { id: MiniMetricId; x: number; y: number; pointer: number; box: DOMRect }
const emptyReadout: MiniReadout = { value: '—', unit: '' }

export default function MiniMonitor() {
  const bridge = window.miniMonitor
  const [snapshot, setSnapshot] = useState<MiniMonitorSnapshot | null>(null)
  const [pinned, setPinned] = useState(true)
  const [opacity, setOpacity] = useState(1)
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState('')
  const [selection, setSelection] = useState<MiniMetricId[]>(() => {
    try { return readMiniSelection(JSON.parse(localStorage.getItem(MINI_PREFERENCE_KEY) || 'null')) }
    catch { return [...defaultMiniMetrics] }
  })
  const selectionRef = useRef(selection)
  selectionRef.current = selection
  const [drag, setDrag] = useState<Drag | null>(null)
  const dragging = useRef<Drag | null>(null)
  const press = useRef<Press | null>(null)
  const grid = useRef<HTMLDivElement>(null)
  const frame = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const settingsButton = useRef<HTMLButtonElement>(null)
  const settingsPanel = useRef<HTMLDivElement>(null)
  const scrollTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [announcement, setAnnouncement] = useState('')
  const displayed = drag?.order ?? selection
  const captureOrder = useReorderMotion(grid, 'data-mini-metric', JSON.stringify(displayed), drag?.id)

  useEffect(() => {
    if (!bridge) { setError('请从 ModLoader 打开小窗监控'); return }
    let active = true, received = false
    const stopSnapshot = bridge.onSnapshot(next => { received = true; setSnapshot(next) })
    const stopPinned = bridge.onAlwaysOnTop(setPinned)
    const stopOpacity = bridge.onOpacity(setOpacity)
    const stopEditing = bridge.onEditing(setEditing)
    void bridge.ready().then(initial => {
      if (active) { if (!received) setSnapshot(initial.snapshot); setPinned(initial.alwaysOnTop); setOpacity(initial.opacity); setEditing(initial.editing) }
    }).catch(issue => { if (active) setError(String(issue.message || issue)) })
    return () => { active = false; stopSnapshot(); stopPinned(); stopOpacity(); stopEditing() }
  }, [bridge])
  useEffect(() => {
    try { localStorage.setItem(MINI_PREFERENCE_KEY, JSON.stringify(selection)); setError('') }
    catch { setError('显示项未能保存，关闭后可能丢失') }
  }, [selection])
  useEffect(() => {
    if (editing) settingsPanel.current?.querySelector<HTMLInputElement>('input')?.focus()
  }, [editing])

  function contentHeight() {
    if (!frame.current || !scroller.current || !content.current) return null
    const scrollStyle = getComputedStyle(scroller.current)
    const contentStyle = getComputedStyle(content.current)
    return Math.ceil(scroller.current.getBoundingClientRect().top - frame.current.getBoundingClientRect().top
      + content.current.getBoundingClientRect().height + parseFloat(scrollStyle.paddingTop) + parseFloat(scrollStyle.paddingBottom)
      + parseFloat(contentStyle.marginTop) + parseFloat(contentStyle.marginBottom)
      + parseFloat(getComputedStyle(frame.current).borderBottomWidth))
  }
  useLayoutEffect(() => {
    if (!bridge || editing || !content.current || !frame.current) return
    let previous: number | null = null
    const measure = () => {
      const height = contentHeight()
      if (height !== null && height !== previous) { previous = height; bridge.reportContentHeight(height) }
    }
    const observer = new ResizeObserver(measure)
    observer.observe(content.current)
    observer.observe(frame.current)
    const header = frame.current.querySelector('.mini-header')
    if (header) observer.observe(header)
    measure()
    return () => observer.disconnect()
  }, [bridge, editing, error])

  async function fitContent() {
    endDrag(false)
    const height = contentHeight()
    if (height === null) return
    try { await bridge?.fitContent(height) }
    catch (issue) { setError(issue instanceof Error ? issue.message : String(issue)) }
  }

  async function edit(value: boolean) {
    endDrag(false)
    try {
      await bridge?.setEditing(value)
      setEditing(value)
      if (!value) requestAnimationFrame(() => settingsButton.current?.focus())
    } catch (issue) { setError(issue instanceof Error ? issue.message : String(issue)) }
  }
  function toggle(id: MiniMetricId) {
    setSelection(previous => previous.includes(id) ? previous.filter(item => item !== id) : [...previous, id])
  }
  function targetBefore(x: number, y: number, id: MiniMetricId): MiniMetricId | null {
    const nodes = [...(grid.current?.querySelectorAll<HTMLElement>('[data-mini-metric]') ?? [])]
    for (const node of nodes) {
      if (node.dataset.miniMetric === id) continue
      const box = reorderLayoutRect(node)
      if (y < box.top || (y <= box.bottom && x < box.left + box.width / 2)) return node.dataset.miniMetric as MiniMetricId
    }
    return null
  }
  function moveDrag(x: number, y: number) {
    const current = dragging.current
    if (!current) return
    const order = moveMiniMetric(current.order, current.id, targetBefore(x, y, current.id))
    if (order.some((id, index) => id !== current.order[index])) captureOrder()
    const next = { ...current, x, y, order }
    dragging.current = next; setDrag(next)
  }
  function endDrag(commit: boolean) {
    const current = dragging.current, pending = press.current
    if (current) captureOrder()
    dragging.current = null; press.current = null; setDrag(null)
    if (pending && grid.current?.hasPointerCapture(pending.pointer)) grid.current.releasePointerCapture(pending.pointer)
    if (current && commit) {
      setSelection(current.order)
      setAnnouncement('排列已更新')
    }
  }
  function pointerDown(event: PointerEvent<HTMLButtonElement>, id: MiniMetricId) {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.focus()
    press.current = { id, x: event.clientX, y: event.clientY, pointer: event.pointerId, box: event.currentTarget.getBoundingClientRect() }
    grid.current?.setPointerCapture(event.pointerId)
  }
  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const pending = press.current
    if (!pending || pending.pointer !== event.pointerId) return
    if (!dragging.current && Math.hypot(event.clientX - pending.x, event.clientY - pending.y) >= 6) {
      dragging.current = { id: pending.id, x: event.clientX, y: event.clientY,
        offsetX: pending.x - pending.box.left, offsetY: pending.y - pending.box.top,
        width: pending.box.width, height: pending.box.height, order: [...selectionRef.current] }
    }
    moveDrag(event.clientX, event.clientY)
  }
  useEffect(() => {
    if (!drag?.id) return
    let frame = 0, previousTime = 0
    const scroll = (time: number) => {
      const container = scroller.current, current = dragging.current
      if (!container || !current) return
      const bounds = container.getBoundingClientRect()
      const elapsed = previousTime ? Math.min(time - previousTime, 40) : 0
      previousTime = time
      const direction = current.y < bounds.top + 24 ? -1 : current.y > bounds.bottom - 24 ? 1 : 0
      const before = container.scrollTop
      container.scrollTop += direction * elapsed * .45
      if (before !== container.scrollTop) moveDrag(current.x, current.y)
      frame = requestAnimationFrame(scroll)
    }
    frame = requestAnimationFrame(scroll)
    return () => cancelAnimationFrame(frame)
  }, [drag?.id])
  useEffect(() => {
    const cancel = () => endDrag(false)
    const key = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      if (press.current) { event.preventDefault(); endDrag(false) }
      else if (editing) { event.preventDefault(); void edit(false) }
    }
    const visibility = () => { if (document.hidden) cancel() }
    window.addEventListener('blur', cancel)
    document.addEventListener('visibilitychange', visibility)
    document.addEventListener('keydown', key)
    return () => {
      window.removeEventListener('blur', cancel); document.removeEventListener('visibilitychange', visibility)
      document.removeEventListener('keydown', key)
    }
  }, [editing])
  useEffect(() => () => { dragging.current = null; press.current = null; if (scrollTimer.current) clearTimeout(scrollTimer.current) }, [])

  function revealScrollbar(element: HTMLDivElement) {
    element.classList.add('is-scrolling')
    if (scrollTimer.current) clearTimeout(scrollTimer.current)
    scrollTimer.current = setTimeout(() => { element.classList.remove('is-scrolling'); scrollTimer.current = null }, 800)
  }

  function keyboardMove(event: React.KeyboardEvent, id: MiniMetricId) {
    if (press.current) return
    if (!event.altKey || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return
    event.preventDefault()
    const ids = selectionRef.current, index = ids.indexOf(id)
    const nextIndex = index + (['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1)
    if (nextIndex < 0 || nextIndex >= ids.length) return
    const next = [...ids]; [next[index], next[nextIndex]] = [next[nextIndex], next[index]]
    setSelection(next); setAnnouncement(`已移至第 ${nextIndex + 1} 项`)
  }
  function readout(id: MiniMetricId, floating = false) {
    const metric = miniMetrics.find(item => item.id === id)!
    const value = snapshot?.metrics[id] ?? emptyReadout
    const pair = id === 'context' || id === 'memory' || id === 'vram'
    const [used, total = '—'] = value.value.split(' / ')
    const basis = value.value !== '—' ? value.basis : undefined
    return <><span className="mini-metric-label"><span>{id === 'cpu_temperature' || id === 'gpu_temperature' ? <><Thermometer size={13} aria-hidden="true" />{id === 'cpu_temperature' ? 'CPU' : 'GPU'}</> : metric.label}</span><small className="mini-unit">{value.unit}</small>{floating && <GripVertical size={12} />}</span>
      <span className={`mini-readout ${pair || basis ? 'mini-multiline' : ''}`}>
        <span className="mini-value-line"><strong>{pair ? used : value.value}</strong></span>
        {(pair || basis) && <span className="mini-readout-secondary">{basis && <span className="mini-basis">{basis}</span>}{pair && <span className="mini-total">/ {total}</span>}</span>}
      </span></>
  }
  const tone = snapshot?.tone ?? 'waiting'
  return <div className="mini-frame" ref={frame} style={{ '--mini-scale': snapshot?.fontScale ?? 1 } as CSSProperties}>
    {editing ? <header className="mini-header mini-edit-header">
      <button className="mini-icon" aria-label="返回监控" onClick={() => void edit(false)}><ArrowLeft size={15} /></button>
      <span className="mini-edit-title">显示项</span><span className="mini-title-space" />
      <button className="mini-done" onClick={() => void edit(false)}>完成</button>
    </header> : <header className="mini-header">
      <span className={`mini-status-dot ${tone}`} />
      <div className="mini-instance"><select aria-label="选择监控实例" value={snapshot?.instanceId ?? ''} disabled={!snapshot?.instances.length}
        onChange={event => bridge?.selectInstance(event.target.value)}>
        {!snapshot?.instances.length && <option value="">未选择实例</option>}
        {snapshot?.instances.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select><ChevronDown size={12} /></div>
      <span className="mini-status" role="status" title={snapshot?.stateLabel ?? '等待主界面'}>{snapshot?.stateLabel ?? '等待主界面'}</span>
      <span className="mini-title-space" />
      <div className="mini-actions">
        <button className="mini-icon" aria-label="适应内容" title="适应内容" onClick={() => void fitContent()}><Minimize2 size={15} /></button>
        <button className={`mini-icon ${pinned ? 'is-pinned' : ''}`} aria-label={pinned ? '取消置顶' : '置顶'} title={pinned ? '取消置顶' : '置顶'} aria-pressed={pinned}
          onClick={() => void bridge?.setAlwaysOnTop(!pinned).then(setPinned).catch(issue => setError(String(issue.message || issue)))}>{pinned ? <Pin size={15} /> : <PinOff size={15} />}</button>
        <button ref={settingsButton} className="mini-icon" aria-label="选择显示项" title="选择显示项" onClick={() => void edit(true)}><SlidersHorizontal size={16} /></button>
        <button className="mini-icon mini-close" aria-label="关闭小窗" title="关闭小窗" onClick={() => bridge?.close()}><X size={16} /></button>
      </div>
    </header>}
    {error && <div className="mini-error" role="alert">{error}</div>}
    {editing ? <div className="mini-settings" ref={settingsPanel} onScroll={event => revealScrollbar(event.currentTarget)}>
      <label className="mini-opacity"><span><span>窗口不透明度</span><output>{Math.round(opacity * 100)}%</output></span>
        <input aria-label="窗口不透明度" type="range" min="0" max="100" step="1" value={Math.round(opacity * 100)} onChange={event => {
          const next = Number(event.target.value) / 100
          setOpacity(next)
          void bridge?.setOpacity(next).catch(issue => setError(String(issue.message || issue)))
        }} /><small>完成后应用</small>
      </label>
      {snapshot && snapshot.gpus.length > 1 && <label className="mini-gpu">GPU<select aria-label="监控 GPU" value={snapshot.gpuId} onChange={event => bridge?.selectGpu(event.target.value)}>
        {snapshot.gpus.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select></label>}
      <div className="mini-settings-groups">{(['generation', 'hardware'] as const).map(group => <section key={group}><h2>{group === 'generation' ? '生成与请求' : '硬件'}</h2>
        {miniMetrics.filter(item => item.group === group).map(metric => <label className="mini-checkbox" key={metric.id}>
          <input type="checkbox" checked={selection.includes(metric.id)} onChange={() => toggle(metric.id)} />
          <span className="mini-check-mark" aria-hidden="true"><Check size={12} /></span><span>{metric.label}</span>
        </label>)}
      </section>)}</div>
    </div> : <div className="mini-scroll" ref={scroller} onScroll={event => revealScrollbar(event.currentTarget)}>
      <div className="mini-content" ref={content}>{!selection.length ? <button className="mini-empty" onClick={() => void edit(true)}>选择显示项</button> : <div className="mini-grid" ref={grid} data-dragging={!!drag}
        onPointerMove={pointerMove} onPointerUp={() => endDrag(true)} onPointerCancel={() => endDrag(false)} onLostPointerCapture={() => endDrag(false)}>
        {displayed.map(id => <button key={id} data-mini-metric={id}
          className={`mini-metric ${id === 'current' ? 'mini-current' : ''} ${drag?.id === id ? 'is-dragging' : ''}`}
          aria-label={`${miniMetrics.find(item => item.id === id)!.label}，Alt 加方向键调整顺序`}
          onPointerDown={event => pointerDown(event, id)} onKeyDown={event => keyboardMove(event, id)}>
          {readout(id)}
        </button>)}
      </div>}</div>
    </div>}
    {drag && <div className="mini-drag-preview" aria-hidden="true" style={{ width: drag.width, minHeight: drag.height,
      transform: `translate3d(${Math.max(5, Math.min(window.innerWidth - drag.width * 1.01 - 5, drag.x - drag.offsetX))}px, ${Math.max(5, Math.min(window.innerHeight - drag.height * 1.01 - 5, drag.y - drag.offsetY))}px, 0)` }}>
      <div className={`mini-metric mini-drag-card ${drag.id === 'current' ? 'mini-current' : ''}`}>{readout(drag.id, true)}</div>
    </div>}
    <span className="mini-resize-mark" aria-hidden="true" />
    <span className="mini-sr-only" aria-live="polite">{announcement}</span>
  </div>
}
