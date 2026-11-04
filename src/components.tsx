import { Children, cloneElement, createElement, isValidElement, useEffect, useId, useRef, useState, type InputHTMLAttributes, type OptionHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react'
import { AlertCircle, Check, ChevronRight, Loader2, Pause, Play, Square, X } from 'lucide-react'
import type { Job } from './types'
import { isJobActive, jobState } from './api'
import './job-status.css'

export function Button({ children, icon: Icon, variant = 'default', small, busy, className = '', ...props }:
  React.ButtonHTMLAttributes<HTMLButtonElement> & { icon?: React.ElementType; variant?: 'default' | 'primary' | 'danger' | 'ghost'; small?: boolean; busy?: boolean }) {
  return <button {...props} className={`button ${variant} ${small ? 'small' : ''} ${className}`} disabled={props.disabled || busy}>
    {busy ? <Loader2 size={15} className="spin" /> : Icon ? <Icon size={15} /> : null}{children}
  </button>
}

export function IconButton({ icon: Icon, label, title: _title, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { icon: React.ElementType; label: string }) {
  return <button {...props} type={props.type || 'button'} className={`icon-button ${props.className || ''}`} data-tooltip={label} aria-label={label}><Icon size={16} /></button>
}

const states: Record<string, { label: string; tone: string }> = {
  running: { label: '运行中', tone: 'green' }, ready: { label: '运行中', tone: 'green' },
  starting: { label: '加载中', tone: 'blue' }, loading: { label: '加载中', tone: 'blue' },
  preparing: { label: '准备中', tone: 'blue' },
  stopping: { label: '停止中', tone: 'amber' }, stopped: { label: '已停止', tone: 'neutral' },
  failed: { label: '失败', tone: 'red' }, error: { label: '失败', tone: 'red' },
  queued: { label: '排队中', tone: 'neutral' }, pending: { label: '等待中', tone: 'neutral' },
  downloading: { label: '下载中', tone: 'blue' }, download: { label: '下载中', tone: 'blue' },
  installing: { label: '安装中', tone: 'blue' }, install: { label: '安装中', tone: 'blue' },
  resolving: { label: '解析中', tone: 'blue' }, resolving_dependencies: { label: '解析中', tone: 'blue' }, resolve: { label: '解析中', tone: 'blue' },
  extracting: { label: '解压中', tone: 'blue' }, extract: { label: '解压中', tone: 'blue' }, unpacking: { label: '解压中', tone: 'blue' },
  planning: { label: '准备中', tone: 'blue' }, prepare: { label: '准备中', tone: 'blue' },
  checking: { label: '校验中', tone: 'blue' }, verify: { label: '校验中', tone: 'blue' }, verifying: { label: '校验中', tone: 'blue' },
  verify_cache: { label: '校验缓存', tone: 'blue' }, verifying_cache: { label: '校验缓存', tone: 'blue' },
  register: { label: '登记中', tone: 'blue' }, registering: { label: '登记中', tone: 'blue' },
  validate: { label: '验证中', tone: 'blue' }, validating: { label: '验证中', tone: 'blue' }, validation: { label: '验证中', tone: 'blue' },
  copying: { label: '复制中', tone: 'blue' }, copy: { label: '复制中', tone: 'blue' },
  phase_complete: { label: '阶段完成', tone: 'neutral' },
  completed: { label: '已完成', tone: 'green' }, complete: { label: '已完成', tone: 'green' },
  paused: { label: '已暂停', tone: 'amber' }, pausing: { label: '暂停中', tone: 'amber' },
  awaiting_confirmation: { label: '待确认', tone: 'amber' },
  cancelled: { label: '已停止', tone: 'neutral' }, canceled: { label: '已停止', tone: 'neutral' },
  interrupted: { label: '已中断', tone: 'amber' }, cancelling: { label: '停止中', tone: 'amber' },
  supported: { label: '已验证支持', tone: 'green' }, unverified: { label: '兼容性未验证', tone: 'amber' },
  unsupported: { label: '不支持', tone: 'red' },
}

export function Badge({ children, tone = 'neutral', dot = false }: { children: ReactNode; tone?: string; dot?: boolean }) {
  return <span className={`badge ${tone}`}>{dot && <span className="status-dot" />}{children}</span>
}

export function StateBadge({ state }: { state: string }) {
  const value = states[state] || { label: state, tone: 'neutral' }
  return <Badge tone={value.tone} dot>{value.label}</Badge>
}

const lifecycleStates = new Set([
  'queued', 'pending', 'paused', 'pausing', 'stopping', 'cancelling', 'awaiting_confirmation',
  'failed', 'error', 'cancelled', 'canceled', 'interrupted', 'stopped', 'completed', 'complete',
])

const phaseBadgeStates: Record<string, string> = {
  complete: 'phase_complete',
  completed: 'phase_complete',
  prepared: 'phase_complete',
}

export function jobBadgeState(job: Job) {
  const status = jobState(job)
  if (lifecycleStates.has(status)) return status
  const phase = String(job.phase || '').toLowerCase().replaceAll('-', '_')
  if (!phase) return status
  return phaseBadgeStates[phase] || (states[phase] ? phase : status)
}

export function Empty({ icon: Icon, title, detail, children }: { icon: React.ElementType; title: string; detail?: string; children?: ReactNode }) {
  return <div className="empty-state"><span className="empty-icon"><Icon size={26} strokeWidth={1.5} /></span>
    <h3>{title}</h3>{detail && <p>{detail}</p>}{children && <div className="empty-actions">{children}</div>}</div>
}

export function PageHeader({ title, description, children }: { eyebrow: string; title: string; description?: string; children?: ReactNode }) {
  return <header className="page-header"><div><h1>{title}</h1>{description && <p>{description}</p>}</div>
    {children && <div className="page-actions">{children}</div>}</header>
}

export function ErrorNotice({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-notice" role="alert"><AlertCircle size={17} /><span>{message}</span>{retry && <Button small onClick={retry}>重试</Button>}</div>
}

export function DiagnosticNotice({ summary, detail, retry }: { summary: string; detail?: unknown; retry?: () => void }) {
  const text = typeof detail === 'string' ? detail : JSON.stringify(detail, null, 2)
  return <div className="diagnostic-notice"><div role="alert"><span>{summary}</span>{retry && <Button small variant="ghost" onClick={retry}>重试</Button>}</div>
    {text && text !== summary && <details><summary>诊断详情</summary><pre>{text}</pre></details>}
  </div>
}

export function Field({ label, children, help, className = '', as: Container = 'label' }: { label: ReactNode; children: ReactNode; help?: string; className?: string; as?: 'label' | 'div' }) {
  const labelId = useId()
  const controls = Children.map(children, child => {
    if (!isValidElement<SelectHTMLAttributes<HTMLSelectElement>>(child) || (child.type !== 'select' && child.type !== Select)) return child
    return cloneElement(child, {
      'aria-labelledby': child.props['aria-labelledby'] || (child.props['aria-label'] ? undefined : labelId),
      'aria-describedby': child.props['aria-describedby'] || (help ? `${labelId}-help` : undefined),
    })
  })
  return <Container className={`field ${className}`}><span id={labelId} className="field-label">{label}</span>{controls}{help && <span id={`${labelId}-help`} className="field-help">{help}</span>}</Container>
}

export function Select({ children, onFocusCapture, onKeyDownCapture, onClick, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  const element = useRef<HTMLSelectElement>(null)
  const options = Children.map(children, child => {
    if (!isValidElement<OptionHTMLAttributes<HTMLOptionElement>>(child) || child.type !== 'option') return child
    const label = Children.toArray(child.props.children).filter(value => typeof value === 'string' || typeof value === 'number').join('')
    return cloneElement(child, { title: child.props.title ?? label }, <span className="select-option-label">{child.props.children}</span>)
  })
  function revealOption() {
    requestAnimationFrame(() => {
      const select = element.current
      if (!select?.matches(':open')) return
      const group = select.querySelector<HTMLElement>('.select-options')
      const focused = document.activeElement
      const option = focused instanceof HTMLOptionElement && select.contains(focused) ? focused : select.querySelector<HTMLOptionElement>('option:checked')
      if (!group || !option) return
      const bounds = group.getBoundingClientRect(), item = option.getBoundingClientRect()
      if (item.top < bounds.top) group.scrollTop -= bounds.top - item.top
      else if (item.bottom > bounds.bottom) group.scrollTop += item.bottom - bounds.bottom
    })
  }
  return <select {...props} ref={element} onFocusCapture={event => { onFocusCapture?.(event); revealOption() }} onKeyDownCapture={event => {
    onKeyDownCapture?.(event)
    const select = element.current
    if (!event.defaultPrevented && select?.matches(':open') && ['Home', 'End'].includes(event.key)) {
      const options = Array.from(select.options).filter(option => !option.disabled)
      event.preventDefault()
      const option = event.key === 'Home' ? options[0] : options.at(-1)
      option?.focus({ preventScroll: true })
    }
    revealOption()
  }} onClick={event => { onClick?.(event); revealOption() }}><button type="button" tabIndex={-1} className="select-value">{createElement('selectedcontent')}</button><optgroup label="" className="select-options">{options}</optgroup></select>
}

export function NumberInput({ onValueChange, ...props }: Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> & { onValueChange: (value: string) => void }) {
  const element = useRef<HTMLInputElement>(null)
  const [draft, setDraft] = useState(String(props.value ?? ''))
  const emitted = useRef<number | string | undefined>(undefined)
  useEffect(() => {
    const next = props.value ?? ''
    if (!Object.is(emitted.current, next)) setDraft(String(next))
    emitted.current = undefined
  }, [props.value])
  function change(value: string, invalid = false) {
    setDraft(value)
    emitted.current = invalid ? '' : value === '' ? '' : Number(value)
    onValueChange(invalid ? 'NaN' : value)
  }
  const changed = useRef(onValueChange)
  changed.current = change
  useEffect(() => {
    const input = element.current!
    const wheel = (event: WheelEvent) => {
      if (document.activeElement !== input || input.disabled || input.readOnly || event.ctrlKey || event.metaKey || !event.deltaY) return
      event.preventDefault()
      event.stopPropagation()
      const previous = input.value
      const step = input.getAttribute('step')
      // "any" allows manual decimals but the DOM stepping API needs a numeric step.
      if (step === 'any') input.step = '1'
      try {
        if (event.deltaY < 0) input.stepUp()
        else input.stepDown()
      } finally {
        if (step === null) input.removeAttribute('step')
        else input.step = step
      }
      if (input.value !== previous) changed.current(input.value)
    }
    input.addEventListener('wheel', wheel, { passive: false })
    return () => input.removeEventListener('wheel', wheel)
  }, [])
  return <input {...props} value={draft} type="number" ref={element} onChange={event => change(event.target.value, event.target.validity.badInput)} />
}

let openModals = 0
let pageOverflow = ''

export function Modal({ title, description, children, footer, onClose, wide = false, dismissible = true, className = '' }: { title: string; description?: string; children: ReactNode; footer?: ReactNode; onClose: () => void; wide?: boolean; dismissible?: boolean; className?: string }) {
  const element = useRef<HTMLDivElement>(null)
  const previousFocus = useRef(document.activeElement as HTMLElement | null)
  const close = useRef(onClose)
  close.current = () => { if (dismissible) onClose() }
  useEffect(() => {
    const target = element.current
    if (openModals++ === 0) pageOverflow = document.documentElement.style.overflow
    document.documentElement.style.overflow = 'hidden'
    if (!target?.contains(document.activeElement)) target?.focus()
    const handler = (event: KeyboardEvent) => {
      if (Array.from(document.querySelectorAll('[role="dialog"][aria-modal="true"]')).at(-1) !== target) return
      if (event.key === 'Escape' && !target?.querySelector('select:open')) { event.stopPropagation(); close.current() }
      if (event.key === 'Tab' && target) {
        const items = Array.from(target.querySelectorAll<HTMLElement>('button:enabled, input:enabled, select:enabled, textarea:enabled, a[href], summary')).filter(item => item.getClientRects().length > 0 && item.tabIndex !== -1)
        const first = items[0], last = items[items.length - 1]
        if (!items.length) { event.preventDefault(); target.focus(); return }
        const current = document.activeElement as HTMLElement | null
        if (!target.contains(current) || current === target) { event.preventDefault(); (event.shiftKey ? last : first).focus() }
        else if (event.shiftKey && current === first) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && current === last) { event.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', handler)
    return () => {
      document.removeEventListener('keydown', handler)
      if (--openModals === 0) document.documentElement.style.overflow = pageOverflow
      // StrictMode replays effects while the dialog is still mounted.
      if (!target?.isConnected) previousFocus.current?.focus({ preventScroll: true })
    }
  }, [])
  return <div className="modal-overlay" onMouseDown={event => { if (event.target === event.currentTarget) close.current() }}>
    <div className={`modal ${wide ? 'wide' : ''} ${className}`} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={element}>
      <div className="modal-header"><div><h2>{title}</h2>{description && <p title={description}>{description}</p>}</div><IconButton icon={X} label="关闭" disabled={!dismissible} onClick={onClose} /></div>
      <div className="modal-body">{children}</div>{footer && <div className="modal-footer">{footer}</div>}
    </div></div>
}

export function formatBytes(value?: number | null) {
  if (value === undefined || value === null || !Number.isFinite(value)) return '未知'
  if (value === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  const index = Math.min(Math.floor(Math.log(Math.abs(value)) / Math.log(1024)), units.length - 1)
  return `${(value / 1024 ** index).toLocaleString('zh-CN', { maximumFractionDigits: index < 2 ? 0 : 1 })} ${units[index]}`
}

export function JobRow({ job, onCancel, onRetry, onPause, onResume, busy = false }: { job: Job; onCancel?: () => void; onRetry?: () => void; onPause?: () => void; onResume?: () => void; busy?: boolean }) {
  const active = isJobActive(job)
  const status = jobState(job)
  const paused = status === 'paused'
  const transitioning = ['pausing', 'cancelling'].includes(status)
  const phase = String(job.phase || '').toLowerCase().replaceAll('-', '_')
  const kind = String(job.kind || '')
  const transfer = ['download', 'downloading', 'copy', 'copying', 'transfer', 'transferring'].includes(phase)
    || (!phase && ['downloading', 'copying'].includes(status))
    || (!phase && ['download', 'model_migration'].includes(kind))
  const totalValue = job.total_bytes
  const total = typeof totalValue === 'number' && Number.isFinite(totalValue) && totalValue >= 0 ? totalValue : undefined
  const byteValue = job.completed_bytes ?? job.downloaded_bytes ?? job.copied_bytes
  const done = typeof byteValue === 'number' && Number.isFinite(byteValue) && byteValue >= 0 ? byteValue : 0
  const hasRatio = total !== undefined && total > 0
  const progress = hasRatio ? Math.max(0, Math.min(100, done / total * 100)) : 0
  const showTransfer = transfer && (active || paused)
  const copying = phase === 'copy' || phase === 'copying' || status === 'copying' || kind === 'model_migration'
  const transferLabel = copying ? '复制进度' : '下载进度'
  const showTransferBytes = showTransfer && kind !== 'model_migration'
  const canRetry = ['failed', 'error', 'cancelled', 'canceled', 'interrupted'].includes(status) && onRetry
  const hasActions = (active && ((job.can_pause && onPause) || onCancel)) || (paused && (onResume || onCancel)) || canRetry
  return <div className="job-row"><div className="job-top"><div className="job-title">{active ? <Loader2 className="spin" size={16} /> : ['completed', 'complete'].includes(status) ? <Check size={16} /> : <ChevronRight size={16} />}
    <strong>{job.name || job.label || job.repo_id || job.backend || '任务'}</strong></div><StateBadge state={jobBadgeState(job)} /></div>
    {showTransfer && <>
      {showTransferBytes && <div className="job-progress-copy"><span>{transferLabel}</span><span>{formatBytes(done)} / {formatBytes(total)}</span></div>}
      {(hasRatio || total === undefined) && <div className={`progress-track ${active && total === undefined ? 'indeterminate' : ''}`} role="progressbar" aria-label={transferLabel} aria-valuemin={0} aria-valuemax={100} aria-valuenow={hasRatio ? Math.round(progress) : undefined}><span style={{ width: `${progress}%` }} /></div>}
    </>}
    {(job.error || job.message || hasActions) && <div className="job-bottom"><span>{job.error || job.message || ''}</span>
      <div className="job-actions">
        {active && job.can_pause && onPause && <Button small icon={Pause} onClick={onPause} disabled={busy || transitioning}>暂停</Button>}
        {paused && onResume && <Button small icon={Play} onClick={onResume} disabled={busy}>继续</Button>}
        {(active || paused) && onCancel && <Button small icon={Square} onClick={onCancel} disabled={busy || status === 'cancelling'}>{status === 'cancelling' ? '停止中' : '停止'}</Button>}
        {canRetry && <Button small onClick={onRetry} disabled={busy}>重试</Button>}
      </div></div>
    }
    {Array.isArray(job.import_errors) && job.import_errors.length > 0 && <details className="download-import-errors"><summary>查看未登记项目</summary>{(job.import_errors as Array<{path:string;error:string}>).map((item,index) => <p key={index}>{item.path}<br />{item.error}</p>)}</details>}
  </div>
}

export function KeyValue({ label, value, mono }: { label: string; value: ReactNode; mono?: boolean }) {
  return <div className="key-value"><span>{label}</span><div className={mono ? 'mono' : ''}>{value}</div></div>
}
