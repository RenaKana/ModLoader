import { Fragment, useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { Check, ChevronDown, ChevronLeft, ChevronRight, Loader2, Play, Plus, Square } from 'lucide-react'
import { effectiveReasoningEffort, reasoningLabel, type ReasoningCapability, type ReasoningEffort } from './chat-reasoning'
import type { ChatQuickLoadOption } from './chat-model-loading'
import { isActive, isReady } from './api'
import './chat-model-loading.css'

export default function ChatReasoningControl({ capability, value, defaultLabel = '引擎默认', disabled, resetKey, modelId, modelName, modelOptions, quickLoadOptions = [], onLoadProfile, onUnloadModel, unloadingIds = [], unloadError, onModelChange, onChange }: {
  capability?: ReasoningCapability; value?: ReasoningEffort; defaultLabel?: string; disabled: boolean; resetKey: string; onChange: (value: ReasoningEffort | undefined) => void
  modelId: string; modelName: string; modelOptions: Array<{ value: string; label: string; state?: string }>; onModelChange: (id: string) => void
  quickLoadOptions?: ChatQuickLoadOption[]; onLoadProfile?: (id: string, parallel?: boolean) => void; onUnloadModel?: (id: string) => void; unloadingIds?: string[]; unloadError?: string
}) {
  const id = useId()
  const anchor = `--reasoning-${id.replace(/[^a-zA-Z0-9-]/g, '')}`
  const popup = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const simpleView = useRef<HTMLDivElement>(null)
  const modelView = useRef<HTMLDivElement>(null)
  const slider = useRef<HTMLInputElement>(null)
  const modelButton = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState(false)
  const [highlighted, setHighlighted] = useState(0)
  const [preview, setPreview] = useState<{ value?: ReasoningEffort } | null>(null)
  const previewValue = useRef<{ value?: ReasoningEffort } | null>(null)
  const dragStart = useRef<ReasoningEffort | undefined>(undefined)
  const dragging = useRef(false)
  const [layout, setLayout] = useState<{ height: number; side: 'top' | 'bottom' }>()
  const [transitionsReady, setTransitionsReady] = useState(false)
  const [customDraft, setCustomDraft] = useState('')
  const [customError, setCustomError] = useState('')
  const effort = effectiveReasoningEffort(capability, preview ? preview.value : value)
  const effortLabel = effort === undefined ? defaultLabel : reasoningLabel(effort)
  const levels = [undefined, ...(capability?.levels.map(level => level.value) || [])]
  const selected = levels.indexOf(effort)
  const deepseek = capability?.kind === 'deepseek'
  const numericEffort = typeof effort === 'number' ? effort : ({ low: 50, medium: 62, high: 75, max: 100 } as Record<string, number>)[effort || '']
  const position = deepseek ? numericEffort === undefined ? 0 : (numericEffort - 1) / 99 : Math.max(0, selected) / Math.max(1, levels.length - 1)
  const unset = deepseek && numericEffort === undefined
  const maximum = effort === 'xhigh' || effort === 'max' || effort === 100
  const loadOptions = onLoadProfile ? quickLoadOptions : []
  const optionCount = modelOptions.length + loadOptions.length
  const modelStates = modelOptions.map(option => {
    const ready = option.state === undefined || isReady(option.state)
    return { ready, pending: !ready && isActive(option.state || ''),
      label: ready ? '已加载' : option.state === 'preparing' ? '准备中' : option.state === 'stopping' ? '正在停止'
        : option.state === 'failed' ? '加载失败' : ['starting', 'loading'].includes(option.state || '') ? '加载中' : '未运行' }
  })
  const showModelGroups = loadOptions.length > 0 || modelStates.some(state => !state.ready)

  function cancelDrag() { dragging.current = false; previewValue.current = null; setPreview(null) }
  function endDrag() {
    if (!dragging.current) return
    const final = previewValue.current
    if (final && !Object.is(dragStart.current, final.value)) onChange(final.value)
    cancelDrag()
  }
  function changeSlider(next: ReasoningEffort | undefined) {
    if (!dragging.current) { onChange(next); return }
    previewValue.current = { value: next }
    setPreview(previewValue.current)
  }
  function showModels() { setHighlighted(Math.max(0, modelOptions.findIndex(option => option.value === modelId))); setModels(true) }
  function chooseModel(index: number) {
    if (disabled) return
    const option = modelOptions[index]
    if (option) { onModelChange(option.value); setModels(false); return }
    const profile = loadOptions[index - modelOptions.length]
    if (profile && !profile.disabled) onLoadProfile?.(profile.value)
  }
  useEffect(() => { popup.current?.hidePopover() }, [resetKey])
  useEffect(() => { if (disabled) popup.current?.hidePopover() }, [disabled])
  useEffect(cancelDrag, [resetKey, modelId])
  useEffect(() => setHighlighted(current => Math.max(0, Math.min(current, optionCount - 1))), [optionCount])
  useEffect(() => { setCustomDraft(numericEffort === undefined ? '' : String(numericEffort)); setCustomError('') }, [effort, resetKey])
  useEffect(() => {
    window.addEventListener('pointerup', endDrag); window.addEventListener('pointercancel', cancelDrag)
    return () => { window.removeEventListener('pointerup', endDrag); window.removeEventListener('pointercancel', cancelDrag) }
  })
  useLayoutEffect(() => {
    if (!open) return
    const panel = popup.current!, button = trigger.current!
    const measure = () => {
      const box = button.getBoundingClientRect(), above = Math.max(0, box.top - 20), below = Math.max(0, innerHeight - box.bottom - 20)
      panel.style.setProperty('--reasoning-available', `${Math.max(above, below)}px`)
      const height = (models ? modelView : simpleView).current!.offsetHeight + 10
      const side = above >= height || above >= below ? 'top' : 'bottom'
      const next = { height: Math.min(height, side === 'top' ? above : below), side } as const
      setLayout(previous => previous?.height === next.height && previous.side === next.side ? previous : next)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(simpleView.current!); observer.observe(modelView.current!); observer.observe(button)
    window.addEventListener('resize', measure)
    const frame = requestAnimationFrame(() => setTransitionsReady(true))
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); cancelAnimationFrame(frame) }
  }, [open, models])
  useLayoutEffect(() => {
    if (!open) return
    ;(models ? list.current : slider.current || modelButton.current)?.focus({ preventScroll: true })
  }, [open, models])
  useLayoutEffect(() => {
    if (!models || !list.current) return
    const option = list.current.querySelectorAll<HTMLElement>('[role="option"]')[highlighted]
    if (!option) return
    const bounds = list.current.getBoundingClientRect(), item = option.getBoundingClientRect()
    if (item.top < bounds.top) list.current.scrollTop -= bounds.top - item.top
    else if (item.bottom > bounds.bottom) list.current.scrollTop += item.bottom - bounds.bottom
  }, [models, highlighted])

  return <>
    <button ref={trigger} type="button" className="chat-model-capsule" aria-label={`对话模型与推理强度：${modelName}，${effortLabel}`}
      aria-haspopup="dialog" aria-expanded={open} aria-controls={id} disabled={disabled || !optionCount} popoverTarget={id}
      style={{ anchorName: anchor } as CSSProperties}>
      <span className="chat-capsule-model">{modelName}</span><span className={`chat-capsule-effort${effort === undefined ? ' is-default' : ''}`}>{effortLabel}</span><ChevronDown size={13} />
    </button>
    <div ref={popup} id={id} className="chat-reasoning-popover" data-view={models ? 'models' : 'reasoning'} data-side={layout?.side || 'top'} data-transitions-ready={transitionsReady} role="dialog" aria-label="模型与推理强度" popover="auto" style={{ positionAnchor: anchor, height: layout?.height } as CSSProperties}
      onKeyDown={event => {
        if (event.key === 'Escape') cancelDrag()
        if (models && modelOptions.length && event.key === 'ArrowLeft') { event.preventDefault(); event.stopPropagation(); setModels(false) }
      }} onBeforeToggle={event => {
        if (event.newState === 'closed') cancelDrag()
        else if (getComputedStyle(event.currentTarget).display === 'none') setTransitionsReady(false)
      }} onToggle={event => {
        const opening = event.newState === 'open'
        setOpen(opening)
        if (opening) { setModels(!modelOptions.length); setHighlighted(Math.max(0, modelOptions.findIndex(option => option.value === modelId))) }
        else cancelDrag()
      }}>
      <div ref={simpleView} className="chat-reasoning-view" aria-hidden={models} inert={models}>
        <div className={`chat-reasoning-heading${effort === undefined ? ' is-default' : ''}`}><strong>{effortLabel}</strong></div>
        <button ref={modelButton} type="button" className="chat-reasoning-model" aria-label="选择对话模型" aria-haspopup="listbox" aria-expanded={models} value={modelId}
          disabled={disabled || !optionCount} onClick={showModels}><span>{modelName}</span><ChevronRight size={12} /></button>
        {capability && <div className="chat-reasoning-slider" data-dragging={preview !== null || undefined} data-unset={unset || undefined} data-maximum={open && !models && maximum || undefined}>
          <span className="chat-reasoning-track" />
          <span className="chat-reasoning-fill" style={{ clipPath: `inset(0 calc(${1 - position} * (100cqi - 28px) + 12px) 0 0 round 12px)` }} />
          <span className="chat-reasoning-light"><i /><i /><i /></span>
          {!deepseek && levels.map((_, index) => <span key={index} className="chat-reasoning-tick" style={{ transform: `translateX(calc(${index / (levels.length - 1)} * (100cqi - 28px)))` }} />)}
          <span className="chat-reasoning-thumb" style={{ transform: `translateX(calc(${position} * (100cqi - 28px)))` }} />
          <input ref={slider} type="range" aria-label="推理强度档位" min={deepseek ? 1 : 0} max={deepseek ? 100 : levels.length - 1} step={1} value={deepseek ? numericEffort ?? 1 : Math.max(0, selected)}
            aria-valuetext={effortLabel} disabled={disabled}
            onPointerDown={event => { dragging.current = true; dragStart.current = effort; previewValue.current = { value: deepseek && unset ? Number(event.currentTarget.value) : effort }; setPreview(previewValue.current) }} onPointerUp={endDrag} onPointerCancel={cancelDrag}
            onKeyDown={event => { if (deepseek && unset && event.key === 'Home') { event.preventDefault(); onChange(1) } }}
            onChange={event => changeSlider(deepseek ? Number(event.target.value) : levels[Number(event.target.value)])} />
        </div>}
        {deepseek && <div className="chat-reasoning-secondary"><button type="button" aria-pressed={effort === undefined} disabled={disabled} onClick={() => onChange(undefined)}>默认</button><button type="button" aria-pressed={effort === 'none'} disabled={disabled} onClick={() => onChange('none')}>关闭</button>
          <input type="number" aria-label="自定义推理强度" aria-invalid={!!customError} aria-describedby={customError ? `${id}-error` : undefined} min={1} max={100} step={1} placeholder="1–100" value={customDraft} disabled={disabled}
            onChange={event => {
              const raw = event.target.value, number = event.target.valueAsNumber
              setCustomDraft(raw)
              if (!raw && !event.target.validity.badInput) { setCustomError(''); onChange(undefined) }
              else if (Number.isInteger(number) && number >= 1 && number <= 100) { setCustomError(''); onChange(number) }
              else setCustomError('请输入 1–100 的整数。')
            }} /></div>}
        {deepseek && customError && <span id={`${id}-error`} className="chat-reasoning-error" role="alert">{customError}</span>}
      </div>
      <div ref={modelView} className="chat-model-view" aria-hidden={!models} inert={!models}>
        {modelOptions.length ? <button type="button" className="chat-model-back" aria-label="返回推理强度" onClick={() => setModels(false)}><ChevronLeft size={14} /><span>模型</span></button> : <div className="chat-model-back"><span>模型</span></div>}
        <div ref={list} className="chat-model-list" role="listbox" aria-label="对话模型" tabIndex={0} aria-activedescendant={optionCount ? `${id}-model-${highlighted}` : undefined}
          onKeyDown={event => {
            if (event.target instanceof HTMLButtonElement) return
            if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
              event.preventDefault()
              if (optionCount) setHighlighted(current => event.key === 'Home' ? 0 : event.key === 'End' ? optionCount - 1 : (current + (event.key === 'ArrowDown' ? 1 : -1) + optionCount) % optionCount)
            } else if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); chooseModel(highlighted) }
          }}>
          {modelOptions.map((option, index) => <Fragment key={option.value}>
            {showModelGroups && modelStates[index].label !== modelStates[index - 1]?.label && <div className="chat-model-group-label" role="presentation">{modelStates[index].label}</div>}
            <div id={`${id}-model-${index}`} role="option" aria-selected={modelId === option.value} data-highlighted={highlighted === index || undefined} data-model-id={option.value} data-model-ready={modelStates[index].ready}
              onPointerMove={() => setHighlighted(index)} onClick={() => chooseModel(index)}><span className="chat-loaded-name" aria-label={option.label} title={option.label}>{option.label}</span>{modelStates[index].pending ? <Loader2 size={14} className="spin" role="img" aria-label={modelStates[index].label} /> : modelStates[index].ready && modelId === option.value && <Check size={14} />}{onUnloadModel && <button type="button" className="chat-instance-action" aria-label={`卸载 ${option.label}`} disabled={unloadingIds.includes(option.value)} onClick={event => { event.stopPropagation(); onUnloadModel(option.value) }}>{unloadingIds.includes(option.value) ? <Loader2 size={12} className="spin" /> : <Square size={12} />}</button>}</div>
          </Fragment>)}
          {!!loadOptions.length && <div className="chat-model-group-label" role="presentation">快捷加载</div>}
          {loadOptions.map((option, index) => <div id={`${id}-model-${modelOptions.length + index}`} key={`profile-${option.value}`} role="option" aria-selected={false} aria-disabled={option.disabled || undefined} data-highlighted={highlighted === modelOptions.length + index || undefined} data-profile-id={option.value}
            onPointerMove={() => setHighlighted(modelOptions.length + index)} onClick={() => chooseModel(modelOptions.length + index)}>
            <span className="chat-quick-load-label"><span aria-label={option.label} title={option.label}>{option.label}</span><span className="chat-quick-load-detail">{option.detail}</span>{option.error && <span className="chat-quick-load-error" role="alert">{option.error}</span>}</span>
            <span className="chat-quick-load-status">{option.loading ? <Loader2 size={12} className="spin" aria-hidden="true" /> : !option.disabled && <Play size={11} aria-hidden="true" />}<span>{option.status}</span></span><button type="button" className="chat-instance-action" aria-label={`另开实例 ${option.label}`} disabled={option.disabled} onClick={event => { event.stopPropagation(); onLoadProfile?.(option.value, true) }}><Plus size={13} /></button>
          </div>)}
        </div>
        {unloadError && <div className="chat-quick-load-error" role="alert">{unloadError}</div>}
      </div>
    </div>
  </>
}
