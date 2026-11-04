import { forwardRef, useImperativeHandle, useLayoutEffect, useRef, useState, type TextareaHTMLAttributes } from 'react'
import { MarkdownContent } from './MessageContent'

export interface MarkdownInputHandle { focus: () => void; resetSize: () => void }
interface Props extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'value'> {
  value: string
  previewLabel: string
  resizeLabel: string
}

/** Keep the draft verbatim; only its unfocused presentation is rendered. */
const MarkdownInput = forwardRef<MarkdownInputHandle, Props>(function MarkdownInput({ value, previewLabel, resizeLabel, onBlur, onFocus, onSelect, onCompositionStart, onCompositionEnd, style, disabled, ...props }, ref) {
  const input = useRef<HTMLTextAreaElement>(null)
  const previewElement = useRef<HTMLDivElement>(null)
  const handle = useRef<HTMLDivElement>(null)
  const [preview, setPreview] = useState(Boolean(value.trim()) && !props.autoFocus)
  const [height, setHeight] = useState<number>()
  const [manualHeight, setManualHeight] = useState<number | null>(null)
  const selection = useRef({ start: value.length, end: value.length, direction: 'none' as 'forward' | 'backward' | 'none' })
  const composing = useRef(false)
  const pendingFocus = useRef(false)
  const drag = useRef<{ pointer: number; y: number; height: number; previous: number | null } | null>(null)
  const [bounds, setBounds] = useState({ min: 50, max: 180 })

  function rememberSelection() {
    const node = input.current
    if (node) selection.current = { start: node.selectionStart, end: node.selectionEnd, direction: node.selectionDirection }
  }
  function focus() {
    if (disabled) return
    if (preview) { pendingFocus.current = true; setPreview(false) }
    else input.current?.focus({ preventScroll: true })
  }
  useImperativeHandle(ref, () => ({ focus, resetSize: () => setManualHeight(null) }))
  useLayoutEffect(() => {
    if (!value) { setPreview(false); setManualHeight(null) }
  }, [value])
  useLayoutEffect(() => {
    if (!preview && pendingFocus.current) {
      pendingFocus.current = false
      input.current?.focus({ preventScroll: true })
      input.current?.setSelectionRange(selection.current.start, selection.current.end, selection.current.direction)
    }
  }, [preview])

  useLayoutEffect(() => {
    const node = input.current
    if (!node) return
    const panel = node.closest<HTMLElement>('.chat-panel')
    const transcript = panel?.querySelector<HTMLElement>('.chat-messages')
    function measure() {
      if (!node || !panel?.clientHeight) return
      const css = getComputedStyle(node)
      const min = parseFloat(css.minHeight) || 50
      const line = parseFloat(css.lineHeight) || 24
      const currentHeight = (preview ? previewElement.current : node)?.getBoundingClientRect().height || min
      // Leave four lines of the transcript visible; the rest of its space can
      // be borrowed by the bottom composer. An inline editor stays in its pane.
      const available = node.closest('.chat-composer')
        ? currentHeight + (transcript?.clientHeight || 0) - line * 4
        : (transcript?.clientHeight || panel.clientHeight) - line * 3
      const max = Math.max(min, available)
      setBounds(previous => previous.min === min && previous.max === max ? previous : { min, max })
      if (manualHeight !== null) {
        setHeight(Math.max(min, Math.min(max, manualHeight)))
      } else if (!preview || height === undefined) {
        const surface = preview ? previewElement.current! : node
        surface.style.height = 'auto'
        const next = Math.max(min, Math.min(max, 180, surface.scrollHeight))
        surface.style.height = `${next}px`
        setHeight(next)
      } else setHeight(previous => Math.max(min, Math.min(max, previous || min)))
    }
    measure()
    const observer = new ResizeObserver(measure)
    if (panel) observer.observe(panel)
    return () => observer.disconnect()
  }, [value, preview, manualHeight])

  function endDrag(cancel: boolean) {
    const active = drag.current
    if (!active) return
    drag.current = null
    if (cancel) setManualHeight(active.previous)
    if (handle.current?.hasPointerCapture(active.pointer)) handle.current.releasePointerCapture(active.pointer)
  }
  useLayoutEffect(() => {
    const cancel = () => endDrag(true)
    window.addEventListener('blur', cancel)
    return () => window.removeEventListener('blur', cancel)
  }, [])

  return <>
    <div ref={handle} className="chat-input-resize" role="separator" aria-label={resizeLabel} aria-orientation="horizontal" aria-valuemin={Math.round(bounds.min)} aria-valuemax={Math.round(bounds.max)} aria-valuenow={Math.round(height || bounds.min)} tabIndex={0}
      onPointerDown={event => {
        if (event.button !== 0) return
        event.preventDefault(); event.currentTarget.focus({ preventScroll: true })
        drag.current = { pointer: event.pointerId, y: event.clientY, height: height || bounds.min, previous: manualHeight }
        event.currentTarget.setPointerCapture(event.pointerId)
      }}
      onPointerMove={event => {
        const active = drag.current
        if (active?.pointer === event.pointerId) setManualHeight(Math.max(bounds.min, Math.min(bounds.max, active.height + active.y - event.clientY)))
      }}
      onPointerUp={() => endDrag(false)} onPointerCancel={() => endDrag(true)} onLostPointerCapture={() => endDrag(true)}
      onKeyDown={event => {
        if (event.key === 'Escape' && drag.current) { event.preventDefault(); event.stopPropagation(); endDrag(true); return }
        const step = parseFloat(getComputedStyle(input.current!).lineHeight) || 24
        const next = event.key === 'ArrowUp' ? (height || bounds.min) + step : event.key === 'ArrowDown' ? (height || bounds.min) - step : event.key === 'Home' ? bounds.min : event.key === 'End' ? bounds.max : null
        if (next !== null) { event.preventDefault(); event.stopPropagation(); setManualHeight(Math.max(bounds.min, Math.min(bounds.max, next))) }
      }} />
    <textarea {...props} ref={input} value={value} disabled={disabled} hidden={preview} style={{ ...style, height, maxHeight: 'none', resize: 'none' }}
      onBlur={event => { rememberSelection(); if (value.trim() && !composing.current) setPreview(true); onBlur?.(event) }}
      onFocus={event => { setPreview(false); onFocus?.(event) }}
      onSelect={event => { rememberSelection(); onSelect?.(event) }}
      onCompositionStart={event => { composing.current = true; onCompositionStart?.(event) }}
      onCompositionEnd={event => { composing.current = false; if (document.activeElement !== event.currentTarget && value.trim()) setPreview(true); onCompositionEnd?.(event) }} />
    {preview && <div ref={previewElement} className="chat-input-preview" style={{ height: height ?? 'var(--input-min-height, 50px)', maxHeight: 'none' }} tabIndex={disabled ? -1 : 0} role="region" aria-label={previewLabel}
      onFocus={event => { if (event.target === event.currentTarget) focus() }}
      onClick={event => { if (!(event.target as Element).closest('a, button, input, textarea, summary, iframe')) focus() }}>
      <MarkdownContent text={value} />
    </div>}
  </>
})

export default MarkdownInput
