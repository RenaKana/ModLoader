import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react'
import { createPortal } from 'react-dom'
import { Pin, PinOff, Trash2 } from 'lucide-react'
import { IconButton } from './components'
import type { ConversationSummary } from './chat-history'
import { ChatTitle } from './ChatTitle'
import { reorderLayoutRect, useReorderMotion } from './useReorderMotion'
import './chat-history-panel.css'

type Placement = { id: string; before: string | null }
type Drag = Placement & { x: number; y: number; offsetX: number; width: number; height: number }
type Gesture = { id: string; pointerId: number; capture: HTMLElement; x: number; y: number; moved: boolean; swipe: boolean; held: boolean }

function previewOrder(items: ConversationSummary[], placement: Placement | null) {
  const active = placement && items.find(item => item.id === placement.id)
  if (!placement || !active) return items
  const result = items.filter(item => item.id !== active.id)
  const last = result.filter(item => Boolean(item.pinned) === Boolean(active.pinned)).at(-1)
  const before = placement.before && result.find(item => item.id === placement.before)
  const index = before ? result.indexOf(before) : last ? result.indexOf(last) + 1 : active.pinned ? 0 : result.length
  result.splice(index, 0, active)
  return result
}
export default function ChatHistoryPanel({ items, currentId, disabled, reset, onOpen, onDelete, onPin, onReorder }: {
  items: ConversationSummary[]; currentId: string; disabled: boolean; reset: number
  onOpen: (id: string) => void; onDelete: (ids: string[]) => void
  onPin: (ids: string[], pinned: boolean) => void; onReorder: (id: string, before: string | null) => void
}) {
  const [selected, setSelected] = useState<string[]>([])
  const [drag, setDrag] = useState<Drag | null>(null)
  const [pendingDrop, setPendingDrop] = useState<Placement | null>(null)
  const [swipeOffset, setSwipeOffset] = useState<{ id: string; value: number } | null>(null)
  const dragging = useRef<Drag | null>(null)
  const gesture = useRef<Gesture | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const suppressClick = useRef<string | null>(null)
  const anchor = useRef<string | null>(null)
  const pointerY = useRef(0)
  const list = useRef<HTMLDivElement>(null)
  const rendered = previewOrder(items, drag || pendingDrop)
  const orderKey = JSON.stringify(rendered.map(item => item.id))
  const capture = useReorderMotion(list, 'data-history-id', orderKey, drag?.id)
  const selecting = selected.length > 0
  const targets = selecting ? selected : items.some(item => item.id === currentId) ? [currentId] : []
  const allPinned = targets.length > 0 && targets.every(id => items.find(item => item.id === id)?.pinned)
  const cancel = () => { setSelected([]); anchor.current = null }
  useEffect(() => { cancel(); endDrag(false) }, [reset])
  useEffect(() => {
    setSelected(previous => previous.filter(id => items.some(item => item.id === id)))
    const active = dragging.current
    if (active && !items.some(item => item.id === active.id)) endDrag(false)
  }, [items])
  useEffect(() => { if (disabled) endDrag(false) }, [disabled])
  // Keep the dropped layout through the parent's save transition; a failed save
  // restores the unchanged items when that transition unlocks the panel.
  useEffect(() => {
    if (pendingDrop && !disabled) {
      if (orderKey !== JSON.stringify(items.map(item => item.id))) capture()
      setPendingDrop(null)
    }
  }, [pendingDrop, disabled, items, orderKey, capture])
  const stopTimer = () => { if (timer.current) clearTimeout(timer.current); timer.current = null }
  function clearGesture() {
    stopTimer()
    const active = gesture.current
    gesture.current = null; setSwipeOffset(null)
    if (active?.capture.hasPointerCapture(active.pointerId)) active.capture.releasePointerCapture(active.pointerId)
  }
  useEffect(() => () => {
    stopTimer()
    const active = gesture.current
    gesture.current = null; dragging.current = null
    if (active?.capture.hasPointerCapture(active.pointerId)) active.capture.releasePointerCapture(active.pointerId)
  }, [])
  function toggle(id: string) {
    anchor.current = id
    setSelected(previous => previous.includes(id) ? previous.filter(value => value !== id) : [...previous, id])
  }
  function choose(id: string, event: MouseEvent) {
    const additive = event.ctrlKey || event.metaKey
    if (event.shiftKey) {
      const start = items.findIndex(item => item.id === (anchor.current || currentId))
      const end = items.findIndex(item => item.id === id)
      const range = items.slice(Math.min(start < 0 ? end : start, end), Math.max(start, end) + 1).map(item => item.id)
      setSelected(previous => additive ? [...new Set([...previous, ...range])] : range)
      anchor.current ??= start < 0 ? id : items[start].id
    } else if (additive) {
      toggle(id)
    } else { setSelected([]); anchor.current = id; onOpen(id) }
  }
  function startDrag(id: string, x: number, y: number, bounds: DOMRect) {
    pointerY.current = y
    const index = items.findIndex(item => item.id === id)
    const next = items[index + 1]
    dragging.current = { id, before: next && Boolean(next.pinned) === Boolean(items[index].pinned) ? next.id : null,
      x, y, offsetX: x - bounds.left, width: bounds.width, height: bounds.height }
    capture()
    setDrag(dragging.current)
  }
  function moveDrag(x: number, y: number) {
    const active = dragging.current, container = list.current
    if (!active || !container) return
    pointerY.current = y
    const pinned = Boolean(items.find(item => item.id === active.id)?.pinned)
    const group = items.filter(item => item.id !== active.id && Boolean(item.pinned) === pinned)
    const before = group.find(item => {
      const row = container.querySelector<HTMLElement>(`[data-history-id="${item.id}"]`)
      const box = row && reorderLayoutRect(row)
      return box && y < box.top + box.height / 2
    })?.id ?? null
    if (before !== active.before) capture()
    if (before !== active.before || x !== active.x || y !== active.y) { dragging.current = { ...active, before, x, y }; setDrag(dragging.current) }
  }
  useEffect(() => {
    if (!drag?.id) return
    let frame = 0
    const scroll = () => {
      const container = list.current
      if (!dragging.current || !container) return
      if (container.closest('[inert]')) { endDrag(false); return }
      const bounds = container.getBoundingClientRect(), y = pointerY.current
      const before = container.scrollTop
      if (y < bounds.top + 24) container.scrollTop -= 10
      else if (y > bounds.bottom - 24) container.scrollTop += 10
      if (container.scrollTop !== before) moveDrag(dragging.current.x, y)
      frame = requestAnimationFrame(scroll)
    }
    frame = requestAnimationFrame(scroll)
    return () => cancelAnimationFrame(frame)
  }, [drag?.id, items])
  function endDrag(commit: boolean) {
    const active = dragging.current
    clearGesture()
    if (!active) return
    capture()
    dragging.current = null; setDrag(null)
    if (active && commit) {
      const index = items.findIndex(item => item.id === active.id)
      const next = items[index + 1]
      const previousBefore = next && Boolean(next.pinned) === Boolean(items[index].pinned) ? next.id : null
      if (active.before !== previousBefore) { setPendingDrop({ id: active.id, before: active.before }); onReorder(active.id, active.before) }
    }
  }
  function pointerDown(id: string, event: PointerEvent<HTMLButtonElement>) {
    if (event.button !== 0 || disabled || gesture.current) return
    suppressClick.current = null
    if (event.ctrlKey || event.metaKey || event.shiftKey) return
    const button = event.currentTarget
    gesture.current = { id, pointerId: event.pointerId, capture: button, x: event.clientX, y: event.clientY, moved: false, swipe: false, held: false }
    button.setPointerCapture(event.pointerId)
    stopTimer()
    timer.current = setTimeout(() => {
      const active = gesture.current, container = list.current
      if (active && container && !active.moved && document.hasFocus() && !button.closest('[inert]')) {
        active.held = true; suppressClick.current = id
        // The stable list keeps capture when React moves the dragged row.
        active.capture = container; container.setPointerCapture(active.pointerId)
        startDrag(id, active.x, active.y, button.getBoundingClientRect())
      }
    }, 450)
  }
  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const active = gesture.current
    if (!active || event.pointerId !== active.pointerId) return
    const dx = event.clientX - active.x, dy = event.clientY - active.y
    if (Math.hypot(dx, dy) > 8) { stopTimer(); active.moved = true; suppressClick.current = active.id }
    if (active.held) { moveDrag(event.clientX, event.clientY); return }
    if (Math.abs(dx) > 10 && Math.abs(dx) > Math.abs(dy) * 1.2) active.swipe = true
    if (active.swipe) setSwipeOffset({ id: active.id, value: Math.max(-40, Math.min(0, dx)) })
  }
  function pointerUp(event: PointerEvent<HTMLDivElement>) {
    const active = gesture.current
    if (!active || event.pointerId !== active.pointerId) return
    if (active.held) endDrag(active.moved)
    else {
      if (active.swipe && event.clientX - active.x < -20) toggle(active.id)
      clearGesture()
    }
  }
  function cancelPointer() {
    if (gesture.current) suppressClick.current = gesture.current.id
    endDrag(false)
  }
  useEffect(() => {
    if (!drag?.id) return
    const cancelDrag = () => endDrag(false)
    const hidden = () => { if (document.hidden) cancelDrag() }
    window.addEventListener('blur', cancelDrag)
    document.addEventListener('visibilitychange', hidden)
    return () => { window.removeEventListener('blur', cancelDrag); document.removeEventListener('visibilitychange', hidden) }
  }, [drag?.id])
  useEffect(() => {
    if (!selecting && !drag) return
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || document.querySelector('[role=dialog]') || list.current?.closest('[inert]')) return
      event.preventDefault(); event.stopPropagation(); cancel(); endDrag(false)
    }
    document.addEventListener('keydown', escape, true)
    return () => document.removeEventListener('keydown', escape, true)
  }, [selecting, drag?.id])
  const dragItem = drag && items.find(item => item.id === drag.id)
  return <><aside className="chat-history" aria-label="对话历史" onKeyDown={event => {
    if (!disabled && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') { event.preventDefault(); event.stopPropagation(); setSelected(items.map(item => item.id)) }
  }}>
    <div className="chat-history-heading"><h3>{selecting ? `已选择 ${selected.length} 条` : '对话历史'}</h3><div>
      <IconButton icon={allPinned ? PinOff : Pin} label={allPinned ? '取消置顶会话' : '置顶会话'} disabled={disabled || !targets.length} onClick={() => onPin(targets, !allPinned)} />
      <IconButton icon={Trash2} label={selecting ? '删除所选会话' : '删除当前会话'} className="danger" disabled={disabled || !targets.length} onClick={() => onDelete(targets)} />
    </div></div>
    <div className="chat-history-list" ref={list} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={cancelPointer}
      onLostPointerCapture={event => { if (gesture.current?.capture === event.target) cancelPointer() }}
      onScroll={() => { const active = dragging.current; if (active) moveDrag(active.x, active.y) }}>{rendered.length ? rendered.map((item, index) => {
      return <HistoryRow key={item.id} item={item} current={item.id === currentId} disabled={disabled} selecting={selecting} selected={selected.includes(item.id)}
        dragging={drag?.id === item.id} dropBefore={drag?.id === item.id} groupStart={index > 0 && Boolean(rendered[index - 1].pinned) !== Boolean(item.pinned)}
        offset={swipeOffset?.id === item.id ? swipeOffset.value : null}
        onChoose={event => { if (suppressClick.current === item.id) { suppressClick.current = null; return }; choose(item.id, event) }} onPointerDown={event => pointerDown(item.id, event)} />
    }) : <p className="chat-history-empty">暂无会话</p>}</div>
  </aside>{drag && dragItem && createPortal(<div className="chat-history-drag-preview" aria-hidden="true" style={{ width: drag.width,
    transform: `translate3d(${Math.max(8, Math.min(window.innerWidth - drag.width * 1.01 - 8, drag.x - drag.offsetX))}px, ${Math.max(8, Math.min(window.innerHeight - drag.height * 1.01 - 8, drag.y - drag.height * 1.01 - 12))}px, 0)` }}>
    <div className="chat-history-item"><HistoryContent item={dragItem} /></div>
  </div>, document.body)}</>
}

function HistoryRow({ item, current, disabled, selecting, selected, dragging, dropBefore, groupStart, offset, onChoose, onPointerDown }: {
  item: ConversationSummary; current: boolean; disabled: boolean; selecting: boolean; selected: boolean
  dragging: boolean; dropBefore: boolean; groupStart: boolean; offset: number | null
  onChoose: (event: MouseEvent) => void; onPointerDown: (event: PointerEvent<HTMLButtonElement>) => void
}) {
  return <div data-history-id={item.id} data-pinned={Boolean(item.pinned)} className={`chat-history-row ${dragging ? 'is-dragging' : ''} ${dropBefore ? 'drop-before' : ''} ${groupStart ? 'group-start' : ''}`}>
    <button className="chat-history-item" data-tooltip={item.title} data-tooltip-format="chat-title" aria-label={item.title} aria-current={current ? 'page' : undefined} aria-pressed={selecting ? selected : undefined} disabled={disabled}
      style={{ transform: dragging ? undefined : `translateX(${offset ?? 0}px)`, transition: offset === null ? undefined : 'none' }}
      onContextMenu={event => event.preventDefault()}
      onPointerDown={onPointerDown} onClick={onChoose}>
      <HistoryContent item={item} />
    </button>
  </div>
}

function HistoryContent({ item }: { item: ConversationSummary }) {
  return <div><strong className="chat-history-title">{item.pinned && <Pin className="chat-history-pin" size={12} aria-label="已置顶" />}<ChatTitle text={item.title} /></strong><span>{item.last_message_at == null ? '草稿' : new Date(item.last_message_at).toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })} · {item.message_count} 条消息</span></div>
}
