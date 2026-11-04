import { useEffect, useRef, useState, type MouseEvent, type PointerEvent, type RefObject } from 'react'
import { reorderLayoutRect, useReorderMotion } from './useReorderMotion'

type Placement = { id: string; before: string | null }
export type ReorderDrag = Placement & { x: number; y: number; offsetX: number; width: number }
type Gesture = { id: string; pointerId: number; capture: HTMLElement; x: number; y: number; moved: boolean }

// The library uses the history panel's hold threshold, capture transfer, FLIP
// displacement and edge scrolling, without changing its existing selection UI.
export function useLongPressReorder<T extends { id: string }>({ items, container, disabled, reset, pageScroll = false, onReorder, onMoveBeforeHold, onGestureEnd }: {
  items: T[]; container: RefObject<HTMLDivElement | null>; disabled?: boolean; reset: string; pageScroll?: boolean
  onReorder: (items: T[]) => void
  onMoveBeforeHold?: (id: string, dx: number, dy: number) => void; onGestureEnd?: () => void
}) {
  const [drag, setDrag] = useState<ReorderDrag | null>(null)
  const dragging = useRef<ReorderDrag | null>(null)
  const gesture = useRef<Gesture | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const suppressClick = useRef<string | null>(null)
  const rendered = placed(items, drag)
  const ids = JSON.stringify(items.map(item => item.id))
  const capture = useReorderMotion(container, 'data-reorder-id', JSON.stringify(rendered.map(item => item.id)), drag?.id)
  const stopTimer = () => { if (timer.current) clearTimeout(timer.current); timer.current = null }

  function endDrag(commit: boolean) {
    stopTimer()
    const active = dragging.current, press = gesture.current
    gesture.current = null; dragging.current = null
    if (active) { capture(); setDrag(null) }
    if (press?.capture.hasPointerCapture(press.pointerId)) press.capture.releasePointerCapture(press.pointerId)
    onGestureEnd?.()
    if (active && commit) {
      const next = placed(items, active)
      if (next.some((item, index) => item.id !== items[index]?.id)) onReorder(next)
    }
  }
  function cancel() {
    if (gesture.current) suppressClick.current = gesture.current.id
    endDrag(false)
  }
  useEffect(() => { cancel() }, [disabled, reset, ids])
  useEffect(() => () => {
    stopTimer()
    const press = gesture.current
    gesture.current = null; dragging.current = null
    if (press?.capture.hasPointerCapture(press.pointerId)) press.capture.releasePointerCapture(press.pointerId)
  }, [])

  function moveDrag(x: number, y: number) {
    const active = dragging.current, root = container.current
    if (!active || !root) return
    const before = [...root.querySelectorAll<HTMLElement>('[data-reorder-id]')].find(row => {
      if (row.dataset.reorderId === active.id) return false
      const box = reorderLayoutRect(row)
      return y < box.top + box.height / 2
    })?.dataset.reorderId ?? null
    if (before !== active.before) capture()
    if (before !== active.before || x !== active.x || y !== active.y) {
      dragging.current = { ...active, before, x, y }; setDrag(dragging.current)
    }
  }
  useEffect(() => {
    if (!drag) return
    let frame = 0
    const scroll = () => {
      const active = dragging.current, root = container.current
      if (!active || !root) return
      if (root.closest('[inert], [hidden]')) { cancel(); return }
      const scroller = pageScroll ? document.scrollingElement : root
      if (!scroller) return
      const bounds = root.getBoundingClientRect(), previous = scroller.scrollTop
      const top = pageScroll ? 0 : Math.max(0, bounds.top)
      const bottom = pageScroll ? innerHeight : Math.min(innerHeight, bounds.bottom)
      if (active.y < top + 24) scroller.scrollTop -= 10
      else if (active.y > bottom - 24) scroller.scrollTop += 10
      if (scroller.scrollTop !== previous) moveDrag(active.x, active.y)
      frame = requestAnimationFrame(scroll)
    }
    const scrolled = () => { const active = dragging.current; if (active) moveDrag(active.x, active.y) }
    const hidden = () => { if (document.hidden) cancel() }
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.preventDefault(); event.stopPropagation(); cancel()
    }
    frame = requestAnimationFrame(scroll)
    window.addEventListener('scroll', scrolled, true)
    window.addEventListener('blur', cancel)
    document.addEventListener('visibilitychange', hidden)
    document.addEventListener('keydown', escape, true)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', scrolled, true)
      window.removeEventListener('blur', cancel)
      document.removeEventListener('visibilitychange', hidden)
      document.removeEventListener('keydown', escape, true)
    }
  }, [drag?.id, ids, pageScroll])

  function pointerDown(id: string, event: PointerEvent<HTMLElement>) {
    suppressClick.current = null
    if (disabled || !event.isPrimary || event.button !== 0 || gesture.current || event.ctrlKey || event.metaKey || event.shiftKey) return
    const target = event.target as Element, handle = event.currentTarget
    // Card actions and editable values retain their own pointer behavior.
    const control = target.closest('button, input, select, textarea, a, [role="button"], [contenteditable="true"]')
    if (control && control !== handle) return
    gesture.current = { id, pointerId: event.pointerId, capture: handle, x: event.clientX, y: event.clientY, moved: false }
    handle.setPointerCapture(event.pointerId)
    timer.current = setTimeout(() => {
      const press = gesture.current, root = container.current
      if (!press || press.moved || !root || !document.hasFocus() || handle.closest('[inert], [hidden]')) return
      suppressClick.current = id
      press.capture = root; root.setPointerCapture(press.pointerId)
      const bounds = handle.getBoundingClientRect(), index = items.findIndex(item => item.id === id)
      dragging.current = { id, before: items[index + 1]?.id ?? null, x: press.x, y: press.y, offsetX: press.x - bounds.left, width: bounds.width }
      capture(); setDrag(dragging.current)
      window.getSelection()?.removeAllRanges()
    }, 450)
  }
  return {
    drag, rendered, pointerDown,
    listProps: {
      onPointerMove(event: PointerEvent<HTMLDivElement>) {
        const press = gesture.current
        if (!press || event.pointerId !== press.pointerId) return
        const dx = event.clientX - press.x, dy = event.clientY - press.y
        if (Math.hypot(dx, dy) > 8) { stopTimer(); press.moved = true; suppressClick.current = press.id }
        if (dragging.current) { event.preventDefault(); moveDrag(event.clientX, event.clientY) }
        else onMoveBeforeHold?.(press.id, dx, dy)
      },
      onPointerUp(event: PointerEvent<HTMLDivElement>) {
        if (gesture.current?.pointerId === event.pointerId) endDrag(gesture.current.moved)
      },
      onPointerCancel: cancel,
      onLostPointerCapture(event: PointerEvent<HTMLDivElement>) { if (gesture.current?.capture === event.target) cancel() },
      onClickCapture(event: MouseEvent<HTMLDivElement>) {
        if (!suppressClick.current) return
        const row = (event.target as Element).closest<HTMLElement>('[data-reorder-id]')
        if (row?.dataset.reorderId === suppressClick.current || event.target === container.current) {
          suppressClick.current = null; event.preventDefault(); event.stopPropagation()
        }
      },
    },
  }
}

function placed<T extends { id: string }>(items: T[], placement: Placement | null) {
  const active = placement && items.find(item => item.id === placement.id)
  if (!placement || !active) return items
  const next = items.filter(item => item.id !== active.id)
  const index = placement.before === null ? next.length : next.findIndex(item => item.id === placement.before)
  if (index < 0) return items
  next.splice(index, 0, active)
  return next
}
