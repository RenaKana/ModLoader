import { useEffect, useId, useRef, useState } from 'react'
import { flushSync } from 'react-dom'
import { ChatTitle } from './ChatTitle'
import './tooltip.css'

const HOVER_DELAY = 300
const WARM_DELAY = 600
const EXIT_DELAY = 120
const EDGE_GAP = 8
const TRIGGER_GAP = 8

export default function Tooltip() {
  const element = useRef<HTMLDivElement>(null)
  const id = useId()
  const [content, setContent] = useState<{ text: string; title: boolean } | null>(null)

  useEffect(() => {
    const tooltip = element.current!
    const hover = window.matchMedia('(hover: hover) and (pointer: fine)')
    let target: HTMLElement | null = null
    let hovered: HTMLElement | null = null
    let focused: HTMLElement | null = null
    let dismissed: HTMLElement | null = null
    let pointerX = 0, pointerY = 0
    let timer: number | undefined
    let exitTimer: number | undefined
    let tooltipHovered = false
    let warmUntil = 0
    const find = (node: EventTarget | null) => node instanceof Element ? node.closest<HTMLElement>('[data-tooltip]') : null
    const inside = (node: EventTarget | null) => node instanceof Node && tooltip.contains(node)
    const available = (node: HTMLElement) => node.isConnected && !node.closest('[hidden], [inert]') && node.getClientRects().length > 0

    function updateContent(text: string) {
      const title = target?.dataset.tooltipFormat === 'chat-title'
      if (title) tooltip.dataset.tooltipFormat = 'chat-title'
      else delete tooltip.dataset.tooltipFormat
      // The popover is positioned immediately after its content changes.
      flushSync(() => setContent({ text, title }))
    }

    function position() {
      if (!target) return
      if (!available(target)) { dismiss(); return }
      const anchor = target.getBoundingClientRect()
      const above = anchor.top >= tooltip.offsetHeight + TRIGGER_GAP + EDGE_GAP || anchor.top > innerHeight - anchor.bottom
      if (target.dataset.tooltipFormat === 'chat-title') {
        const room = above ? anchor.top - TRIGGER_GAP - EDGE_GAP : innerHeight - anchor.bottom - TRIGGER_GAP - EDGE_GAP
        tooltip.style.setProperty('--tooltip-max-height', `${Math.max(0, room)}px`)
      }
      const width = tooltip.offsetWidth, height = tooltip.offsetHeight
      const left = Math.max(EDGE_GAP, Math.min(anchor.left + (anchor.width - width) / 2, innerWidth - width - EDGE_GAP))
      const top = Math.max(EDGE_GAP, Math.min(above ? anchor.top - height - TRIGGER_GAP : anchor.bottom + TRIGGER_GAP, innerHeight - height - EDGE_GAP))
      tooltip.style.left = `${left}px`
      tooltip.style.top = `${top}px`
      tooltip.style.transformOrigin = `${anchor.left + anchor.width / 2 - left}px ${above ? '100%' : '0'}`
      tooltip.style.setProperty('--tooltip-shift', above ? '3px' : '-3px')
    }

    const resize = new ResizeObserver(position)
    const mutation = new MutationObserver(records => {
      if (!target) return
      if (!target.isConnected || target.closest('[hidden], [inert]')) { dismiss(); return }
      if (records.some(record => record.target === target && ['data-tooltip', 'data-tooltip-format'].includes(record.attributeName || ''))) {
        const text = target.dataset.tooltip?.trim()
        if (!text) { dismiss(); return }
        updateContent(text)
        position()
      }
    })

    function hide() {
      window.clearTimeout(timer)
      window.clearTimeout(exitTimer)
      timer = undefined
      exitTimer = undefined
      resize.disconnect()
      mutation.disconnect()
      if (target) {
        const descriptions = target.getAttribute('aria-describedby')?.split(/\s+/).filter(value => value && value !== id)
        if (descriptions?.length) target.setAttribute('aria-describedby', descriptions.join(' '))
        else target.removeAttribute('aria-describedby')
      }
      if (tooltip.matches(':popover-open')) {
        warmUntil = performance.now() + WARM_DELAY
        tooltip.hidePopover()
      }
      tooltip.setAttribute('aria-hidden', 'true')
      target = null
      tooltipHovered = false
    }

    function dismiss() {
      dismissed = target && (focused === target || hovered === target && target.contains(document.elementFromPoint(pointerX, pointerY))) ? target : null
      hovered = focused = null
      hide()
    }

    function schedule(next: HTMLElement | null, keyboard = false) {
      window.clearTimeout(exitTimer)
      if (next === target) return
      const instant = keyboard || tooltip.matches(':popover-open') || performance.now() < warmUntil
      hide()
      if (!next || next === dismissed || !next.dataset.tooltip?.trim()) return
      target = next
      const show = () => {
        const text = target?.dataset.tooltip?.trim()
        if (!target || !text || document.hidden || !available(target)) { hide(); return }
        updateContent(text)
        tooltip.toggleAttribute('data-instant', instant)
        tooltip.removeAttribute('aria-hidden')
        tooltip.showPopover()
        position()
        const descriptions = target.getAttribute('aria-describedby')?.split(/\s+/).filter(Boolean) || []
        target.setAttribute('aria-describedby', [...descriptions, id].join(' '))
        resize.observe(target)
        resize.observe(tooltip)
        mutation.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-tooltip', 'data-tooltip-format', 'hidden', 'inert'] })
      }
      if (instant) show()
      else timer = window.setTimeout(show, HOVER_DELAY)
    }

    function reconcile() {
      window.clearTimeout(exitTimer)
      if (tooltipHovered) return
      const next = hovered || focused
      if (next) schedule(next, Boolean(focused && !hovered))
      else if (tooltip.matches(':popover-open')) exitTimer = window.setTimeout(hide, EXIT_DELAY)
      else hide()
    }

    const move = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse') return
      pointerX = event.clientX; pointerY = event.clientY
    }
    const over = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse' || !hover.matches) return
      move(event)
      if (inside(event.target)) { tooltipHovered = true; window.clearTimeout(exitTimer); return }
      const next = find(event.target)
      if (next === hovered) return
      hovered = next
      if (next !== dismissed) dismissed = null
      reconcile()
    }
    const out = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse' || !hover.matches) return
      move(event)
      const intoTooltip = inside(event.relatedTarget)
      if (inside(event.target) && intoTooltip) return
      const next = intoTooltip ? null : find(event.relatedTarget)
      if (find(event.target) === dismissed && next !== dismissed) dismissed = null
      if (next === hovered && !inside(event.target) && !intoTooltip) return
      if (hovered === dismissed && hovered !== next) dismissed = null
      tooltipHovered = intoTooltip
      hovered = next
      reconcile()
    }
    const focus = (event: FocusEvent) => {
      focused = event.target instanceof Element && event.target.matches(':focus-visible') ? find(event.target) : null
      if (!focused) return
      dismissed = null
      schedule(focused, true)
    }
    const blur = (event: FocusEvent) => {
      if (focused?.contains(event.relatedTarget as Node | null)) return
      if (dismissed && find(event.target) === dismissed && !dismissed.matches(':hover')) dismissed = null
      focused = null
      reconcile()
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !target) return
      event.preventDefault()
      event.stopPropagation()
      dismiss()
    }
    const visibility = () => { if (document.hidden) dismiss() }
    const press = (event: PointerEvent) => { if (!inside(event.target)) dismiss() }
    const scroll = (event: Event) => {
      if (inside(event.target)) return
      // Auto-follow can shift a footer by a subpixel while it stays under the mouse.
      if (target && hovered === target && available(target) && target.contains(document.elementFromPoint(pointerX, pointerY))) {
        if (tooltip.matches(':popover-open')) position()
        return
      }
      dismiss()
    }

    document.addEventListener('pointerover', over)
    document.addEventListener('pointerout', out)
    document.addEventListener('pointermove', move)
    document.addEventListener('focusin', focus)
    document.addEventListener('focusout', blur)
    document.addEventListener('keydown', escape, true)
    document.addEventListener('pointerdown', press, true)
    document.addEventListener('scroll', scroll, true)
    document.addEventListener('visibilitychange', visibility)
    window.addEventListener('resize', dismiss)
    window.addEventListener('blur', dismiss)
    window.addEventListener('pagehide', dismiss)
    return () => {
      hide()
      document.removeEventListener('pointerover', over)
      document.removeEventListener('pointerout', out)
      document.removeEventListener('pointermove', move)
      document.removeEventListener('focusin', focus)
      document.removeEventListener('focusout', blur)
      document.removeEventListener('keydown', escape, true)
      document.removeEventListener('pointerdown', press, true)
      document.removeEventListener('scroll', scroll, true)
      document.removeEventListener('visibilitychange', visibility)
      window.removeEventListener('resize', dismiss)
      window.removeEventListener('blur', dismiss)
      window.removeEventListener('pagehide', dismiss)
    }
  }, [id])

  return <div ref={element} id={id} className="app-tooltip" role="tooltip" popover="manual" aria-hidden="true">{content?.title ? <ChatTitle text={content.text} /> : content?.text}</div>
}
