import { useCallback, useEffect, useLayoutEffect, useRef, type RefObject } from 'react'

// Hit testing uses the destination layout, not the in-flight FLIP translation.
export function reorderLayoutRect(node: HTMLElement): DOMRect {
  const box = node.getBoundingClientRect()
  const transform = getComputedStyle(node).transform
  const offset = transform === 'none' ? null : new DOMMatrixReadOnly(transform)
  return new DOMRect(box.x - (offset?.m41 ?? 0), box.y - (offset?.m42 ?? 0), box.width, box.height)
}

// Capture immediately before changing the preview order. Retarget from the current
// visual position so another pointer move can interrupt the previous displacement.
export function useReorderMotion(container: RefObject<HTMLElement | null>, keyAttribute: string, orderKey: string, skippedId?: string) {
  const pending = useRef<Map<string, DOMRect> | null>(null)
  const animations = useRef(new Map<HTMLElement, Animation>())
  const capture = useCallback(() => {
    if (pending.current) return
    pending.current = new Map([...container.current?.querySelectorAll<HTMLElement>(`[${keyAttribute}]`) ?? []]
      .map(node => [node.getAttribute(keyAttribute)!, node.getBoundingClientRect()]))
  }, [container, keyAttribute])
  useLayoutEffect(() => {
    const before = pending.current
    pending.current = null
    if (!before) return
    for (const animation of animations.current.values()) animation.cancel()
    animations.current.clear()
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const root = container.current
    if (!root) return
    const easing = getComputedStyle(root).getPropertyValue('--ease-out').trim()
    for (const node of root.querySelectorAll<HTMLElement>(`[${keyAttribute}]`)) {
      const id = node.getAttribute(keyAttribute)!
      const old = before.get(id)
      if (!old || id === skippedId) continue
      const next = node.getBoundingClientRect(), x = old.left - next.left, y = old.top - next.top
      if (Math.abs(x) < .5 && Math.abs(y) < .5) continue
      const animation = node.animate([
        { transform: `translate3d(${x}px, ${y}px, 0)` },
        { transform: 'translate3d(0, 0, 0)' },
      ], { duration: 160, easing })
      animation.id = 'reorder-position'
      animations.current.set(node, animation)
      animation.onfinish = () => { if (animations.current.get(node) === animation) animations.current.delete(node) }
    }
  }, [container, keyAttribute, orderKey, skippedId])
  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)')
    const stop = () => { for (const animation of animations.current.values()) animation.cancel(); animations.current.clear() }
    const changed = () => { if (reduce.matches) stop() }
    reduce.addEventListener('change', changed)
    return () => { reduce.removeEventListener('change', changed); stop(); pending.current = null }
  }, [])
  return capture
}
