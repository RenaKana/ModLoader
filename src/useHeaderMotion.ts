import { useLayoutEffect, useRef, type RefObject } from 'react'

// Keep each icon at its current screen position across the compact/expanded layout change.
export function useHeaderMotion(header: RefObject<HTMLElement | null>, open: boolean) {
  const positions = useRef<{ icon: DOMRect; surface: DOMRect }[]>([])
  const animations = useRef<Animation[]>([])
  const capture = () => {
    positions.current = Array.from(header.current?.querySelectorAll('.navigation button') || [], button => ({
      icon: button.querySelector('.nav-icon')!.getBoundingClientRect(),
      surface: button.querySelector('.nav-surface')!.getBoundingClientRect(),
    }))
  }
  useLayoutEffect(() => {
    const node = header.current
    if (!node) return
    const before = positions.current
    positions.current = []
    // Capture happens before cancellation so reversing a running transition never teleports.
    animations.current.forEach(animation => animation.cancel())
    animations.current = []
    if (!before.length || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const style = getComputedStyle(node)
    const time = style.getPropertyValue('--header-motion-duration').trim()
    const duration = parseFloat(time) * (time.endsWith('ms') ? 1 : 1000)
    const easing = style.getPropertyValue('--header-move-ease').trim()
    node.querySelectorAll<HTMLButtonElement>('.navigation button').forEach((button, index) => {
      const icon = button.querySelector('.nav-icon')!
      const surface = button.querySelector<HTMLElement>('.nav-surface')!
      const after = icon.getBoundingClientRect(), afterSurface = surface.getBoundingClientRect(), previous = before[index]
      if (!previous) return
      const x = previous.icon.left - after.left, y = previous.icon.top - after.top
      if (Math.abs(x) < .5 && Math.abs(y) < .5) return
      animations.current.push(button.animate([
        { transform: `translate(${x}px, ${y}px)` }, { transform: 'translate(0, 0)' },
      ], { duration, easing }))
      // Only the background/selection outline changes size; icons and labels remain crisp.
      animations.current.push(surface.animate([
        { transform: `translate(${previous.surface.left - afterSurface.left - x}px, ${previous.surface.top - afterSurface.top - y}px) scale(${previous.surface.width / afterSurface.width}, ${previous.surface.height / afterSurface.height})` },
        { transform: 'translate(0, 0) scale(1, 1)' },
      ], { duration, easing }))
    })
  }, [header, open])
  useLayoutEffect(() => {
    const cancel = () => { animations.current.forEach(animation => animation.cancel()); animations.current = [] }
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
    reduced.addEventListener('change', cancel)
    window.addEventListener('resize', cancel)
    return () => { cancel(); reduced.removeEventListener('change', cancel); window.removeEventListener('resize', cancel) }
  }, [])
  return capture
}
