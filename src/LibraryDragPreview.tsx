import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import type { ReorderDrag } from './useLongPressReorder'
import './library-reorder.css'

export default function LibraryDragPreview({ drag, icon, name }: { drag: ReorderDrag; icon: ReactNode; name: string }) {
  const element = useRef<HTMLDivElement>(null)
  const [height, setHeight] = useState(0)
  useLayoutEffect(() => { setHeight(element.current!.getBoundingClientRect().height) }, [drag.id, name])
  const width = Math.min(drag.width, 360, innerWidth - 16)
  const left = Math.max(8, Math.min(innerWidth - width - 8, drag.x - Math.min(drag.offsetX, width)))
  const top = Math.max(8, Math.min(innerHeight - height - 8, drag.y - height - 12))
  return createPortal(<div className="library-drag-preview" aria-hidden="true" style={{ width, transform: `translate3d(${left}px, ${top}px, 0)` }}>
    <div className="library-drag-preview-card" ref={element}>{icon}<strong>{name}</strong></div>
  </div>, document.body)
}
