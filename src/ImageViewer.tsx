import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type HTMLAttributes } from 'react'
import { createPortal } from 'react-dom'
import { ChevronLeft, ChevronRight, Download, Image as ImageIcon, Maximize2, RotateCcw, RotateCw, ZoomIn, ZoomOut } from 'lucide-react'
import { errorMessage, saveImage } from './api'
import './image-viewer.css'

interface ImageItem { src: string; name: string; width: number; height: number }
interface View { mode: 'fit' | 'actual' | 'custom'; scale: number; rotation: number; x: number; y: number }
const initialView: View = { mode: 'fit', scale: 1, rotation: 0, x: 0, y: 0 }
const dragThreshold = 4
const GalleryContext = createContext(false)

export function useImagePreview(name: string) {
  return useContext(GalleryContext) ? {
    'data-image-preview': '', role: 'button', tabIndex: 0, 'aria-label': `查看图片：${name}`,
  } : {}
}

/** Snapshot this message's rendered images only when the user opens one. */
export function ImageGallery({ children, ...props }: HTMLAttributes<HTMLDivElement>) {
  const group = useRef<HTMLDivElement>(null)
  const [opened, setOpened] = useState<{ images: ImageItem[]; index: number; trigger: HTMLImageElement } | null>(null)
  function open(target: EventTarget | null) {
    if (!(target instanceof HTMLImageElement) || !target.hasAttribute('data-image-preview')) return false
    if (target.closest('[data-image-gallery]') !== group.current) return false
    const images = Array.from(group.current!.querySelectorAll<HTMLImageElement>('img[data-image-preview]')).filter(image => image.closest('[data-image-gallery]') === group.current)
    const index = images.indexOf(target)
    if (index < 0) return false
    target.focus({ preventScroll: true })
    setOpened({ images: images.map(image => ({ src: image.currentSrc || image.src, name: image.alt || '图片', width: image.naturalWidth, height: image.naturalHeight })), index, trigger: target })
    return true
  }
  return <GalleryContext.Provider value={true}><div {...props} data-image-gallery="" ref={group} onClickCapture={event => {
    if (open(event.target)) { event.preventDefault(); event.stopPropagation() }
  }} onKeyDownCapture={event => {
    if ((event.key === 'Enter' || event.key === ' ') && open(event.target)) { event.preventDefault(); event.stopPropagation() }
  }}>{children}</div>{opened && <ImageViewer images={opened.images} initialIndex={opened.index} onClose={() => {
    setOpened(null)
    if (opened.trigger.isConnected) opened.trigger.focus({ preventScroll: true })
  }} />}</GalleryContext.Provider>
}

export function GalleryImage(props: React.ImgHTMLAttributes<HTMLImageElement>) {
  const preview = useImagePreview(props.alt || '图片')
  return <img {...props} {...preview} />
}

function ImageViewer({ images, initialIndex, onClose }: { images: ImageItem[]; initialIndex: number; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const stage = useRef<HTMLDivElement>(null)
  const image = useRef<HTMLImageElement>(null)
  const interaction = useRef<{ pointer: number; origin: 'image' | 'background'; x: number; y: number; startX: number; startY: number; panning: boolean; moved: boolean } | null>(null)
  const [index, setIndex] = useState(initialIndex)
  const item = images[index]
  const [size, setSize] = useState({ width: item.width, height: item.height })
  const [viewport, setViewport] = useState({ width: 0, height: 0 })
  const [view, setView] = useState<View>(initialView)
  const [status, setStatus] = useState<'loading' | 'loaded' | 'failed'>('loading')
  const [dragging, setDragging] = useState(false)
  const [saving, setSaving] = useState(false)
  const [feedback, setFeedback] = useState<{ text: string; error?: boolean } | null>(null)
  const activeItem = useRef(item)
  activeItem.current = item

  useLayoutEffect(() => {
    const element = dialog.current!
    element.showModal()
    element.focus({ preventScroll: true })
    return () => element.close()
  }, [])
  useLayoutEffect(() => {
    const element = stage.current!
    const observer = new ResizeObserver(() => setViewport({ width: element.clientWidth, height: element.clientHeight }))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  useLayoutEffect(() => {
    setSize({ width: item.width, height: item.height })
    setView(initialView)
    setStatus('loading')
    setFeedback(null)
    interaction.current = null
    setDragging(false)
  }, [item])

  function geometry(value: View) {
    const quarterTurn = value.rotation % 180 !== 0
    const width = quarterTurn ? size.height : size.width
    const height = quarterTurn ? size.width : size.height
    const scale = value.mode === 'fit' && width && height
      ? Math.min(1, Math.max(1, viewport.width - 32) / width, Math.max(1, viewport.height - 32) / height)
      : value.mode === 'actual' ? 1 : value.scale
    return { scale, maxX: Math.max(0, (width * scale - viewport.width) / 2), maxY: Math.max(0, (height * scale - viewport.height) / 2) }
  }
  function constrain(value: View): View {
    const { maxX, maxY } = geometry(value)
    return { ...value, x: Math.max(-maxX, Math.min(maxX, value.x)), y: Math.max(-maxY, Math.min(maxY, value.y)) }
  }
  const { scale, maxX, maxY } = geometry(view)
  const canDrag = status === 'loaded' && (maxX > 0 || maxY > 0)
  useLayoutEffect(() => { setView(previous => constrain(previous)) }, [size, viewport])

  function zoom(factor: number, point = { x: 0, y: 0 }) {
    if (status !== 'loaded') return
    setView(previous => {
      const currentScale = geometry(previous).scale
      const next = currentScale * factor
      if (!(next > 0) || !Number.isFinite(next * Math.max(size.width, size.height))) return previous
      return constrain({ ...previous, mode: 'custom', scale: next, x: point.x - (point.x - previous.x) * factor, y: point.y - (point.y - previous.y) * factor })
    })
  }
  useEffect(() => {
    const element = stage.current!
    function wheel(event: WheelEvent) {
      event.preventDefault()
      event.stopPropagation()
      const bounds = element.getBoundingClientRect()
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? element.clientHeight : 1)
      zoom(Math.exp(-Math.max(-500, Math.min(500, delta)) * .002), { x: event.clientX - bounds.left - bounds.width / 2, y: event.clientY - bounds.top - bounds.height / 2 })
    }
    element.addEventListener('wheel', wheel, { passive: false })
    return () => element.removeEventListener('wheel', wheel)
  }, [status, size, viewport])

  function navigate(direction: number) {
    setIndex(previous => Math.max(0, Math.min(images.length - 1, previous + direction)))
  }
  function rotate() { setView(previous => constrain({ ...previous, rotation: (previous.rotation + 90) % 360, x: 0, y: 0 })) }
  function fit() { setView(previous => ({ ...previous, mode: 'fit', x: 0, y: 0 })) }
  function actual() { setView(previous => ({ ...previous, mode: 'actual', x: 0, y: 0 })) }
  async function save() {
    if (saving || status !== 'loaded') return
    const source = item
    setSaving(true)
    setFeedback(null)
    try {
      const result = await saveImage(source.src, source.name)
      if (!result.canceled && activeItem.current === source) setFeedback({ text: '图片已保存' })
    } catch (error) {
      if (activeItem.current === source) setFeedback({ text: errorMessage(error), error: true })
    } finally { setSaving(false) }
  }
  function hit(target: EventTarget | null): 'image' | 'background' | undefined {
    if (target === image.current) return 'image'
    if (target === dialog.current || target === stage.current || target instanceof HTMLElement && (target.classList.contains('image-viewer-top') || target.classList.contains('image-viewer-state'))) return 'background'
  }
  function endInteraction(event: React.PointerEvent<HTMLDialogElement>, canceled = false) {
    const active = interaction.current
    if (!active || active.pointer !== event.pointerId) return
    event.stopPropagation()
    interaction.current = null
    setDragging(false)
    const isClick = !active.moved && Math.hypot(event.clientX - active.x, event.clientY - active.y) <= dragThreshold
    const close = !canceled && active.origin === 'background' && isClick && hit(document.elementFromPoint(event.clientX, event.clientY)) === 'background'
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    if (close) dialog.current?.close()
  }
  const toolsDisabled = status !== 'loaded'
  const zoomLabel = scale < .01 ? '<1%' : `${Math.round(scale * 100)}%`
  useEffect(() => {
    function keyDown(event: KeyboardEvent) {
      if (event.ctrlKey || event.metaKey || event.altKey) return
      const action = ({ ArrowLeft: () => navigate(-1), ArrowRight: () => navigate(1), '+': () => zoom(1.25), '=': () => zoom(1.25), '-': () => zoom(.8), f: fit, F: fit, '1': actual, r: rotate, R: rotate, '0': () => setView(initialView) } as Record<string, () => void>)[event.key]
      // Disabling a focused toolbar button can move focus to body. Keep the
      // modal's shortcuts working there as well as inside its own controls.
      if (action) { event.preventDefault(); event.stopPropagation(); action() }
    }
    document.addEventListener('keydown', keyDown, true)
    return () => document.removeEventListener('keydown', keyDown, true)
  })
  return createPortal(<dialog ref={dialog} tabIndex={-1} className={`image-viewer${dragging ? ' is-dragging' : ''}`} aria-label="图片查看器" onClose={onClose} onKeyDown={event => event.stopPropagation()} onPointerDown={event => {
    if (!event.isPrimary || event.button !== 0 || interaction.current) return
    const origin = hit(event.target)
    if (!origin) return
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.setPointerCapture(event.pointerId)
    interaction.current = { pointer: event.pointerId, origin, x: event.clientX, y: event.clientY, startX: view.x, startY: view.y, panning: origin === 'image' && canDrag, moved: false }
  }} onPointerMove={event => {
    const active = interaction.current
    if (!active || active.pointer !== event.pointerId) return
    event.stopPropagation()
    if (Math.hypot(event.clientX - active.x, event.clientY - active.y) > dragThreshold) active.moved = true
    if (active.panning && active.moved) {
      setDragging(true)
      setView(previous => constrain({ ...previous, x: active.startX + event.clientX - active.x, y: active.startY + event.clientY - active.y }))
    }
  }} onPointerUp={event => endInteraction(event)} onPointerCancel={event => endInteraction(event, true)} onLostPointerCapture={event => endInteraction(event, true)}>
    <header className="image-viewer-top">
      <div className="image-viewer-heading"><ImageIcon size={15} aria-hidden="true" /><strong title={item.name}>{item.name}</strong>
        {status === 'loaded' && <span className="image-viewer-dimensions">{size.width} × {size.height}{view.rotation !== 0 && ` · ${view.rotation}°`}</span>}
      </div>
    </header>
    <div ref={stage} className={`image-viewer-stage${canDrag ? ' is-draggable' : ''}`}>
      <img ref={image} key={index} className="image-viewer-image" src={item.src} alt={item.name} draggable={false} referrerPolicy="no-referrer" style={{ width: size.width || undefined, height: size.height || undefined, visibility: status === 'loaded' ? 'visible' : 'hidden', transform: `translate(-50%, -50%) translate(${view.x}px, ${view.y}px) rotate(${view.rotation}deg) scale(${scale})` }} onLoad={event => {
        setSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })
        setStatus('loaded')
      }} onError={() => setStatus('failed')} />
      {status !== 'loaded' && <div className="image-viewer-state" role={status === 'failed' ? 'alert' : 'status'}>{status === 'failed' ? '图片加载失败' : '正在加载图片…'}</div>}
      {(saving || feedback) && <div className={`image-viewer-feedback${feedback?.error ? ' is-error' : ''}`} role={feedback?.error ? 'alert' : 'status'}>{saving ? '正在保存图片…' : feedback?.text}</div>}
    </div>
    <div className="image-viewer-toolbar" role="toolbar" aria-label="图片工具">
      {images.length > 1 && <><div className="image-viewer-navigation">
        <button type="button" title="上一张（←）" aria-label="上一张图片" disabled={index === 0} onClick={() => navigate(-1)}><ChevronLeft size={18} /></button>
        <span aria-live="polite" aria-label="图片位置">{index + 1} / {images.length}</span>
        <button type="button" title="下一张（→）" aria-label="下一张图片" disabled={index === images.length - 1} onClick={() => navigate(1)}><ChevronRight size={18} /></button>
      </div><span className="image-viewer-divider" /></>}
      <div className="image-viewer-tools">
        <button type="button" title="缩小（−）" aria-label="缩小图片" disabled={toolsDisabled} onClick={() => zoom(.8)}><ZoomOut size={18} /></button>
        <output className="image-viewer-zoom" aria-label="图片缩放比例">{status === 'loaded' ? zoomLabel : '—'}</output>
        <button type="button" title="放大（+）" aria-label="放大图片" disabled={toolsDisabled} onClick={() => zoom(1.25)}><ZoomIn size={18} /></button>
        <span className="image-viewer-divider" />
        <button type="button" title="适应窗口（F）" aria-label="适应窗口" aria-pressed={view.mode === 'fit'} disabled={toolsDisabled} onClick={fit}><Maximize2 size={17} /></button>
        <button type="button" className="image-viewer-actual" title="原始比例（1）" aria-label="原始比例" aria-pressed={view.mode === 'actual'} disabled={toolsDisabled} onClick={actual}>1:1</button>
        <button type="button" title="顺时针旋转（R）" aria-label="旋转图片" disabled={toolsDisabled} onClick={rotate}><RotateCw size={18} /></button>
        <button type="button" title="复位（0）" aria-label="复位图片" disabled={toolsDisabled} onClick={() => setView(initialView)}><RotateCcw size={17} /></button>
        <span className="image-viewer-divider" />
        <button type="button" title="保存原图" aria-label="保存图片" disabled={toolsDisabled || saving} onClick={() => void save()}><Download size={18} /></button>
      </div>
    </div>
  </dialog>, document.body)
}
