import { memo, useEffect, useState, type ReactNode } from 'react'
import type { DOMPurify } from 'dompurify'
import { HtmlPreview } from './HtmlPreview'
import './message-preview.css'

type Preview = { kind: 'image'; content: string; alt: string } | { kind: 'document'; content: string }
type PreviewResult = { source: string; language: string; preview?: Preview; error?: string }

const previewWaitMs = 30_000
const svgNamespace = 'http://www.w3.org/2000/svg'
const forbiddenTags = ['script', 'iframe', 'frame', 'frameset', 'object', 'embed', 'base', 'meta', 'link', 'form', 'foreignObject', 'foreignobject', 'animate', 'animateMotion', 'animateTransform', 'set']
const forbiddenAttributes = ['srcdoc', 'srcset', 'action', 'formaction', 'target', 'formtarget', 'download', 'ping', 'autofocus', 'autoplay']
const previewPolicy = "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src https: http: data:; font-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; media-src 'none'; base-uri 'none'; form-action 'none'"
const frameHead = `<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${previewPolicy}"><meta name="referrer" content="no-referrer"><style>html{color-scheme:light;background:#fff}body{margin:16px;color:#27344a;font:14px/1.6 'Microsoft YaHei UI','Microsoft YaHei',sans-serif;overflow-wrap:anywhere}img,svg{max-width:100%}pre{overflow:auto}a{color:inherit}</style>`
let nextDiagramId = 0
let purifierPromise: Promise<DOMPurify> | undefined
let mermaidPromise: Promise<typeof import('mermaid')['default']> | undefined

function loadPurifier() {
  return purifierPromise ??= import('dompurify').then(module => module.default)
}

function loadMermaid() {
  return mermaidPromise ??= import('mermaid').then(({ default: mermaid }) => {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      suppressErrorRendering: true,
      htmlLabels: false,
      theme: 'neutral',
      fontFamily: "'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif",
      // The library's sample-size limits are not product limits. Keep complete diagrams valid.
      maxTextSize: Number.MAX_SAFE_INTEGER,
      maxEdges: Number.MAX_SAFE_INTEGER,
      secure: ['secure', 'securityLevel', 'startOnLoad', 'suppressErrorRendering', 'maxTextSize', 'maxEdges', 'htmlLabels', 'flowchart', 'themeCSS', 'fontFamily', 'dompurifyConfig'],
    })
    return mermaid
  })
}

function renderGraphviz(source: string, signal: AbortSignal) {
  return new Promise<string>((resolve, reject) => {
    signal.throwIfAborted()
    const worker = new Worker(new URL('./graphviz.worker.ts', import.meta.url), { type: 'module' })
    const cleanup = () => { worker.terminate(); signal.removeEventListener('abort', abort) }
    const abort = () => { cleanup(); reject(new DOMException('Preview cancelled', 'AbortError')) }
    worker.onmessage = ({ data }: MessageEvent<{ svg?: string; failed?: boolean }>) => {
      cleanup()
      if (typeof data.svg === 'string') resolve(data.svg)
      else reject(new Error('Graphviz rendering failed'))
    }
    worker.onerror = event => { event.preventDefault(); cleanup(); reject(new Error('Graphviz worker failed')) }
    worker.onmessageerror = () => { cleanup(); reject(new Error('Graphviz worker result failed')) }
    signal.addEventListener('abort', abort, { once: true })
    try { worker.postMessage(source) } catch (error) { cleanup(); reject(error) }
  })
}

function imageUrl(value: string) {
  const trimmed = value.trim()
  // SVG data URLs need their own sanitation; only inert raster data is accepted here.
  if (/^data:image\/(?:png|jpeg|gif|webp|avif|bmp|x-icon);base64,[a-z\d+/=\s]+$/i.test(trimmed)) return trimmed
  try {
    const url = new URL(trimmed)
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : ''
  } catch { return '' }
}

function restrictResources(root: Element | DocumentFragment) {
  for (const element of root.querySelectorAll('*')) {
    const tag = element.localName.toLowerCase()
    for (const attribute of [...element.attributes]) {
      const name = attribute.name.toLowerCase()
      if (name === 'href' || name === 'xlink:href') {
        const value = attribute.value.trim()
        const allowed = tag === 'a' ? '' : tag === 'image' ? imageUrl(value) : value.startsWith('#') ? value : ''
        if (allowed) element.setAttribute(name, allowed)
        else element.removeAttribute(attribute.name)
      } else if (name === 'src' || name === 'poster' || name === 'background') {
        const allowed = tag === 'img' && name === 'src' ? imageUrl(attribute.value) : ''
        if (allowed) element.setAttribute(name, allowed)
        else element.removeAttribute(attribute.name)
      }
    }
  }
}

function sanitizeSvg(source: string, purifier: DOMPurify) {
  const parsed = new DOMParser().parseFromString(source, 'image/svg+xml')
  const namespace = parsed.documentElement.namespaceURI
  if (parsed.querySelector('parsererror') || parsed.documentElement.localName !== 'svg' || (namespace !== null && namespace !== svgNamespace)) throw new Error('Invalid SVG')
  const fragment = purifier.sanitize(source, {
    USE_PROFILES: { svg: true, svgFilters: true },
    FORBID_TAGS: forbiddenTags,
    FORBID_ATTR: forbiddenAttributes,
    RETURN_DOM_FRAGMENT: true,
  })
  restrictResources(fragment)
  const svg = fragment.querySelector('svg')
  if (!svg) throw new Error('Missing SVG')
  svg.setAttribute('xmlns', svgNamespace)
  return svg
}

function diagramPreview(svgSource: string, label: string, purifier: DOMPurify): Preview {
  const svg = sanitizeSvg(svgSource, purifier)
  // Mermaid uses width="100%". Explicit intrinsic dimensions keep original-size viewing useful.
  const viewBox = svg.getAttribute('viewBox')?.trim().split(/[\s,]+/).map(Number)
  if (viewBox?.length === 4 && viewBox.every(Number.isFinite) && viewBox[2] > 0 && viewBox[3] > 0) {
    svg.setAttribute('width', String(viewBox[2]))
    svg.setAttribute('height', String(viewBox[3]))
    svg.style.removeProperty('max-width')
    svg.style.removeProperty('width')
    svg.style.removeProperty('height')
  }
  const title = svg.querySelector('title')?.textContent?.trim()
  return { kind: 'image', content: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(new XMLSerializer().serializeToString(svg))}`, alt: title ? `${label}：${title}` : label }
}

function frameDocument(body: string) {
  return `<!doctype html><html><head>${frameHead}</head><body>${body}</body></html>`
}

function staticPreview(source: string, language: string, purifier: DOMPurify): Preview {
  if (language === 'svg') return { kind: 'document', content: frameDocument(new XMLSerializer().serializeToString(sanitizeSvg(source, purifier))) }
  const safe = purifier.sanitize(source, {
    USE_PROFILES: { html: true, svg: true, svgFilters: true },
    WHOLE_DOCUMENT: true,
    FORBID_TAGS: forbiddenTags,
    FORBID_ATTR: forbiddenAttributes,
    RETURN_DOM: true,
  })
  if (!(safe instanceof Element)) throw new Error('Invalid preview document')
  restrictResources(safe)
  safe.querySelector('head')?.insertAdjacentHTML('afterbegin', frameHead)
  return { kind: 'document', content: `<!doctype html>${safe.outerHTML}` }
}

async function createPreview(source: string, language: string, signal: AbortSignal): Promise<Preview> {
  const purifier = await loadPurifier()
  signal.throwIfAborted()
  if (language === 'mermaid') {
    const mermaid = await loadMermaid()
    signal.throwIfAborted()
    const host = document.createElement('div')
    host.className = 'message-preview-measure'
    host.setAttribute('aria-hidden', 'true')
    document.body.append(host)
    const removeHost = () => host.remove()
    signal.addEventListener('abort', removeHost, { once: true })
    try {
      const { svg } = await mermaid.render(`message-diagram-${++nextDiagramId}`, source, host)
      signal.throwIfAborted()
      return diagramPreview(svg, 'Mermaid 图表', purifier)
    } finally {
      signal.removeEventListener('abort', removeHost)
      host.remove()
    }
  }
  if (language === 'dot' || language === 'graphviz') {
    const svg = await renderGraphviz(source, signal)
    signal.throwIfAborted()
    return diagramPreview(svg, 'Graphviz 图表', purifier)
  }
  return staticPreview(source, language, purifier)
}

export const MessagePreview = memo(function MessagePreview({ source, language, pending, copyAction }: { source: string; language: string; pending: boolean; copyAction: ReactNode }) {
  const [result, setResult] = useState<PreviewResult | null>(null)
  const [naturalSize, setNaturalSize] = useState(false)
  useEffect(() => {
    if (pending) return
    const controller = new AbortController()
    setNaturalSize(false)
    const timer = window.setTimeout(() => {
      controller.abort()
      setResult({ source, language, error: '预览耗时过长，请查看源码。' })
    }, previewWaitMs)
    void createPreview(source, language, controller.signal).then(preview => {
      if (!controller.signal.aborted) setResult({ source, language, preview })
    }).catch(() => {
      if (!controller.signal.aborted) setResult({ source, language, error: '暂时无法预览，请检查语法或查看源码。' })
    }).finally(() => window.clearTimeout(timer))
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [source, language, pending])

  const current = result?.source === source && result.language === language ? result : null
  const preview = current?.preview
  const label = <div className="message-code-label"><span>{language}</span>{copyAction}</div>
  if (pending) return <>{label}<div className="message-preview-status" role="status">接收中…</div></>
  if (current?.error) return <>{label}<div className="message-preview-status message-preview-error" role="status">{current.error}</div></>
  if (!preview) return <>{label}<div className="message-preview-status" role="status">正在准备预览…</div></>
  if (preview.kind === 'document') return <HtmlPreview source={source} language={language} document={preview.content} copyAction={copyAction} />
  return <>{label}<div className="message-diagram-preview">
    <div className="message-preview-tools"><button type="button" aria-pressed={naturalSize} onClick={() => setNaturalSize(value => !value)}>{naturalSize ? '适应宽度' : '原始尺寸'}</button></div>
    <div className={`message-diagram-scroll${naturalSize ? ' natural-size' : ''}`} tabIndex={0} role="region" aria-label={`${preview.alt}，可滚动查看`}><img src={preview.content} alt={preview.alt} onError={() => setResult({ source, language, error: '图表图像无法显示，请查看源码。' })} /></div>
  </div></>
})
