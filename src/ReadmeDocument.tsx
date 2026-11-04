import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import { openExternal, errorMessage } from './api'
import documentStyles from './readme-document.css?raw'

const policy = "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src https: http: data:; font-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; media-src 'none'; base-uri 'none'; form-action 'none'"

function safeUrl(value: string, image: boolean, resolveUrl: (url: string, image: boolean) => string) {
  if (!image && value.startsWith('#')) return value
  if (image && /^data:image\/(?:png|jpe?g|gif|webp|avif|bmp);base64,[a-z\d+/=\s]+$/i.test(value)) return value
  try {
    const url = new URL(resolveUrl(value, image))
    return ['http:', 'https:'].includes(url.protocol) ? url.href : ''
  } catch { return '' }
}

function renderDocument(text: string, resolveUrl: (url: string, image: boolean) => string) {
  // README HTML belongs to a single document. Code fences remain code, while
  // ordinary HTML, IDs and language anchors retain their document semantics.
  const fragment = DOMPurify.sanitize(marked.parse(text, { async: false, gfm: true }), {
    USE_PROFILES: { html: true }, RETURN_DOM_FRAGMENT: true,
    FORBID_TAGS: ['script', 'iframe', 'frame', 'frameset', 'object', 'embed', 'base', 'meta', 'link', 'form', 'input', 'button', 'textarea', 'select'],
    FORBID_ATTR: ['srcdoc', 'srcset', 'action', 'formaction', 'target', 'formtarget', 'download', 'ping', 'autofocus', 'autoplay'],
  })
  for (const element of fragment.querySelectorAll('*')) {
    for (const attribute of [...element.attributes]) {
      if (['href', 'src', 'poster', 'background'].includes(attribute.name.toLowerCase())) {
        const isLink = element.localName === 'a' && attribute.name === 'href'
        const isImage = element.localName === 'img' && attribute.name === 'src'
        const value = isLink || isImage ? safeUrl(attribute.value, isImage, resolveUrl) : ''
        if (value) element.setAttribute(attribute.name, value)
        else element.removeAttribute(attribute.name)
      }
    }
    if (element.localName === 'img') element.setAttribute('referrerpolicy', 'no-referrer')
  }
  const ids = new Set(Array.from(fragment.querySelectorAll('[id]'), element => element.id))
  for (const heading of fragment.querySelectorAll('h1,h2,h3,h4,h5,h6')) {
    if (heading.id) continue
    const base = (heading.textContent || '').trim().toLowerCase().replace(/[^\p{L}\p{N}\p{M}_\s-]/gu, '').replace(/\s/g, '-')
    if (!base) continue
    let id = base, suffix = 0
    while (ids.has(id)) id = `${base}-${++suffix}`
    heading.id = id; ids.add(id)
  }
  const content = document.createElement('main')
  content.append(fragment)
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${policy}"><meta name="referrer" content="no-referrer"><style>${documentStyles}</style></head><body>${content.outerHTML}</body></html>`
}

export function ReadmeDocument({ text, resolveUrl, onDocumentLink }: { text: string; resolveUrl: (url: string, image: boolean) => string; onDocumentLink: (url: string) => boolean }) {
  const frame = useRef<HTMLIFrameElement>(null)
  const cleanup = useRef<(() => void) | undefined>(undefined)
  const openDocument = useRef(onDocumentLink)
  openDocument.current = onDocumentLink
  const [linkError, setLinkError] = useState('')
  const html = useMemo(() => renderDocument(text, resolveUrl), [text, resolveUrl])
  useLayoutEffect(() => {
    const doc = frame.current?.contentDocument
    if (!doc) return
    // Parse our sanitized document synchronously. Waiting for iframe load would
    // leave the reader at zero height until every remote image finishes loading.
    doc.open(); doc.write(html); doc.close()
    connectDocument()
    return () => cleanup.current?.()
  }, [html])

  function connectDocument() {
    cleanup.current?.()
    const element = frame.current!, doc = element.contentDocument
    const content = doc?.body.firstElementChild
    if (!doc || !content) return
    const resize = () => {
      const font = getComputedStyle(element)
      doc.body.style.fontSize = font.fontSize
      // Measure the content, not the viewport: collapsing <details> can shrink it.
      const height = Math.ceil(content.getBoundingClientRect().height)
      if (element.style.height !== `${height}px`) element.style.height = `${height}px`
    }
    const observer = new ResizeObserver(resize)
    observer.observe(content)
    const appearance = new MutationObserver(resize)
    appearance.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] })
    const click = (event: MouseEvent) => {
      const anchor = (event.target as Element | null)?.closest('a[href]')
      if (!anchor) return
      event.preventDefault()
      const href = anchor.getAttribute('href') || ''
      setLinkError('')
      if (href.startsWith('#')) {
        let id = href.slice(1)
        try { id = decodeURIComponent(id) } catch { /* Literal malformed fragments can still name an ID. */ }
        const target = id ? doc.getElementById(id) || Array.from(doc.getElementsByName(id)).find(node => node.localName === 'a') : content
        const scroll = element.closest('.modal-body')
        if (target && scroll) {
          const offset = element.getBoundingClientRect().top + target.getBoundingClientRect().top - scroll.getBoundingClientRect().top - 12
          scroll.scrollTop += offset
        }
      } else if (/^https?:\/\//i.test(href)) {
        try {
          if (!openDocument.current(href)) void openExternal(href).catch(error => setLinkError(errorMessage(error)))
        } catch (error) { setLinkError(errorMessage(error)) }
      }
    }
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    }
    // Source scripts are disabled. Only parent-owned handlers receive link clicks.
    doc.addEventListener('click', click)
    doc.addEventListener('keydown', key)
    resize()
    cleanup.current = () => { observer.disconnect(); appearance.disconnect(); doc.removeEventListener('click', click); doc.removeEventListener('keydown', key) }
  }

  return <>{linkError && <p className="error-notice" role="alert">{linkError}</p>}<iframe ref={frame} className="readme-document" title="模型 README" sandbox="allow-same-origin" referrerPolicy="no-referrer" /></>
}
