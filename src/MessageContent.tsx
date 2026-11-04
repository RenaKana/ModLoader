import { createContext, memo, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from 'react'
import Markdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Element, Root } from 'hast'
import { visit } from 'unist-util-visit'
import { Check, Copy, X } from 'lucide-react'
import { copy, errorMessage, openExternal } from './api'
import { remarkMessageSyntax, remarkTexDelimiters } from './markdown-syntax'
import { MessagePreview } from './MessagePreview'
import { useImagePreview } from './ImageViewer'
import { ChatLocalResource } from './ChatLocalResources'
import { localResourceReference, type LocalResource } from './chat-local-resources'
import 'katex/dist/katex.min.css'
import 'highlight.js/styles/github.css'
import './message-content.css'

function webUrl(url: string) {
  try {
    const parsed = new URL(url)
    return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : ''
  } catch { return '' }
}

function imageUrl(url: string) {
  if (/^data:image\/(?:png|jpe?g|gif|webp|avif|bmp);base64,[a-z\d+/=\s]+$/i.test(url)) return url
  return webUrl(url)
}

const MessageImage = memo(function MessageImage({ src, alt }: { src?: string; alt?: string }) {
  const preview = useImagePreview(alt || '消息中的图片')
  const [failed, setFailed] = useState(false)
  useEffect(() => { setFailed(false) }, [src])
  return <span className="message-image">
    {!src ? <span className="message-image-error">{alt || '图片'} · 图片地址不受支持</span>
      : failed ? <span className="message-image-error">{alt || '图片'} · 图片加载失败</span>
      : <img src={src} alt={alt || '消息中的图片'} {...preview} loading="lazy" referrerPolicy="no-referrer" onError={() => setFailed(true)} />}
    {src && webUrl(src) && <a href={src} onClick={event => { event.preventDefault(); void openExternal(src) }}>打开原图</a>}
  </span>
})

let highlighter: ReturnType<typeof importHighlighter> | undefined
function importHighlighter() { return import('highlight.js').then(module => module.default) }
const HighlightedCode = memo(function HighlightedCode({ source, language, pending }: { source: string; language: string; pending: boolean }) {
  const [highlighted, setHighlighted] = useState<{ source: string; language: string; html: string }>()
  useEffect(() => {
    if (pending || !language) return
    let cancelled = false
    highlighter ||= importHighlighter()
    void highlighter.then(engine => {
      if (cancelled || !engine.getLanguage(language)) return
      const html = engine.highlight(source, { language, ignoreIllegals: true }).value
      setHighlighted({ source, language, html })
    }).catch(() => { /* The original code remains visible when highlighting fails. */ })
    return () => { cancelled = true }
  }, [source, language, pending])
  const ready = highlighted?.source === source && highlighted.language === language
  return <pre>{ready
    ? <code className={`hljs language-${language}`} dangerouslySetInnerHTML={{ __html: highlighted.html }} />
    : <code className={language ? `language-${language}` : undefined}>{source}</code>}</pre>
})

const previewLanguages = new Set(['mermaid', 'dot', 'graphviz', 'html', 'htm', 'svg'])
const CodeBlock = memo(function CodeBlock({ source, language, pending }: { source: string; language: string; pending: boolean }) {
  const preview = previewLanguages.has(language)
  const [copying, setCopying] = useState(false)
  const [feedback, setFeedback] = useState<{ source: string; error?: string } | null>(null)
  const current = feedback?.source === source ? feedback : null
  const Icon = current ? current.error ? X : Check : Copy
  async function copySource() {
    setCopying(true)
    setFeedback(null)
    try { await copy(source); setFeedback({ source }) }
    catch (error) { setFeedback({ source, error: errorMessage(error) }) }
    finally { setCopying(false) }
  }
  const copyAction = <button type="button" className={`message-code-copy${current?.error ? ' is-error' : ''}`} aria-label={preview ? '复制源码' : '复制内容'} title={current?.error ? `复制失败：${current.error}` : undefined} disabled={copying} onClick={() => void copySource()}>
    <Icon size={13} aria-hidden="true" /><span role={current ? current.error ? 'alert' : 'status' : undefined}>{copying ? '复制中…' : current ? current.error ? '复制失败' : '已复制' : '复制'}</span>
  </button>
  return <div className="message-code-block" data-language={language}>
    {preview ? <>
      <MessagePreview source={source} language={language} pending={pending} copyAction={copyAction} />
      <details className="message-code-source" open={pending || undefined}><summary>源码</summary><HighlightedCode source={source} language={language} pending={pending} /></details>
    </> : <><div className="message-code-label"><span>{language || 'text'}</span>{copyAction}</div><HighlightedCode source={source} language={language} pending={pending} /></>}
  </div>
})

function renderCode(node?: Element) {
  const code = node?.children.find((child): child is Element => child.type === 'element' && child.tagName === 'code')
  if (!code) return null
  const source = code.children.map(child => child.type === 'text' ? child.value : '').join('')
  const classes = code.properties.className
  const language = (Array.isArray(classes) ? classes : []).map(String).find(name => name.startsWith('language-'))?.slice(9).toLowerCase() || ''
  return <CodeBlock source={source} language={language} pending={Boolean(code.properties['data-pending'] ?? code.properties.dataPending)} />
}

const emptyResources: readonly LocalResource[] = []
const LocalResources = createContext<{ resources: readonly LocalResource[]; directory: string }>({ resources: emptyResources, directory: '' })

function rehypeLocalResources({ resources }: { resources: readonly LocalResource[] }) {
  return (tree: Root) => {
    if (!resources.length) return
    const used = new Set<string>()
    visit(tree, 'element', node => {
      const url = node.tagName === 'img' ? node.properties.src : node.tagName === 'a' ? node.properties.href : undefined
      const resource = typeof url === 'string' ? localResourceReference(url, resources) : undefined
      if (!resource) return
      used.add(resource.id)
      node.tagName = 'div'
      node.properties = { 'data-local-resource': resource.id }
      node.children = []
    })
    const containsResource = (node: Element): boolean => Boolean(node.properties['data-local-resource']) || node.children.some(child => child.type === 'element' && containsResource(child))
    visit(tree, 'element', node => {
      // Resource cards contain buttons and image viewers, not paragraph text.
      if (node.tagName === 'p' && containsResource(node)) {
        node.tagName = 'div'; node.properties.className = ['message-resource-paragraph']
      }
    })
    for (const resource of resources) if (!used.has(resource.id)) {
      used.add(resource.id)
      tree.children.push({ type: 'element', tagName: 'div', properties: { 'data-local-resource': resource.id }, children: [] })
    }
  }
}

function MessageBlock({ node, children, ...props }: ComponentPropsWithoutRef<'div'> & { node?: Element }) {
  const context = useContext(LocalResources)
  const resource = context.resources.find(item => item.id === node?.properties['data-local-resource'])
  return resource ? <ChatLocalResource resource={resource} directory={context.directory} /> : <div {...props}>{children}</div>
}

function MessageLink({ href, children }: ComponentPropsWithoutRef<'a'>) {
  if (!href) return <span>{children}</span>
  return <a href={href} onClick={event => { event.preventDefault(); void openExternal(href) }}>{children}</a>
}

// Stable component identities preserve loaded images and previews between stream chunks.
const messageComponents: Components = {
  div: MessageBlock,
  a: MessageLink,
  img: ({ src, alt }) => <MessageImage src={typeof src === 'string' ? src : undefined} alt={alt} />,
  table: ({ children }) => <div className="message-table-scroll" tabIndex={0}><table>{children}</table></div>,
  pre: ({ node }) => renderCode(node),
}

// Raw HTML is converted to isolated previews by the remark plugin. It is never
// interpreted as markup in the parent document, including during streaming.
export const MarkdownContent = memo(function MarkdownContent({ text, streaming = false, resources = emptyResources, directory = '' }: { text: string; streaming?: boolean; resources?: readonly LocalResource[]; directory?: string }) {
  const plugins = useMemo(() => [remarkGfm, remarkMath, remarkTexDelimiters, [remarkMessageSyntax, { streaming }]] as import('unified').PluggableList, [streaming])
  const rehypePlugins = useMemo(() => [[rehypeKatex, { trust: false, strict: false }], [rehypeLocalResources, { resources }]] as import('unified').PluggableList, [resources])
  const local = useMemo(() => ({ resources, directory }), [resources, directory])
  return <LocalResources.Provider value={local}><div className="message-markdown"><Markdown remarkPlugins={plugins} rehypePlugins={rehypePlugins} skipHtml urlTransform={(url, key, node) => node.tagName === 'img' && key === 'src' ? imageUrl(url) : webUrl(url)} components={messageComponents}>{text}</Markdown></div></LocalResources.Provider>
})

export function ReasoningContent({ text, thinking, active = thinking, children, className = '', summary }: { text: string; thinking: boolean; active?: boolean; children?: ReactNode; className?: string; summary?: ReactNode }) {
  const [open, setOpen] = useState(active)
  const [contentHeight, setContentHeight] = useState(0)
  const viewport = useRef<HTMLDivElement>(null)
  const follow = useRef(true)
  const lastScrollTop = useRef(0)
  function scrollToEnd() {
    const element = viewport.current
    if (!element || !open || !active || !follow.current) return
    element.scrollTop = element.scrollHeight
    lastScrollTop.current = element.scrollTop
  }
  useEffect(() => { setOpen(active) }, [active])
  useLayoutEffect(scrollToEnd, [text, open, active])
  useLayoutEffect(() => {
    const element = viewport.current
    if (!element || (children === undefined && (!open || !active))) return
    const resized = () => {
      if (children !== undefined) setContentHeight(element.getBoundingClientRect().height)
      scrollToEnd()
    }
    resized()
    // Markdown images and previews can resize after the text has already rendered.
    const observer = new ResizeObserver(resized)
    observer.observe(element)
    if (element.firstElementChild) observer.observe(element.firstElementChild)
    return () => observer.disconnect()
  }, [open, active, children !== undefined])
  const content = <div className="reasoning-scroll" ref={viewport} tabIndex={0} role="region" aria-label="思考内容" onPointerDown={children === undefined ? undefined : () => { follow.current = false }} onWheel={event => {
      if (active && event.deltaY < 0) follow.current = false
    }} onKeyDown={event => {
      if (!active) return
      if (children !== undefined && event.target !== event.currentTarget && (event.key === 'Enter' || event.key === ' ')) follow.current = false
      if (['ArrowUp', 'PageUp', 'Home'].includes(event.key)) follow.current = false
      if (event.target !== event.currentTarget || (event.key !== 'Home' && event.key !== 'End')) return
      event.preventDefault()
      follow.current = event.key === 'End'
      event.currentTarget.scrollTop = follow.current ? event.currentTarget.scrollHeight : 0
    }} onScroll={event => {
      if (!open || !active) return
      const element = event.currentTarget
      const atEnd = element.scrollHeight - element.scrollTop - element.clientHeight < 24
      // A queued programmatic scroll or layout anchoring is not a request to pause.
      if (atEnd) follow.current = true
      else if (element.scrollTop < lastScrollTop.current) follow.current = false
      lastScrollTop.current = element.scrollTop
    }}>{children === undefined ? <MarkdownContent text={text} streaming={active} /> : children}</div>
  return <details className={`message-reasoning${className ? ` ${className}` : ''}`} open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>{summary ?? (thinking ? '思考中' : '思考过程')}</summary>
    {children === undefined ? content : <div className="reasoning-collapse" style={{ height: open ? contentHeight : 0 }} aria-hidden={!open} inert={!open}>{content}</div>}
  </details>
}
