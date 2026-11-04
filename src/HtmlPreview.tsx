import { useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Code2, Download, ExternalLink, Eye, Maximize2, X } from 'lucide-react'
import { errorMessage, exportHtml } from './api'
import './html-preview.css'

type Props = { source: string; language: string; document: string }

function PreviewFrame({ document, language }: Pick<Props, 'document' | 'language'>) {
  return <iframe title={`${language.toUpperCase()} 静态预览`} sandbox="" referrerPolicy="no-referrer" srcDoc={document} />
}

function ExpandedPreview({ source, language, document, actions, feedback, onClose }: Props & { actions: React.ReactNode; feedback: React.ReactNode; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [view, setView] = useState<'preview' | 'source'>('preview')
  useLayoutEffect(() => {
    const element = dialog.current!
    element.showModal()
    return () => element.close()
  }, [])
  return createPortal(<dialog className="html-preview-dialog" ref={dialog} aria-label={`${language.toUpperCase()} 预览`} onClose={onClose}>
    <header className="html-preview-header">
      <strong>{language.toUpperCase()}</strong>
      <div className="html-preview-views" aria-label="显示方式">
        <button type="button" aria-pressed={view === 'preview'} onClick={() => setView('preview')}><Eye size={15} />预览</button>
        <button type="button" aria-pressed={view === 'source'} onClick={() => setView('source')}><Code2 size={15} />源码</button>
      </div>
      <div className="html-preview-actions">{actions}<button type="button" className="html-preview-close" aria-label="关闭预览" title="关闭预览（Esc）" onClick={() => dialog.current?.close()}><X size={18} /></button></div>
    </header>
    {feedback}
    <div className="html-preview-canvas" hidden={view !== 'preview'}><PreviewFrame document={document} language={language} /></div>
    {view === 'source' && <pre className="html-preview-source" tabIndex={0} aria-label="HTML 源码"><code>{source}</code></pre>}
  </dialog>, window.document.body)
}

export function HtmlPreview({ source, language, document, copyAction }: Props & { copyAction: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false)
  const [exporting, setExporting] = useState<'open' | 'save' | null>(null)
  const [feedback, setFeedback] = useState<{ text: string; error?: boolean } | null>(null)
  async function exportDocument(action: 'open' | 'save') {
    setExporting(action)
    setFeedback(null)
    try {
      const result = await exportHtml(source, action)
      if (!result.canceled) setFeedback({ text: action === 'open' ? '已在浏览器中打开' : '已保存 HTML' })
    } catch (error) { setFeedback({ text: errorMessage(error), error: true }) }
    finally { setExporting(null) }
  }
  const actions = <>
    {copyAction}
    <button type="button" disabled={!!exporting} title="在浏览器中打开原始 HTML，可运行脚本" onClick={() => void exportDocument('open')}><ExternalLink size={15} />{exporting === 'open' ? '打开中…' : '浏览器打开'}</button>
    <button type="button" disabled={!!exporting} title="保存原始 HTML" onClick={() => void exportDocument('save')}><Download size={15} />{exporting === 'save' ? '保存中…' : '保存'}</button>
  </>
  const notice = feedback && <div className={`html-preview-feedback${feedback.error ? ' is-error' : ''}`} role={feedback.error ? 'alert' : 'status'}>{feedback.text}</div>
  return <>
    <div className="message-code-label html-preview-header">
      <span>{language.toUpperCase()}<span className="html-preview-kind">静态预览</span></span>
      <div className="html-preview-actions"><button type="button" onClick={() => setExpanded(true)}><Maximize2 size={15} />展开预览</button>{actions}</div>
    </div>
    {notice}
    <div className="message-preview-frame"><PreviewFrame document={document} language={language} /></div>
    {expanded && <ExpandedPreview source={source} language={language} document={document} actions={actions} feedback={notice} onClose={() => setExpanded(false)} />}
  </>
}
