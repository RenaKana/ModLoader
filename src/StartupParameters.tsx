import { useId, useRef, useState, type ReactNode } from 'react'
import { Check, Copy } from 'lucide-react'
import { copy, errorMessage } from './api'
import { Button, DiagnosticNotice, KeyValue, StateBadge } from './components'
import { uncheckedTooltip, type ParameterDiagnostic } from './startup-parameters'
import type { useStartupParameters } from './useStartupParameters'
import './startup-parameters.css'

export default function StartupParameters({ editor }: { editor: ReturnType<typeof useStartupParameters> }) {
  const textarea = useRef<HTMLTextAreaElement>(null)
  const composing = useRef(false), pasted = useRef(false)
  const [copied, setCopied] = useState(false), [copyError, setCopyError] = useState('')
  const announcement = useId()
  const unchecked = editor.result?.unchecked || []
  const unverified = new Set(unchecked.map(item => item.flag)).size
  const count = (editor.text.match(/(?:^|\s)--?[a-zA-Z][\w-]*/g) || []).length
  const fragments: ReactNode[] = []
  let cursor = 0
  function selectError(error: ParameterDiagnostic) {
    textarea.current?.focus()
    textarea.current?.setSelectionRange(error.value_start ?? error.start, error.value_end ?? error.end)
  }
  for (const [index, error] of [...editor.errors].sort((a, b) => a.start - b.start).entries()) {
    if (error.start < cursor) continue
    fragments.push(editor.text.slice(cursor, error.start))
    fragments.push(<mark key={`${error.start}-${index}`} className="startup-parameter-error" data-tooltip={error.message}
      tabIndex={0} role="button" aria-label={`${error.flag}：${error.message}`}
      onMouseDown={event => { event.preventDefault(); selectError(error) }}
      onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectError(error) } }}
    >{editor.text.slice(error.start, error.end) || ' '}</mark>)
    cursor = error.end
  }
  fragments.push(editor.text.slice(cursor), '\n')
  async function copyParameters() {
    setCopyError('')
    const value = await editor.flush()
    if (!value) return
    try { await copy(editor.currentText()); setCopied(true) } catch (error) { setCopyError(errorMessage(error)) }
  }
  const detail = editor.result
  return <section className="startup-parameters" aria-label="启动参数编辑器">
    <div className="startup-parameters-toolbar">
      <div>{detail?.compatibility && <StateBadge state={detail.compatibility.status} />}</div>
      <Button small variant="ghost" icon={copied ? Check : Copy} disabled={!editor.text || !!editor.errors.length} onClick={() => void copyParameters()}>{copied ? '已复制' : '复制参数'}</Button>
    </div>
    <div className="startup-code" aria-busy={editor.pending}>
      <div className="startup-code-lines">
        <pre className="startup-code-gutter" aria-hidden="true">{editor.text.split('\n').map((_, index) => String(index + 1).padStart(2, '0')).join('\n')}</pre>
        <div className="startup-code-input">
          <textarea ref={textarea} aria-label="启动参数" aria-invalid={!!editor.errors.length} aria-describedby={editor.errors.length ? announcement : undefined}
            value={editor.text} wrap="off" spellCheck={false} autoCapitalize="off" autoComplete="off"
            onPaste={() => { pasted.current = true }}
            onCompositionStart={() => { composing.current = true }}
            onCompositionEnd={event => { composing.current = false; editor.edit(event.currentTarget.value) }}
            onChange={event => { setCopied(false); setCopyError(''); editor.edit(event.target.value, pasted.current, composing.current); pasted.current = false }}
            onBlur={() => { if (!composing.current) void editor.flush() }} />
          <pre className="startup-code-paint">{fragments.map((fragment, index) => typeof fragment === 'string' ? <span aria-hidden="true" key={`text-${index}`}>{fragment}</span> : fragment)}</pre>
        </div>
      </div>
    </div>
    <div className="startup-parameters-meta">
      <span role="status">{editor.pending ? '校验中…' : ''}</span>
      <span className={`startup-parameter-count${editor.errors.length ? ' error-text' : ''}`} data-tooltip={unverified ? uncheckedTooltip(unchecked) : undefined} tabIndex={unverified ? 0 : undefined}>
        {editor.errors.length ? `${editor.errors.length} 处错误` : `${count} 项参数${unverified ? ` · ${unverified} 项未校验` : ''}`}
      </span>
    </div>
    <span className="startup-parameter-announcement" id={announcement} role="status">{editor.errors.map(item => `${item.flag}：${item.message}`).join('；')}</span>
    {editor.failure && <DiagnosticNotice summary="启动参数读取失败" detail={editor.failure} retry={editor.retry} />}
    {copyError && <p className="error-text" role="alert">{copyError}</p>}
    {detail?.preview_error && <p className="startup-preview-error">{detail.preview_error}</p>}
    {detail?.preparation && detail.preparation.status !== 'ready' && <p className="startup-preview-error">{detail.preparation.message}</p>}
    {detail?.cwd && <KeyValue label="工作目录" value={detail.cwd} mono />}
    {detail?.config != null && <details className="startup-effective-config"><summary>有效配置</summary><pre>{JSON.stringify(detail.config, null, 2)}</pre></details>}
  </section>
}
