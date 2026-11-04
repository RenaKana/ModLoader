import { useEffect, useRef, useState } from 'react'
import { requestProgress, errorMessage } from './api'
import OperationStatus, { type OperationProgress } from './OperationStatus'
import { DiagnosticNotice, Field, Select } from './components'
import type { LoadProfile, PreparationStatus } from './types'
import './profile-preparation.css'

type Draft = Pick<LoadProfile, 'model_id' | 'runtime_id' | 'backend' | 'parameters' | 'extra_args'> & Partial<Pick<LoadProfile, 'id' | 'name' | 'generation'>>

// Share only work in progress. Reopening still checks the files on the backend,
// where successful metadata checks are invalidated by file changes.
type PreflightRequest = {
  promise: Promise<PreparationStatus>; abort: AbortController; startedAt: number; progress?: OperationProgress
  listeners: Set<(progress: OperationProgress) => void>
}
const inFlight = new Map<string, PreflightRequest>()
function preflight(body: string, revision: string) {
  const key = JSON.stringify([body, revision])
  let request = inFlight.get(key)
  if (!request) {
    const entry: PreflightRequest = { promise: null!, abort: new AbortController(), startedAt: Date.now(), listeners: new Set() }
    entry.promise = requestProgress<PreparationStatus>('POST', '/api/profiles/preflight', JSON.parse(body), {
      signal: entry.abort.signal, onProgress: progress => {
        entry.progress = progress
        entry.listeners.forEach(listener => listener(progress))
      },
    })
    request = entry
    inFlight.set(key, request)
    const done = () => { if (inFlight.get(key) === request) inFlight.delete(key) }
    void request.promise.then(done, done)
  }
  return request
}

export function useProfilePreflight(draft: Draft, active = true, revision = '') {
  const key = JSON.stringify({ id: draft.id, name: draft.name, model_id: draft.model_id, runtime_id: draft.runtime_id,
    backend: draft.backend, parameters: draft.parameters, extra_args: draft.extra_args,
    ...(draft.backend === 'custom' ? { generation: draft.generation } : {}) })
  const scope = JSON.stringify([draft.model_id, draft.runtime_id, draft.backend])
  const [result, setResult] = useState<{ key: string; scope: string; value?: PreparationStatus; error?: string } | null>(null)
  const [retry, setRetry] = useState(0)
  const [pending, setPending] = useState(true)
  const [operation, setOperation] = useState<{ startedAt: number; progress?: OperationProgress }>({ startedAt: Date.now() })
  const lastKey = useRef(key)
  useEffect(() => {
    if (!active || !draft.runtime_id) { setPending(false); return }
    let disposed = false
    let release: (() => void) | undefined
    const delay = lastKey.current === key ? 0 : 250
    lastKey.current = key
    setPending(true)
    setOperation({ startedAt: Date.now() })
    const timer = window.setTimeout(async () => {
      const request = preflight(key, revision)
      setOperation({ startedAt: request.startedAt, progress: request.progress })
      const onProgress = (progress: OperationProgress) => { if (!disposed) setOperation({ startedAt: request.startedAt, progress }) }
      request.listeners.add(onProgress)
      release = () => {
        request.listeners.delete(onProgress)
        if (!request.listeners.size) {
          request.abort.abort()
          const requestKey = JSON.stringify([key, revision])
          if (inFlight.get(requestKey) === request) inFlight.delete(requestKey)
        }
      }
      try {
        const value = await request.promise
        if (!disposed) setResult({ key, scope, value })
      } catch (error) {
        if (!disposed) setResult({ key, scope, error: errorMessage(error) })
      } finally {
        if (!disposed) setPending(false)
      }
    }, delay)
    return () => { disposed = true; window.clearTimeout(timer); release?.() }
  }, [key, scope, active, revision, retry, draft.runtime_id, draft.backend])
  const checking = active && !!draft.runtime_id && (pending || result?.key !== key)
  return { status: result?.scope === scope ? result.value : undefined,
    checkedDraft: result?.scope === scope ? JSON.parse(result.key) as Draft : undefined, checking, ...operation,
    error: !checking && result?.key === key ? result.error : undefined, retry: () => setRetry(value => value + 1) }
}

export function PreparationDetails({ status, checking, progress, startedAt, error, retry, selected, onSelect }: {
  status?: PreparationStatus; checking?: boolean; error?: string; retry: () => void; selected?: string; onSelect: (id: string | undefined) => void
  progress?: OperationProgress; startedAt: number
}) {
  const ready = status?.status === 'ready'
  const heading = <><strong>模型准备文件</strong><span className={status?.status === 'error' || error ? 'error-text' : ''}>
    {error ? '检查失败' : checking || !status ? <OperationStatus progress={progress} startedAt={startedAt} fallback="检查准备文件" /> : ready ? status.source?.kind === 'explicit' ? '已指定本地文件' : '复用已有资源' : status?.status === 'needs_preparation' ? '首次加载时生成' : status?.status === 'choice_required' ? '选择已有资源' : status?.status === 'error' ? '需要处理' : ''}
  </span></>
  return <section className="profile-preparation" aria-label="模型准备文件">
    {ready && !error ? <details className="preparation-paths"><summary className="preparation-heading">{heading}</summary><dl>
      <dt>来源</dt><dd>{status.source?.label || '当前配置'}</dd>
      {status.pack && <><dt>运行文件</dt><dd className="mono">{status.pack}</dd></>}
      {status.tokenizer && <><dt>分词器</dt><dd className="mono">{status.tokenizer}</dd></>}
      {status.expert_profile && <><dt>专家配置</dt><dd className="mono">{status.expert_profile}</dd></>}
    </dl></details> : <div className="preparation-heading">{heading}</div>}
    {error ? <DiagnosticNotice summary="准备文件检查失败" detail={error} retry={retry} />
      : status && <>
        {status.status === 'error' && <DiagnosticNotice summary="准备文件不可用" detail={[status.message, ...(status.errors || [])].filter(Boolean).join('\n')} />}
        {(status.status === 'choice_required' || !!selected) && <Field label="使用已有准备文件"><Select value={selected || ''} onChange={event => onSelect(event.target.value || undefined)}>
          <option value="">{status.status === 'choice_required' ? '请选择资源来源' : '自动选择'}</option>
          {status.candidates.map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.label}</option>)}
        </Select></Field>}
      </>}
  </section>
}

export function PreparationSummary({ status, checking, progress, startedAt, error, retry }: {
  status?: PreparationStatus; checking: boolean; progress?: OperationProgress; startedAt: number; error?: string; retry: () => void
}) {
  if (checking) return <div className="preparation-summary"><OperationStatus progress={progress} startedAt={startedAt} fallback="检查准备文件" /></div>
  if (status?.status === 'ready' && !error) return null
  return <div className={`preparation-summary ${error || status?.status === 'error' || status?.status === 'choice_required' ? 'error-text' : ''}`}>
    {error ? <DiagnosticNotice summary="准备文件检查失败" detail={error} retry={retry} /> : status?.status === 'needs_preparation' ? '首次加载将生成本地运行文件。'
      : status?.status === 'ready' ? `准备文件：${status.source?.label || '已就绪'}` : status?.message || '正在检查模型准备文件…'}
  </div>
}
