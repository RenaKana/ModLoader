import { useCallback, useEffect, useRef, useState } from 'react'
import { BookOpen } from 'lucide-react'
import type { SourceRepository } from './types'
import { errorMessage, requestProgress } from './api'
import { Badge, Empty, ErrorNotice } from './components'
import { ReadmeDocument } from './ReadmeDocument'
import OperationStatus, { type OperationProgress } from './OperationStatus'
import './repository-readme.css'

type Readme = { content: string; path: string | null; url: string; revision: string }

export function RepositoryReadme({ repository }: { repository: SourceRepository }) {
  const [readme, setReadme] = useState<Readme | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [filePath, setFilePath] = useState<string | undefined>(undefined)
  const [progress, setProgress] = useState<OperationProgress | null>(null)
  const [startedAt, setStartedAt] = useState(Date.now())
  const introduction = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const controller = new AbortController()
    setReadme(null); setError(''); setProgress(null); setStartedAt(Date.now())
    void requestProgress<Readme>('POST', '/api/sources/readme',
      { source: repository.source, repo_id: repository.repo_id, revision: repository.revision, ...(filePath ? { file_path: filePath } : {}) },
      { signal: controller.signal, onProgress: value => { if (!controller.signal.aborted) setProgress(value) } },
    ).then(result => {
      if (!controller.signal.aborted) setReadme(result)
    }).catch(error => { if (!controller.signal.aborted) setError(errorMessage(error)) })
    return () => controller.abort()
  }, [repository.source, repository.repo_id, repository.revision, filePath, attempt])

  function openDocument(href: string) {
    const url = new URL(href)
    const revisions = new Set([repository.revision, readme?.revision, String(repository.metadata.requested_revision || 'main')])
    let path: string | null = null
    if (repository.source === 'hf' && url.origin === 'https://huggingface.co') {
      for (const revision of revisions) {
        for (const view of ['blob', 'resolve']) {
          const prefix = `/${repository.repo_id}/${view}/${encodeURIComponent(revision || '')}/`
          if (url.pathname.startsWith(prefix)) path = decodeURIComponent(url.pathname.slice(prefix.length))
        }
      }
    } else if (repository.source === 'modelscope' && url.origin === 'https://modelscope.cn' && url.pathname === `/models/${repository.repo_id}/files` && revisions.has(url.searchParams.get('revision') || 'main')) {
      path = url.searchParams.get('fileName')
    }
    if (!path || !/^README(?:[._-].*)?\.(?:md|markdown|rst|txt)$|^README$/i.test(path.split('/').at(-1) || '')) return false
    if (path !== readme?.path) setFilePath(path)
    introduction.current?.closest('.modal-body')?.scrollTo({ top: 0 })
    return true
  }

  const resolveUrl = useCallback((value: string, image: boolean) => {
    // Absolute URLs still pass through the Markdown renderer's scheme allowlist.
    if (/^[a-z][a-z\d+.-]*:/i.test(value) || value.startsWith('//')) return value.startsWith('//') ? `https:${value}` : value
    const revision = encodeURIComponent(readme?.revision || repository.revision)
    const file = (readme?.path || 'README.md').split('/').map(encodeURIComponent).join('/')
    if (repository.source === 'hf') {
      return new URL(value, `https://huggingface.co/${repository.repo_id}/${image ? 'resolve' : 'blob'}/${revision}/${file}`).href
    }
    if (value.startsWith('/')) return new URL(value, 'https://modelscope.cn').href
    const target = new URL(value, `https://modelscope.cn/${file}`)
    const filePath = decodeURIComponent(target.pathname.slice(1))
    return image
      ? `https://modelscope.cn/api/v1/models/${repository.repo_id}/repo?${new URLSearchParams({ Revision: readme?.revision || repository.revision, FilePath: filePath })}`
      : `https://modelscope.cn/models/${repository.repo_id}/files?${new URLSearchParams({ revision: readme?.revision || repository.revision, fileName: filePath })}${target.hash}`
  }, [readme?.path, readme?.revision, repository.source, repository.repo_id, repository.revision])

  const text = readme?.content.replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/, '').trim() || ''
  const tags = Array.isArray(repository.metadata.tags) ? repository.metadata.tags.filter((tag): tag is string => typeof tag === 'string') : []
  const license = tags.find(tag => tag.startsWith('license:'))?.slice(8)
  return <div className="repository-introduction" ref={introduction}>
    <div className="repository-facts"><Badge>{repository.source === 'hf' ? 'Hugging Face' : 'ModelScope'}</Badge>{license && <Badge>许可 · {license}</Badge>}<span>版本 {String(repository.metadata.requested_revision || repository.revision)}</span></div>
    {repository.description && <p className="repository-description">{repository.description}</p>}
    {error ? <ErrorNotice message={`README 读取失败：${error}`} retry={() => setAttempt(value => value + 1)} />
      : !readme ? <div className="readme-loading"><OperationStatus progress={progress} startedAt={startedAt} fallback="读取 README…" /></div>
      : !readme.path ? <Empty icon={BookOpen} title="仓库未提供 README" />
      : <><div className="readme-heading"><BookOpen size={15} /><span>{readme.path}</span></div>
        {text ? <ReadmeDocument text={text} resolveUrl={resolveUrl} onDocumentLink={openDocument} /> : <p className="muted">README 为空。</p>}
        <details className="readme-source"><summary>查看 README 原文</summary><pre>{readme.content}</pre></details>
      </>}
  </div>
}
