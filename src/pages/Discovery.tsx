import { useEffect, useRef, useState } from 'react'
import { ArrowDownToLine, ArrowRight, BookOpen, Check, Download, ExternalLink, File, FolderDown, Globe2, Link2, Search, SlidersHorizontal, X } from 'lucide-react'
import type { PageProps, SourceFile, SourceGroup, SourceRepository, SourceSearchItem } from '../types'
import { errorMessage, isJobActive, openExternal, request, requestProgress } from '../api'
import { Badge, Button, Empty, ErrorNotice, Field, IconButton, JobRow, KeyValue, Modal, PageHeader, formatBytes } from '../components'
import { useAction } from '../hooks'
import { RepositoryReadme } from '../RepositoryReadme'
import OperationStatus, { type OperationProgress } from '../OperationStatus'
import '../download-selection.css'

type SearchResult = { items: SourceSearchItem[]; next_cursor: string | null }

export default function Discovery(props: PageProps) {
  const [source, setSource] = useState('hf')
  const [query, setQuery] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [results, setResults] = useState<SourceSearchItem[]>([])
  const [cursor, setCursor] = useState<string | null>(null)
  const [searching, setSearching] = useState<'search' | 'more' | null>(null)
  const [searched, setSearched] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [resolving, setResolving] = useState<string | null>(null)
  const [resolvingTab, setResolvingTab] = useState<'intro' | 'files'>('files')
  const [resolveProgress, setResolveProgress] = useState<OperationProgress | null>(null)
  const [resolveStartedAt, setResolveStartedAt] = useState(Date.now())
  const searchController = useRef<AbortController | null>(null)
  const resolveController = useRef<AbortController | null>(null)
  const [repository, setRepository] = useState<SourceRepository | null>(null)
  const [repositoryTab, setRepositoryTab] = useState<'intro' | 'files'>('files')
  const [directOpen, setDirectOpen] = useState(false)
  const [repoOpen, setRepoOpen] = useState(false)
  const [jobFilter, setJobFilter] = useState('active')
  const action = useAction(props.notify, props.refresh)
  const jobs = props.state.downloads.filter(job => jobFilter === 'all' || isJobActive(job) || ['paused', 'failed', 'error'].includes(job.state || job.status || ''))
  const activeJobs = props.state.downloads.filter(isJobActive)
  useEffect(() => () => { searchController.current?.abort(); resolveController.current?.abort() }, [])

  function cancelSearch() {
    searchController.current?.abort(); searchController.current = null; setSearching(null)
  }
  function cancelResolve() {
    resolveController.current?.abort(); resolveController.current = null; setResolving(null); setResolveProgress(null)
  }
  function changeSource(value: string) {
    cancelSearch(); cancelResolve(); setSource(value); setResults([]); setSearched(false); setSearchError(''); setCursor(null)
  }
  function changeQuery(value: string) {
    cancelSearch(); cancelResolve(); setQuery(value)
  }
  function openRepository() { cancelResolve(); setRepoOpen(true) }

  async function search(more = false) {
    if (!query.trim() && !more) return
    const controller = new AbortController()
    searchController.current?.abort(); searchController.current = controller; cancelResolve()
    setSearching(more ? 'more' : 'search'); setSearchError('')
    const searchText = more ? searchQuery : query.trim()
    try {
      const payload = await requestProgress<SearchResult>('GET', `/api/sources/${source}/search?q=${encodeURIComponent(searchText)}${more && cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`, undefined, { signal: controller.signal })
      if (controller.signal.aborted) return
      const items = payload.items
      setResults(previous => more ? [...previous, ...items.filter(item => !previous.some(existing => existing.id === item.id))] : items)
      setCursor(payload.next_cursor)
      setSearchQuery(searchText); setSearched(true)
    } catch (error) { if (!controller.signal.aborted) setSearchError(errorMessage(error)) }
    finally { if (searchController.current === controller) { searchController.current = null; setSearching(null) } }
  }

  async function resolve(repo: string, revision = 'main', tab: 'intro' | 'files' = 'files') {
    const controller = new AbortController()
    resolveController.current?.abort(); resolveController.current = controller; cancelSearch()
    setResolving(repo); setResolvingTab(tab); setResolveProgress(null); setResolveStartedAt(Date.now())
    try {
      const result = await requestProgress<SourceRepository | { repository: SourceRepository }>('POST', '/api/sources/resolve', { source, repo_id: repo.trim(), revision }, { signal: controller.signal, onProgress: value => { if (!controller.signal.aborted) setResolveProgress(value) } })
      if (controller.signal.aborted) return
      setRepository('repository' in result ? result.repository : result)
      setRepositoryTab(tab)
      setRepoOpen(false)
    } catch (error) { if (!controller.signal.aborted) props.notify(errorMessage(error), 'error') }
    finally { if (resolveController.current === controller) { resolveController.current = null; setResolving(null) } }
  }

  return <>
    <PageHeader eyebrow="FIND YOUR NEXT MODEL" title="发现模型"><Button icon={Link2} onClick={() => { cancelResolve(); setDirectOpen(true) }}>直接链接下载</Button><Button icon={FolderDown} onClick={openRepository}>打开模型仓库</Button></PageHeader>
    <div className="discover-layout"><div className="discover-main">
      <div className="panel source-search"><div className="source-tabs"><button className={source === 'hf' ? 'active' : ''} onClick={() => changeSource('hf')}><span className="source-letter hf">H</span>Hugging Face</button><button className={source === 'modelscope' ? 'active' : ''} onClick={() => changeSource('modelscope')}><span className="source-letter ms">M</span>ModelScope</button></div>
        <form className="search-form" onSubmit={event => { event.preventDefault(); void search() }}><div className="search-input"><Search size={18} /><input aria-label="搜索在线模型" value={query} onChange={event => changeQuery(event.target.value)} placeholder="模型名称 / 仓库 ID" />{query && <IconButton icon={X} label="清空搜索" onClick={() => changeQuery('')} />}</div><Button variant="primary" icon={Search} busy={searching === 'search'} disabled={!query.trim()}>{searching === 'search' ? '搜索中…' : '搜索'}</Button></form>
        <div className="source-note"><Globe2 size={13} /><span>{source === 'hf' ? 'huggingface.co' : 'modelscope.cn'}</span><Badge tone={(source === 'hf' ? props.state.settings.hf_configured : props.state.settings.modelscope_configured) ? 'green' : 'neutral'}>{(source === 'hf' ? props.state.settings.hf_configured : props.state.settings.modelscope_configured) ? '已配置凭据' : '公开访问'}</Badge></div></div>
      {searchError && <ErrorNotice message={searchError} retry={() => void search()} />}
      {searched && <div className="results-heading"><h3>“{searchQuery}”</h3><span>{results.length} 个结果</span></div>}
      {results.length ? <div className="repository-list">{results.map(repo => <div className="panel repository-card" key={`${source}-${repo.id}`}><div className="repository-icon"><FolderDown size={21} /></div><div className="repository-content"><div className="repository-title"><h3>{repo.name || repo.id.split('/').at(-1)}</h3>{repo.gated && <Badge tone="amber">需授权访问</Badge>}</div><span className="repository-id mono">{repo.id}</span>{repo.description && <p>{repo.description}</p>}<div className="repository-tags">{typeof repo.downloads === 'number' && <span><Download size={12} />{repo.downloads.toLocaleString('zh-CN')}</span>}</div>{resolving === repo.id && !repoOpen && <OperationStatus progress={resolveProgress} startedAt={resolveStartedAt} fallback="读取仓库清单…" />}</div><div className="repository-actions"><Button small icon={BookOpen} busy={resolving === repo.id && resolvingTab === 'intro'} onClick={() => void resolve(repo.id, 'main', 'intro')}>模型介绍</Button>{repo.url && <IconButton icon={ExternalLink} label="在浏览器打开仓库" onClick={() => void openExternal(repo.url)} />}<Button small icon={ArrowRight} busy={resolving === repo.id && resolvingTab === 'files'} onClick={() => void resolve(repo.id)}>选择文件</Button>{resolving === repo.id && !repoOpen && <Button small variant="ghost" onClick={cancelResolve}>取消读取</Button>}</div></div>)}</div>
        : <div className="panel catalog-empty"><Empty icon={searched || searching === 'search' ? Search : Globe2} title={searching === 'search' ? '正在搜索模型…' : searched ? '没有找到匹配的仓库' : '查找需要的模型'}>{searching !== 'search' && <Button small icon={FolderDown} onClick={openRepository}>打开模型仓库</Button>}</Empty></div>}
      {cursor && <div className="load-more"><Button busy={searching === 'more'} disabled={searching === 'search'} onClick={() => void search(true)}>{searching === 'more' ? '读取更多…' : '加载更多'}</Button></div>}
    </div><aside className="panel download-panel"><div className="panel-heading"><div><h3><ArrowDownToLine size={16} />下载任务 <span className="heading-count">{activeJobs.length}</span></h3></div></div><div className="sub-tabs compact"><button className={jobFilter === 'active' ? 'active' : ''} onClick={() => setJobFilter('active')}>进行中</button><button className={jobFilter === 'all' ? 'active' : ''} onClick={() => setJobFilter('all')}>全部任务</button></div><div className="download-list">{jobs.length ? jobs.map(job => <JobRow key={job.id} job={job} busy={action.busy} onPause={() => void action.run(() => request('POST', `/api/jobs/${job.id}/pause`), '已请求暂停下载')} onResume={() => void action.run(() => request('POST', `/api/jobs/${job.id}/resume`), '已继续下载')} onCancel={() => void action.run(() => request('POST', `/api/jobs/${job.id}/cancel`), '已请求停止下载')} onRetry={() => void action.run(() => request('POST', `/api/downloads/${job.id}/retry`), '下载任务已重试')} />) : <Empty icon={Download} title="暂无下载任务" />}</div><div className="download-location"><FolderDown size={14} /><span title={props.state.settings.model_directory}>{props.state.settings.model_directory || '模型目录由服务指定'}</span><IconButton icon={SlidersHorizontal} label="设置下载目录" onClick={() => props.navigate('settings')} /></div></aside></div>
    {repository && <DownloadRepository key={`${repository.source}-${repository.repo_id}`} repository={repository} initialTab={repositoryTab} onClose={() => setRepository(null)} {...props} />}
    {directOpen && <DirectDownload onClose={() => setDirectOpen(false)} {...props} />}
    {repoOpen && <OpenRepository source={source} onClose={() => { cancelResolve(); setRepoOpen(false) }} onCancel={cancelResolve} busy={!!resolving} progress={resolveProgress} startedAt={resolveStartedAt} onResolve={resolve} />}
  </>
}

function OpenRepository({ source, onClose, onCancel, busy, progress, startedAt, onResolve }: { source: string; onClose: () => void; onCancel: () => void; busy: boolean; progress: OperationProgress | null; startedAt: number; onResolve: (repo: string, revision: string) => Promise<void> }) {
  const [repo, setRepo] = useState('')
  const [revision, setRevision] = useState('main')
  return <Modal title={`打开 ${source === 'hf' ? 'Hugging Face' : 'ModelScope'} 仓库`} onClose={onClose} footer={<><Button onClick={busy ? onCancel : onClose}>{busy ? '取消读取' : '取消'}</Button><Button variant="primary" busy={busy} disabled={!repo.trim()} onClick={() => void onResolve(repo, revision || 'main')}>读取文件清单</Button></>}><Field label="仓库 ID"><input autoFocus value={repo} onChange={event => { onCancel(); setRepo(event.target.value) }} placeholder="owner/repo" /></Field><Field label="版本或分支"><input value={revision} onChange={event => { onCancel(); setRevision(event.target.value) }} placeholder="main" /></Field>{busy && <OperationStatus progress={progress} startedAt={startedAt} fallback="读取仓库清单…" />}</Modal>
}

type DownloadPlan = { total_bytes?: number; destination?: string; revision?: string; warnings?: string[]; files?: SourceFile[]; groups?: SourceGroup[]; unknown_size_files?: string[]; automatic_files?: SourceFile[]; optional_files?: SourceFile[]; model_groups?: SourceGroup[]; vision_groups?: SourceGroup[]; required_vision_groups?: SourceGroup[]; requires_vision_selection?: boolean }

function selectionSize(files: SourceFile[]) {
  const known = files.reduce((sum, file) => sum + (file.size ?? 0), 0)
  const unknown = files.filter(file => file.size === null).length
  return unknown ? `${known ? `至少 ${formatBytes(known)} · ` : ''}${unknown} 项大小未知` : formatBytes(known)
}

function groupEntry(group: SourceGroup) {
  return group.files.find(path => /\.(?:safetensors|bin)\.index\.json$/i.test(path)) || group.files.find(path => /\.(?:gguf|safetensors|bin|pt|pth)$/i.test(path)) || group.files[0]
}

function fileReason(file: SourceFile) {
  const reasons: Record<string, string> = { shard: '模型分片', required: '模型配置或 tokenizer', vision: '视觉组件', model: '模型权重' }
  return file.reason || reasons[file.role || ''] || '额外文件'
}

function DownloadRepository({ repository, initialTab, onClose, ...props }: PageProps & { repository: SourceRepository; initialTab: 'intro' | 'files'; onClose: () => void }) {
  const [tab, setTab] = useState(initialTab)
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const [includeVision, setIncludeVision] = useState(false)
  const [fileQuery, setFileQuery] = useState('')
  const [revision, setRevision] = useState(repository.revision || 'main')
  const [plan, setPlan] = useState<DownloadPlan | null>(null)
  const [requiredVisionGroups, setRequiredVisionGroups] = useState<SourceGroup[]>([])
  const [requiredChoiceGroups, setRequiredChoiceGroups] = useState<SourceGroup[] | null>(null)
  const [planning, setPlanning] = useState(false)
  const [progress, setProgress] = useState<OperationProgress | null>(null)
  const [startedAt, setStartedAt] = useState(Date.now())
  const planController = useRef<AbortController | null>(null)
  const action = useAction(props.notify, props.refresh)
  useEffect(() => () => planController.current?.abort(), [])
  const groups = plan?.groups || requiredChoiceGroups || repository.groups || []
  const modelGroups = groups.filter(group => group.kind === 'model')
  const visionGroups = groups.filter(group => group.kind === 'vision')
  const chosenGroups = groups.filter(group => group.files.some(path => selected.has(path) && /\.(?:(?:safetensors|bin)\.index\.json|gguf|safetensors|bin|pt|pth)$/i.test(path)))
  const chosenModelGroups = chosenGroups.filter(group => group.kind === 'model')
  const groupedPaths = new Set(chosenGroups.flatMap(group => group.files))
  const effectivePaths = new Set([...selected, ...groupedPaths])
  const files = repository.files.filter(file => file.path.toLowerCase().includes(fileQuery.toLowerCase()))
  const query = fileQuery.toLowerCase()
  const shownGroups = modelGroups.filter(group => group.name.toLowerCase().includes(query) || group.files.some(path => path.toLowerCase().includes(query)))
  const body = { source: repository.source, repo_id: repository.repo_id, revision, files: [...selected], include_vision: includeVision }
  const plannedFiles = new Map(plan?.files?.map(file => [file.path, file]) || [])
  const effectiveFiles = plan?.files || repository.files.filter(file => effectivePaths.has(file.path))
  const automaticFiles = plan?.automatic_files || repository.files.filter(file => groupedPaths.has(file.path) && !selected.has(file.path))
  const optionalFiles = plan?.optional_files || repository.files.filter(file => file.role === 'vision' && !effectivePaths.has(file.path))
  const modelFiles = repository.files.filter(file => chosenModelGroups.some(group => group.files.includes(file.path)))
  function cancelPlan() {
    planController.current?.abort(); planController.current = null; setPlanning(false); setProgress(null)
  }
  function close() { cancelPlan(); onClose() }
  function toggle(path: string) {
    cancelPlan()
    const group = groups.find(item => item.files.includes(path) && /\.(?:(?:safetensors|bin)\.index\.json|gguf|safetensors|bin|pt|pth)$/i.test(path))
    if (group) { toggleGroup(group); return }
    setSelected(previous => { const next = new Set(previous); if (next.has(path)) next.delete(path); else next.add(path); return next }); setPlan(null); setRequiredVisionGroups([])
  }
  function toggleGroup(group: SourceGroup) {
    cancelPlan()
    const chosen = chosenGroups.some(item => item.id === group.id)
    setSelected(previous => { const next = new Set(previous); if (chosen) group.files.forEach(path => next.delete(path)); else next.add(groupEntry(group)); return next }); setPlan(null)
    if (group.kind === 'model') { setRequiredVisionGroups([]); setRequiredChoiceGroups(null) }
  }
  function toggleVision(value: boolean) {
    cancelPlan()
    setIncludeVision(value)
    if (!value) setSelected(previous => new Set([...previous].filter(path => requiredVisionGroups.some(group => group.files.includes(path)) || !repository.files.some(file => file.path === path && file.role === 'vision'))))
    setPlan(null)
  }
  async function submit() {
    if (!plan || plan.requires_vision_selection) {
      const controller = new AbortController()
      planController.current?.abort(); planController.current = controller
      setPlanning(true); setProgress(null); setStartedAt(Date.now())
      try {
        const result = await requestProgress<DownloadPlan>('POST', '/api/downloads/plan', body, { signal: controller.signal, onProgress: value => { if (!controller.signal.aborted) setProgress(value) } })
        if (controller.signal.aborted) return
        setPlan(result); setRequiredVisionGroups(result.required_vision_groups || [])
        if (result.requires_vision_selection && !includeVision) setRequiredChoiceGroups(result.groups || groups)
      } catch (error) { if (!controller.signal.aborted) props.notify(errorMessage(error), 'error') }
      finally { if (planController.current === controller) { planController.current = null; setPlanning(false) } }
    } else {
      const result = await action.run(() => request('POST', '/api/downloads', { ...body, revision: plan.revision || revision, checked_files: plan.files }), '已创建下载任务')
      if (result.ok) onClose()
    }
  }
  function groupChoice(group: SourceGroup) {
    const checked = chosenGroups.some(item => item.id === group.id) || !!plan?.vision_groups?.some(item => item.id === group.id)
    const members = group.files.map(path => repository.files.find(file => file.path === path)).filter((file): file is SourceFile => !!file)
    const weights = members.filter(file => /\.(?:gguf|safetensors|bin|pt|pth)$/i.test(file.path))
    return <div className={`download-model-choice ${checked ? 'selected' : ''}`} key={group.id}>
      <label className="download-model-row"><input type="checkbox" aria-label={group.name} checked={checked} disabled={action.busy} onChange={() => toggleGroup(group)} /><span className="download-model-name"><strong>{group.name}</strong><span>{group.format.toUpperCase()}{group.quantization && ` · ${group.quantization}`} · {weights.length > 1 ? `${weights.length} 个分片` : '1 个权重'}</span></span><span className="download-model-size">{group.size === null ? '大小未知' : formatBytes(group.size)}</span></label>
      <details className="download-group-contents"><summary>{members.length} 项文件</summary><div>{members.map(file => <div className="download-group-file" key={file.path}><File size={13} /><span title={file.path}>{file.path}<small>{fileReason(plannedFiles.get(file.path) || file)}</small></span><small>{formatBytes(file.size)}</small></div>)}</div></details>
    </div>
  }
  return <Modal wide title={tab === 'intro' ? '模型介绍' : '选择下载文件'} description={repository.repo_id} dismissible={!action.busy} onClose={close} footer={tab === 'intro' ? <>
    {repository.url && <div className="repository-footer-source"><IconButton icon={ExternalLink} label="在源站查看" onClick={() => void openExternal(repository.url)} /></div>}<Button onClick={close}>关闭</Button><Button variant="primary" icon={ArrowRight} onClick={() => setTab('files')}>选择文件</Button>
  </> : <><span className="footer-note">{effectiveFiles.length} 项 · {selectionSize(effectiveFiles)}</span><Button disabled={action.busy} onClick={planning ? cancelPlan : close}>{planning ? '取消检查' : '取消'}</Button><Button variant="primary" icon={Download} busy={action.busy || planning} disabled={!selected.size} onClick={() => void submit()}>{plan && !plan.requires_vision_selection ? '开始下载' : '检查下载'}</Button></>}>
    <div className="repository-tabs" role="group" aria-label="仓库内容"><Button small icon={BookOpen} disabled={action.busy} aria-pressed={tab === 'intro'} variant={tab === 'intro' ? 'primary' : 'default'} onClick={() => { cancelPlan(); setTab('intro') }}>模型介绍</Button><Button small icon={File} disabled={action.busy} aria-pressed={tab === 'files'} variant={tab === 'files' ? 'primary' : 'default'} onClick={() => setTab('files')}>下载文件</Button></div>
    {tab === 'intro' ? <RepositoryReadme repository={repository} /> : <div className="download-repo-layout download-selection"><div className="download-file-selector">
      <div className="search-input"><Search size={15} /><input aria-label="筛选仓库文件" value={fileQuery} onChange={event => setFileQuery(event.target.value)} placeholder="文件名" /></div>
      <div className="file-select-toolbar"><span>{modelGroups.length} 个模型版本</span><button disabled={action.busy || !modelGroups.length} onClick={() => { cancelPlan(); setSelected(previous => new Set([...previous, ...modelGroups.map(groupEntry)])); setPlan(null); setRequiredVisionGroups([]); setRequiredChoiceGroups(null) }}>全选模型</button><button disabled={action.busy || !selected.size} onClick={() => { cancelPlan(); setSelected(new Set()); setPlan(null); setRequiredVisionGroups([]); setRequiredChoiceGroups(null) }}>清空</button></div>
      <div className="download-model-options">{shownGroups.map(groupChoice)}{!shownGroups.length && <p className="download-no-models muted">{modelGroups.length ? '没有匹配的模型版本。' : '未识别到模型版本，可展开全部文件选择。'}</p>}</div>
      <div className="download-vision-option"><label><input type="checkbox" checked={includeVision} disabled={action.busy} onChange={event => toggleVision(event.target.checked)} /><span>同时下载视觉组件<small>模型必需的组件仍会自动补齐</small></span></label>{(includeVision || requiredChoiceGroups || plan?.requires_vision_selection) && <div className="download-vision-choices">{visionGroups.length === 1 ? <p>检查时自动补入匹配组件。</p> : visionGroups.length ? <><p>选择与模型匹配的组件。</p>{visionGroups.map(groupChoice)}</> : <p>未找到视觉组件，请查看模型说明。</p>}</div>}</div>
      <details className="download-all-files"><summary>全部文件 <span>{repository.files.length} 项</span></summary><div className="file-select-toolbar"><button disabled={action.busy} onClick={() => { cancelPlan(); setSelected(new Set(repository.files.filter(file => includeVision || file.role !== 'vision').map(file => file.path))); setPlan(null); setRequiredVisionGroups([]); setRequiredChoiceGroups(null) }}>全选文件</button></div><div className="download-files">{files.map(file => {
        const planned = plannedFiles.get(file.path)
        const automatic = !!planned && !selected.has(file.path)
        const locked = groupedPaths.has(file.path) || automatic
        const checked = effectivePaths.has(file.path) || !!planned
        return <label className={`download-file ${checked ? 'selected' : ''}`} key={file.path}>
          <input type="checkbox" aria-label={`文件 ${file.path}`} checked={checked} disabled={action.busy || locked} onChange={() => toggle(file.path)} /><File size={15} /><span title={file.path}>{file.path}{checked && locked && <small className="automatic-file">{fileReason(planned || file)}</small>}</span><small>{formatBytes(file.size)}</small>
        </label>
      })}{!files.length && <p className="muted">没有匹配的文件。</p>}</div></details>
    </div><div className="download-selection-summary">
      <Field label="仓库版本"><input value={revision} disabled={action.busy} onChange={event => { cancelPlan(); setRevision(event.target.value); setPlan(null); setRequiredVisionGroups([]); setRequiredChoiceGroups(null) }} /></Field><KeyValue label="来源" value={repository.source === 'hf' ? 'Hugging Face' : 'ModelScope'} />
      {planning && <OperationStatus progress={progress} startedAt={startedAt} fallback="检查下载文件…" />}
      <div className="download-selection-counts"><KeyValue label="已选模型" value={`${chosenModelGroups.length} 个 · ${selectionSize(modelFiles)}`} /><KeyValue label="自动补齐" value={`${automaticFiles.length} 项 · ${selectionSize(automaticFiles)}`} /><KeyValue label="可选配套" value={`${optionalFiles.length} 项 · ${selectionSize(optionalFiles)}`} /></div>
      <p className="selection-guide">必要分片不可单独取消。</p>
      {plan && <div className={`download-plan ${plan.requires_vision_selection ? 'needs-choice' : ''}`}><span className="plan-ready"><Check size={15} />{plan.requires_vision_selection ? '请选择视觉组件' : '下载计划已生成'}</span>{plan.revision && <p className="mono" title={plan.revision}>版本 {plan.revision.length > 18 ? plan.revision.slice(0, 12) : plan.revision}</p>}{plan.destination && <p className="mono">{plan.destination}</p>}{!!plan.unknown_size_files?.length && <p>{plan.unknown_size_files.length} 项文件大小未知。</p>}{plan.warnings?.map(warning => <p className="warning-text" key={warning}>{warning}</p>)}<details className="download-plan-details"><summary>查看 {effectiveFiles.length} 项下载文件与原因</summary>{effectiveFiles.map(file => <div key={file.path}><span className="mono">{file.path}</span><small>{fileReason(file)} · {formatBytes(file.size)}</small></div>)}</details></div>}
    </div></div>}
  </Modal>
}

function DirectDownload({ onClose, ...props }: PageProps & { onClose: () => void }) {
  const [url, setUrl] = useState('')
  const [filename, setFilename] = useState('')
  const [plan, setPlan] = useState<DownloadPlan | null>(null)
  const [planning, setPlanning] = useState(false)
  const [progress, setProgress] = useState<OperationProgress | null>(null)
  const [startedAt, setStartedAt] = useState(Date.now())
  const planController = useRef<AbortController | null>(null)
  const action = useAction(props.notify, props.refresh)
  useEffect(() => () => planController.current?.abort(), [])
  const body = { source: 'url', url: url.trim(), filename: filename.trim() }
  function cancelPlan() { planController.current?.abort(); planController.current = null; setPlanning(false); setProgress(null) }
  function close() { cancelPlan(); onClose() }
  async function submit() {
    if (!plan) {
      const controller = new AbortController()
      planController.current?.abort(); planController.current = controller
      setPlanning(true); setProgress(null); setStartedAt(Date.now())
      try {
        const result = await requestProgress<DownloadPlan>('POST', '/api/downloads/plan', body, { signal: controller.signal, onProgress: value => { if (!controller.signal.aborted) setProgress(value) } })
        if (!controller.signal.aborted) setPlan(result)
      } catch (error) { if (!controller.signal.aborted) props.notify(errorMessage(error), 'error') }
      finally { if (planController.current === controller) { planController.current = null; setPlanning(false) } }
    } else {
      const result = await action.run(() => request('POST', '/api/downloads', { ...body, checked_files: plan.files }), '已创建下载任务')
      if (result.ok) onClose()
    }
  }
  return <Modal title="直接链接下载" dismissible={!action.busy} onClose={close} footer={<><Button disabled={action.busy} onClick={planning ? cancelPlan : close}>{planning ? '取消检查' : '取消'}</Button><Button icon={Download} variant="primary" busy={action.busy || planning} disabled={!url.trim()} onClick={() => void submit()}>{plan ? '开始下载' : '检查下载'}</Button></>}><Field label="下载链接"><textarea rows={3} autoFocus disabled={action.busy} value={url} onChange={event => { cancelPlan(); setUrl(event.target.value); setPlan(null) }} placeholder="https://…/model.gguf" /></Field><Field label="保存文件名" help="留空自动解析。"><input disabled={action.busy} value={filename} onChange={event => { cancelPlan(); setFilename(event.target.value); setPlan(null) }} /></Field>{planning && <OperationStatus progress={progress} startedAt={startedAt} fallback="检查链接与文件大小…" />}{plan && <div className="download-plan"><span className="plan-ready"><Check size={15} />下载计划已生成</span>{plan.destination && <p className="mono">{plan.destination}</p>}{plan.total_bytes !== undefined && <p>{formatBytes(plan.total_bytes)}</p>}{plan.warnings?.map(warning => <p key={warning}>{warning}</p>)}</div>}</Modal>
}
