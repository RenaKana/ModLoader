import { useEffect, useRef, useState } from 'react'
import { ArrowRightLeft, Check, ChevronLeft, ChevronRight, Download, Loader2, Pause, Play, Square, Trash2 } from 'lucide-react'
import { apiFetch, errorMessage, isJobActive, jobState, request } from './api'
import { Badge, Button, ErrorNotice, IconButton, jobBadgeState, Modal, StateBadge, formatBytes } from './components'
import { useAction } from './hooks'
import type { InstallationHistory, Job, PageProps, RuntimeInstallation } from './types'
import RuntimeManifest from './RuntimeManifest'

export const isInstallationTerminal = (job: Job) => ['completed', 'complete', 'failed', 'error', 'cancelled', 'canceled', 'interrupted', 'stopped'].includes(jobState(job))

function timeLabel(value?: string | null) {
  if (!value) return '时间未记录'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function Progress({ label, ariaLabel, completed, total, active }: { label: string; ariaLabel?: string; completed?: number; total?: number | null; active: boolean }) {
  const hasTotal = typeof total === 'number' && Number.isFinite(total) && total >= 0
  const known = hasTotal && total > 0 && typeof completed === 'number' && Number.isFinite(completed)
  const percent = known ? Math.max(0, Math.min(100, completed / total * 100)) : 0
  return <div className="runtime-progress"><div><strong>{formatBytes(completed)} / {formatBytes(total)}</strong></div><div className={`progress-track ${active && !hasTotal ? 'indeterminate' : ''}`} role="progressbar" aria-label={ariaLabel || label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={known ? Math.round(percent) : undefined}><span style={{ width: `${percent}%` }} /></div></div>
}

function InstallationRow({ job, props, busy, onConfirm, onRetry, onSwitch, onDelete, onControl, history = false }: {
  job: Job; props: PageProps; busy: boolean; history?: boolean; onConfirm: () => void; onRetry: () => void;
  onSwitch: (source: RuntimeInstallation, target: string) => void; onDelete: () => void; onControl: (operation: 'pause' | 'resume' | 'cancel') => void;
}) {
  const status = jobState(job)
  const active = isJobActive(job)
  const terminal = isInstallationTerminal(job)
  const awaiting = status === 'awaiting_confirmation'
  const resolving = job.phase === 'resolve'
  const verifyingCache = job.phase === 'verify_cache'
  const transition = ['pausing', 'cancelling'].includes(status)
  const stages = job.stages?.length ? job.stages : job.manifest ? [job.manifest] : []
  const files = stages.flatMap(stage => stage.files || [])
  const completedFiles = job.completed_files ?? files.filter(file => ['completed', 'complete', 'cached', 'downloaded'].includes(file.status || '') || file.cached).length
  const totalFiles = job.total_files ?? files.length
  const backend = props.state.backends.find(item => item.id === job.backend)
  const source = props.state.runtimes.find(runtime => runtime.id === job.update_from)
  const target = props.state.runtimes.find(runtime => runtime.id === job.runtime_id)
  const switchAvailable = job.operation === 'update' && ['complete', 'completed'].includes(status) && source?.managed && target
  const references = source ? props.state.profiles.filter(profile => profile.runtime_id === source.id) : []
  const currentFile = job.current_file || job.file
  const startupPrefetch = job.operation === 'prepare_startup_prefetch'
  const downloading = files.filter(file => file.status === 'downloading')
  const currentFiles = downloading.length ? downloading.map(file => ({
    id: file.id, name: file.name, downloaded: file.downloaded_bytes ?? (file.name === currentFile ? job.file_downloaded_bytes : 0), total: file.size_bytes,
  })) : currentFile ? [{ id: 'current', name: currentFile, downloaded: job.file_downloaded_bytes, total: job.file_total_bytes }] : []
  const phase = String(job.phase || '').toLowerCase().replaceAll('-', '_')
  const downloadPhase = ['download', 'downloading'].includes(phase)
    || (phase === 'prepare' && status === 'running' && totalFiles > 0)
    || (!phase && (status === 'downloading' || downloading.length > 0))
  const fileList = <div className="runtime-job-stages" role="region" aria-label="安装文件明细" tabIndex={0}>{stages.map(stage => <RuntimeManifest manifest={stage} key={stage.id} />)}</div>

  return <article className={`runtime-install-job job-row ${history ? 'runtime-history-job' : ''}`} data-job-id={job.id}>
    <div className="job-top"><div className="job-title">{active ? <Loader2 size={16} className="spin" /> : ['complete', 'completed'].includes(status) ? <Check size={16} /> : <Download size={16} />}<strong>{job.name || job.label || backend?.label || job.backend || '环境安装'}</strong><Badge>{startupPrefetch ? '启动预取' : job.operation === 'update' ? '更新' : '安装'}</Badge>{job.version && <span className="runtime-job-version">{job.version}</span>}</div><StateBadge state={jobBadgeState(job)} /></div>
    <div className="runtime-job-times"><span>创建 {timeLabel(job.created_at)}</span>{job.started_at && <span>开始 {timeLabel(job.started_at)}</span>}{terminal && <span>结束 {timeLabel(job.finished_at)}</span>}</div>
    {!terminal && resolving && <p className="runtime-subtle">正在解析依赖，下载大小待确定。</p>}
    {!terminal && verifyingCache && <p className="runtime-subtle">正在校验下载缓存。</p>}
    {startupPrefetch && !terminal && !downloadPhase && <div className={`progress-track ${active ? 'indeterminate' : ''}`} role="progressbar" aria-label="启动预取准备进度"><span /></div>}
    {!terminal && downloadPhase && !resolving && !verifyingCache && totalFiles > 0 && <div className="runtime-job-progress"><Progress label="总下载进度" completed={job.downloaded_bytes ?? job.completed_bytes} total={job.total_bytes} active={active && !awaiting} /><div className="runtime-file-count">{completedFiles} / {totalFiles} 文件</div>{currentFiles.map(file => <div className="runtime-current-file" key={file.id}><span className="mono" title={file.name}>{file.name}</span><Progress label="当前文件进度" ariaLabel={currentFiles.length > 1 ? `${file.name} 下载进度` : undefined} completed={file.downloaded} total={file.total} active={active && !awaiting} /></div>)}</div>}
    {job.message && <p className="runtime-job-message">{job.message}</p>}{job.error && <p className="runtime-job-error" role="alert">{job.error}</p>}
    {stages.length > 0 && <details className="runtime-history-files"><summary>文件清单 · {totalFiles}</summary>{fileList}</details>}
    <div className="runtime-job-actions">
      {!terminal && active && job.can_pause && !awaiting && <Button small icon={Pause} disabled={busy || transition} onClick={() => onControl('pause')}>暂停</Button>}
      {status === 'paused' && <Button small icon={Play} disabled={busy} onClick={() => onControl('resume')}>继续</Button>}
      {awaiting && job.manifest && <Button small icon={Play} variant="primary" disabled={busy} onClick={onConfirm}>继续安装</Button>}
      {!terminal && <Button small icon={Square} disabled={busy || status === 'cancelling'} onClick={() => onControl('cancel')}>{status === 'cancelling' ? '停止中' : '停止'}</Button>}
      {['failed', 'error', 'cancelled', 'canceled', 'interrupted', 'stopped'].includes(status) && <Button small disabled={busy || (startupPrefetch ? !target || props.state.installations.some(item => item.id !== job.id && item.operation === 'prepare_startup_prefetch' && item.runtime_id === job.runtime_id && isJobActive(item)) : !backend || backend.installable === false)} onClick={onRetry}>{startupPrefetch ? '重试准备' : '重试安装'}</Button>}
      {switchAvailable && <Button small icon={ArrowRightLeft} disabled={busy || !references.length} title={!references.length ? '原环境已无待切换的加载配置' : undefined} onClick={() => onSwitch(source, target.id)}>切换配置</Button>}
      {history && terminal && <Button small icon={Trash2} variant="ghost" disabled={busy} onClick={onDelete}>删除记录</Button>}
    </div>
  </article>
}

export default function RuntimeInstallJobs({ active, onRetry, onSwitch, ...props }: PageProps & { active: boolean; onRetry: (job: Job) => void; onSwitch: (source: RuntimeInstallation, target: string) => void }) {
  const unfinished = props.state.installations.filter(job => !isInstallationTerminal(job))
  const [tab, setTab] = useState<'unfinished' | 'history'>('unfinished')
  const [page, setPage] = useState(1)
  const [history, setHistory] = useState<InstallationHistory | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [reload, setReload] = useState(0)
  const [deleteRecord, setDeleteRecord] = useState<Job | 'all' | null>(null)
  const [deletionError, setDeletionError] = useState('')
  const previousIds = useRef(new Set<string>())
  const action = useAction(props.notify, props.refresh)
  const unfinishedIds = unfinished.map(job => job.id).join('|')

  useEffect(() => {
    const ids = new Set(unfinished.map(job => job.id))
    if ([...ids].some(id => !previousIds.current.has(id))) setTab('unfinished')
    else if (!ids.size && previousIds.current.size) { setTab('history'); setPage(1); setReload(value => value + 1) }
    previousIds.current = ids
  }, [unfinishedIds])

  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    let inFlight = false
    setLoading(true)
    async function load() {
      if (inFlight) return
      inFlight = true
      try {
        const response = await apiFetch(`/api/installations/history?page=${page}`, { signal: controller.signal })
        const result = await response.json() as InstallationHistory
        if (controller.signal.aborted) return
        setHistory(result); setError('')
        const last = Math.max(1, Math.ceil(result.total / result.page_size))
        if (page > last) setPage(last)
      } catch (cause) { if (!controller.signal.aborted) setError(errorMessage(cause)) }
      finally { inFlight = false; if (!controller.signal.aborted) setLoading(false) }
    }
    void load()
    const timer = window.setInterval(() => void load(), 3000)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [active, page, reload])

  async function control(job: Job, operation: 'pause' | 'resume' | 'cancel') {
    await action.run(() => request('POST', `/api/jobs/${job.id}/${operation}`), operation === 'pause' ? '已请求暂停环境包下载' : operation === 'resume' ? '已继续环境包下载' : job.operation === 'prepare_startup_prefetch' ? '已请求停止启动预取准备' : '已请求停止安装')
  }

  async function continueInstallation(job: Job) {
    if (!job.manifest) return
    await action.run(() => request('POST', `/api/jobs/${job.id}/confirm`, { stage_id: job.manifest!.id }), '已继续安装')
  }

  async function deleteHistory() {
    if (!deleteRecord) return
    setDeletionError('')
    const result = await action.run(async () => {
      try { return await request('DELETE', deleteRecord === 'all' ? '/api/installations/history' : `/api/jobs/${deleteRecord.id}`) }
      catch (cause) { setDeletionError(errorMessage(cause)); throw cause }
    }, deleteRecord === 'all' ? '已清空结束的安装记录' : '安装记录已删除')
    if (result.ok) { setDeleteRecord(null); setReload(value => value + 1) }
  }

  const visibleHistory = history?.page === page ? history.items : []
  const pageCount = Math.max(1, Math.ceil((history?.total || 0) / (history?.page_size || 10)))
  return <div className="panel installation-panel runtime-installations"><div className="panel-heading"><div><h3>环境安装任务</h3></div><div className="runtime-job-tabs" role="tablist" aria-label="环境安装任务"><button role="tab" aria-selected={tab === 'unfinished'} onClick={() => setTab('unfinished')}>未完成 <span>{unfinished.length}</span></button><button role="tab" aria-selected={tab === 'history'} onClick={() => setTab('history')}>历史记录{history && <span>{history.total}</span>}</button></div></div>
    {tab === 'unfinished' ? <div className="runtime-installation-list">{unfinished.length ? unfinished.map(job => <InstallationRow key={job.id} job={job} props={props} busy={action.busy} onControl={operation => void control(job, operation)} onConfirm={() => void continueInstallation(job)} onRetry={() => onRetry(job)} onSwitch={onSwitch} onDelete={() => {}} />) : <p className="runtime-task-empty">暂无未完成的安装任务</p>}</div> : <div className="runtime-installation-list"><div className="runtime-history-toolbar"><span>10 条 / 页</span><Button small icon={Trash2} variant="ghost" disabled={!history?.total || action.busy} onClick={() => { setDeletionError(''); setDeleteRecord('all') }}>清空历史</Button></div>{error && <ErrorNotice message={error} retry={() => setReload(value => value + 1)} />}{loading && !visibleHistory.length ? <p className="runtime-task-empty" role="status" aria-label="正在读取历史记录"><Loader2 size={16} className="spin" /></p> : visibleHistory.length ? visibleHistory.map(job => <InstallationRow key={job.id} job={job} props={props} busy={action.busy} history onControl={operation => void control(job, operation)} onConfirm={() => void continueInstallation(job)} onRetry={() => onRetry(job)} onSwitch={onSwitch} onDelete={() => { setDeletionError(''); setDeleteRecord(job) }} />) : !error && <p className="runtime-task-empty">暂无安装历史</p>}<div className="runtime-history-pagination"><IconButton icon={ChevronLeft} label="上一页" disabled={page <= 1 || loading} onClick={() => setPage(value => value - 1)} /><span>{page} / {pageCount}</span><IconButton icon={ChevronRight} label="下一页" disabled={page >= pageCount || loading} onClick={() => setPage(value => value + 1)} /></div></div>}
    {deleteRecord && <Modal title={deleteRecord === 'all' ? '清空安装历史？' : '删除这条安装记录？'} dismissible={!action.busy} onClose={() => setDeleteRecord(null)} footer={<><Button disabled={action.busy} onClick={() => setDeleteRecord(null)}>取消</Button><Button icon={Trash2} variant="danger" busy={action.busy} onClick={() => void deleteHistory()}>{deleteRecord === 'all' ? '清空已结束记录' : '删除记录'}</Button></>}><p className="runtime-subtle">仅删除已结束记录，运行环境与下载文件保留。</p>{deleteRecord !== 'all' && <p className="runtime-subtle">{deleteRecord.name || deleteRecord.backend || '环境安装'} · {timeLabel(deleteRecord.finished_at || deleteRecord.created_at)}</p>}{deletionError && <ErrorNotice message={deletionError} />}</Modal>}
  </div>
}
