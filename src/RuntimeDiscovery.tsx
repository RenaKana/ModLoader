import { useEffect, useRef, useState } from 'react'
import { Check, FolderOpen, Plus, RefreshCw, ScanLine } from 'lucide-react'
import type { PageProps, RuntimeInstallation } from './types'
import { errorMessage, pickPath, requestProgress } from './api'
import { Badge, Button, Empty, ErrorNotice, IconButton, Modal } from './components'
import { useAction } from './hooks'
import { runtimePlatform, sameRuntimePath } from './platforms'
import { runtimeDisplayName } from './runtime-presentation'
import OperationStatus, { type OperationProgress } from './OperationStatus'
import './runtime-discovery.css'

type Discovery = {
  runtimes: (RuntimeInstallation & { registered?: boolean })[]
  searched: string[]
  errors: { path: string; error: string }[]
  notes: string[]
}

export default function RuntimeDiscovery({ onClose, ...props }: PageProps & { onClose: () => void }) {
  const [folder, setFolder] = useState('')
  const [revision, setRevision] = useState(0)
  const [result, setResult] = useState<Discovery | null>(null)
  const [busy, setBusy] = useState(true)
  const [cancelled, setCancelled] = useState(false)
  const [error, setError] = useState('')
  const [importingId, setImportingId] = useState<string | null>(null)
  const [progress, setProgress] = useState<OperationProgress | null>(null)
  const [startedAt, setStartedAt] = useState(Date.now())
  const [importProgress, setImportProgress] = useState<OperationProgress | null>(null)
  const [importStartedAt, setImportStartedAt] = useState(Date.now())
  const controller = useRef<AbortController | null>(null)
  const action = useAction(props.notify, props.refresh)
  useEffect(() => {
    const abort = new AbortController()
    controller.current = abort
    let mounted = true
    setBusy(true); setCancelled(false); setResult(null); setError(''); setProgress(null); setStartedAt(Date.now())
    void requestProgress<Discovery>('GET', `/api/runtimes/detect${folder ? `?path=${encodeURIComponent(folder)}` : ''}`, undefined, { signal: abort.signal, onProgress: value => { if (mounted && !abort.signal.aborted) setProgress(value) } })
      .then(data => { if (mounted && !abort.signal.aborted) setResult(data) })
      .catch(reason => { if (mounted && !abort.signal.aborted) setError(errorMessage(reason)) })
      .finally(() => { if (mounted) setBusy(false) })
    return () => { mounted = false; abort.abort() }
  }, [folder, revision])
  async function chooseFolder() {
    try {
      const selected = await pickPath('directory')
      if (selected) { setFolder(selected); setRevision(value => value + 1) }
    } catch (reason) { setError(errorMessage(reason)) }
  }
  function cancel() { controller.current?.abort(); setBusy(false); setCancelled(true) }
  async function importCandidate(runtime: RuntimeInstallation) {
    setImportingId(runtime.id); setImportProgress(null); setImportStartedAt(Date.now())
    try { await action.run(() => requestProgress('POST', '/api/runtimes/import', { backend: runtime.backend, platform: runtimePlatform(runtime, props.state.backends), path: runtime.path, metadata: { ...runtime.metadata, version: runtime.version } }, { onProgress: setImportProgress })) }
    finally { setImportingId(null) }
  }
  return <Modal wide title="自动搜索运行环境" onClose={onClose} dismissible={!action.busy}
    footer={<>{busy && <Button onClick={cancel}>取消搜索</Button>}<Button disabled={action.busy} onClick={onClose}>关闭</Button></>}>
    <div className="runtime-search-toolbar"><div><strong>{folder ? '指定文件夹' : '常见安装位置'}</strong><span title={folder || undefined}>{folder || '系统路径、工具目录、Python 环境与应用托管目录'}</span></div>
      <IconButton icon={FolderOpen} label="选择文件夹搜索" disabled={busy || action.busy} onClick={() => void chooseFolder()} /><IconButton icon={RefreshCw} label="重新搜索" disabled={busy || action.busy} onClick={() => setRevision(value => value + 1)} />
    </div>
    {folder && <Button small variant="ghost" disabled={busy || action.busy} onClick={() => setFolder('')}>返回常见位置</Button>}
    {error && <ErrorNotice message={error} retry={() => setRevision(value => value + 1)} />}
    {busy ? <div className="runtime-search-status" role="status" aria-label="正在查找运行环境"><OperationStatus progress={progress} startedAt={startedAt} fallback="查找运行环境…" /></div>
      : cancelled ? <div className="runtime-search-status" role="status">搜索已取消</div>
      : result && <>
        <div className="runtime-search-count"><span>找到 {result.runtimes.length} 个环境</span></div>
        <div className="detected-list">{result.runtimes.length ? result.runtimes.map(runtime => {
          const imported = runtime.registered || props.state.runtimes.some(item => item.backend === runtime.backend && sameRuntimePath(item, runtime, props.state.backends))
          const label = props.state.backends.find(backend => backend.id === runtime.backend)?.label || runtime.backend
          const displayName = runtimeDisplayName(runtime)
          const name = displayName.startsWith(`${label} ·`) ? label : displayName
          return <div className="detected-row runtime-search-candidate" key={runtime.id}><div><div className="runtime-search-name"><strong>{name}</strong>{name !== label && displayName === runtime.name && <Badge>{label}</Badge>}{displayName === runtime.name && <Badge tone={imported ? 'green' : 'neutral'}>{runtime.version === 'unknown' ? '版本待验证' : runtime.version}</Badge>}</div><p className="mono">{runtime.path}</p>{typeof runtime.metadata.discovery_source === 'string' && <span className="runtime-search-source">{runtime.metadata.discovery_source}</span>}{runtime.id === importingId && <OperationStatus progress={importProgress} startedAt={importStartedAt} fallback="检查运行入口与版本…" />}</div>
            <Button small icon={imported ? Check : Plus} disabled={imported || action.busy} busy={runtime.id === importingId} onClick={() => void importCandidate(runtime)}>{runtime.id === importingId ? '验证并导入…' : imported ? '已导入' : '导入'}</Button></div>
        }) : <Empty icon={ScanLine} title="没有找到可导入的环境" detail="也可通过“导入运行环境”指定路径。" />}</div>
        {!!result.errors?.length && <details className="runtime-search-details" open><summary>有 {result.errors.length} 个位置无法读取</summary>{result.errors.map((item, index) => <p key={index}><span className="mono">{item.path}</span><span className="error-text">{item.error}</span></p>)}</details>}
        {!!result.searched?.length && <details className="runtime-search-details"><summary>已搜索的位置（{result.searched.length}）</summary>{result.searched.map(path => <p className="mono" key={path}>{path}</p>)}</details>}
        {!!result.notes?.length && <details className="runtime-search-details"><summary>搜索说明</summary>{result.notes.map((note, index) => <p key={index}>{note}</p>)}</details>}
      </>}
  </Modal>
}
