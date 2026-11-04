import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { ArrowUp, File, Folder, FolderOpen, RefreshCw } from 'lucide-react'
import { apiFetch, errorMessage, pickPath, request } from './api'
import { Badge, Button, ErrorNotice, Field, Modal, Select, formatBytes } from './components'
import type { ModelLocation, RuntimeInstallation, WslCheck, WslDirectory, WslDistributions } from './types'
import './wsl-platform.css'

export function useWslDistributions(active = true) {
  const [data, setData] = useState<WslDistributions | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    if (!active) return
    const controller = new AbortController()
    setBusy(true); setError('')
    void apiFetch('/api/platforms/wsl', { signal: controller.signal }).then(response => response.json()).then((result: WslDistributions) => {
      if (!controller.signal.aborted) setData(result)
    }).catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause)) })
      .finally(() => { if (!controller.signal.aborted) setBusy(false) })
    return () => controller.abort()
  }, [active, revision])
  return { data, busy, error, reload: () => setRevision(value => value + 1) }
}

export function WslDistributionField({ value, onChange, disabled }: { value: string; onChange: (value: string) => void; disabled?: boolean }) {
  const list = useWslDistributions()
  return <>
    <Field label="WSL 发行版"><Select value={value} disabled={disabled || list.busy} onChange={event => onChange(event.target.value)}>
      <option value="">{list.busy ? '正在读取发行版…' : '选择 WSL2 发行版'}</option>
      {value && !list.data?.distributions.some(item => item.name === value) && <option value={value}>{value}（当前未列出）</option>}
      {list.data?.distributions.map(item => <option key={item.name} value={item.name} disabled={item.version !== 2}>{item.name} · {item.managed ? 'ModLoader 托管' : '用户发行版'}{item.version !== 2 ? ' · 非 WSL2' : ''}</option>)}
    </Select></Field>
    {list.error && <ErrorNotice message={list.error} retry={list.reload} />}
    {!list.busy && list.data && !list.data.distributions.some(item => item.version === 2) && <p className="field-help">未发现 WSL2 发行版。请先在 Windows 中启用 WSL2 并准备发行版。</p>}
  </>
}

export function WslDistributionStatus({ active, runtimes }: { active: boolean; runtimes: RuntimeInstallation[] }) {
  const list = useWslDistributions(active)
  const [checking, setChecking] = useState('')
  const [checks, setChecks] = useState<Record<string, WslCheck | undefined>>({})
  const [checkErrors, setCheckErrors] = useState<Record<string, string>>({})
  async function check(distribution: string) {
    setChecking(distribution); setCheckErrors(previous => ({ ...previous, [distribution]: '' }))
    setChecks(previous => ({ ...previous, [distribution]: undefined }))
    try { const result = await request<WslCheck>('POST', '/api/platforms/wsl/check', { distribution }); setChecks(previous => ({ ...previous, [distribution]: result })); list.reload() }
    catch (cause) { setCheckErrors(previous => ({ ...previous, [distribution]: errorMessage(cause) })) }
    finally { setChecking('') }
  }
  const names = new Set(list.data?.distributions.map(item => item.name))
  const missing = [...new Set(runtimes.map(runtime => String(runtime.metadata.distribution || 'ModLoader')).filter(name => !names.has(name)))]
  return <div className="wsl-distributions">
    <div className="wsl-distribution-heading"><span>发行版</span><Button small variant="ghost" icon={RefreshCw} disabled={list.busy || !!checking} onClick={list.reload}>刷新列表</Button></div>
    {list.error && <ErrorNotice message={list.error} retry={list.reload} />}
    {list.busy && !list.data && <p className="field-help" role="status">正在读取发行版列表…</p>}
    {list.data && !list.data.available && <p className="field-help">WSL 当前不可用。请先在 Windows 中启用 WSL2。</p>}
    {list.data?.available && !list.data.distributions.length && <p className="field-help">尚无已登记的 WSL 发行版。</p>}
    {list.data?.distributions.map(distribution => {
      const result = checks[distribution.name]
      const running = distribution.state.toLowerCase() === 'running'
      return <div className="wsl-distribution" key={distribution.name} data-distribution={distribution.name}>
        <div className="wsl-distribution-summary"><strong>{distribution.name}</strong><Badge tone={running ? 'green' : 'neutral'}>{running ? '运行中' : distribution.state.toLowerCase() === 'stopped' ? '已停止' : distribution.state}</Badge><Badge>WSL {distribution.version}</Badge><span>{distribution.managed ? 'ModLoader 托管' : '用户发行版 · 仅登记外部环境'}</span><Button small variant="ghost" disabled={!!checking || distribution.version !== 2} busy={checking === distribution.name} title="检查会启动所选发行版，读取 Linux、Python 与 GPU 信息" onClick={() => void check(distribution.name)}>检查发行版</Button></div>
        {checkErrors[distribution.name] && <ErrorNotice message={checkErrors[distribution.name]} />}
        {result && <div className="wsl-check-result" role="status"><Badge tone={result.ok ? 'green' : 'amber'}>{result.ok ? '检查通过' : '检查未通过'}</Badge>{result.python && <span>{result.python}</span>}{result.gpu && <span>GPU：{result.gpu}</span>}{result.errors.map((error, index) => <p className="error-text" key={index}>{error}</p>)}</div>}
      </div>
    })}
    {list.data && !list.error && missing.map(name => <p className="warning-note" key={name}>{name} 当前未列出；相关运行环境登记仍保留。</p>)}
    <p className="wsl-list-note">列表不会启动发行版；检查和浏览会启动所选发行版。</p>
  </div>
}

export function PlatformPathInput({ location, kind, value, onChange, disabled, placeholder, label, autoFocus, allowDirectories }: {
  location?: ModelLocation; kind: 'file' | 'directory'; value: string; onChange: (value: string) => void
  disabled?: boolean; placeholder?: string; label?: string; autoFocus?: boolean
  allowDirectories?: boolean
}) {
  return <div className="input-with-button"><input aria-label={label} autoFocus={autoFocus} value={value} disabled={disabled} placeholder={placeholder} onChange={event => onChange(event.target.value)} /><PlatformBrowseButton location={location} kind={kind} value={value} onChange={onChange} disabled={disabled} allowDirectories={allowDirectories} /></div>
}

export function PlatformBrowseButton({ location, kind, value, onChange, disabled, allowDirectories }: {
  location?: ModelLocation; kind: 'file' | 'directory'; value: string; onChange: (value: string) => void; disabled?: boolean
  allowDirectories?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [error, setError] = useState('')
  const wsl = location?.platform === 'wsl2'
  async function browse() {
    setError('')
    if (wsl) { setOpen(true); return }
    try { const result = await pickPath(kind); if (result) onChange(result) }
    catch (cause) { setError(errorMessage(cause)) }
  }
  return <><Button icon={FolderOpen} disabled={disabled || wsl && !location.distribution} onClick={() => void browse()}>浏览</Button>
    {error && <ErrorNotice message={error} />}
    {open && location?.distribution && createPortal(<WslPathBrowser distribution={location.distribution} value={value} kind={kind} allowDirectories={allowDirectories} onClose={() => setOpen(false)} onSelect={path => { onChange(path); setOpen(false) }} />, document.body)}
  </>
}

function WslPathBrowser({ distribution, value, kind, allowDirectories, onClose, onSelect }: { distribution: string; value: string; kind: 'file' | 'directory'; allowDirectories?: boolean; onClose: () => void; onSelect: (path: string) => void }) {
  const initialPath = value.startsWith('/') ? kind === 'directory' ? value : value.slice(0, value.lastIndexOf('/')) || '/' : '/'
  const [path, setPath] = useState(initialPath)
  const [requestedPath, setRequestedPath] = useState(initialPath)
  const [result, setResult] = useState<WslDirectory | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(true)
  const [revision, setRevision] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setBusy(true); setError(''); setResult(null)
    void apiFetch('/api/platforms/wsl/browse', { method: 'POST', body: JSON.stringify({ distribution, path: requestedPath }), signal: controller.signal }).then(response => response.json()).then((data: WslDirectory) => {
      if (!controller.signal.aborted) { setResult(data); setPath(data.path) }
    }).catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause)) })
      .finally(() => { if (!controller.signal.aborted) setBusy(false) })
    return () => controller.abort()
  }, [distribution, requestedPath, revision])
  function navigate(path: string) { setRequestedPath(path); setRevision(value => value + 1) }
  return <Modal wide title={`浏览 Linux ${kind === 'directory' ? '目录' : allowDirectories ? '文件或目录' : '文件'}`} description={distribution} className="wsl-path-browser" onClose={onClose} footer={<><Button onClick={onClose}>取消</Button>{(kind === 'directory' || allowDirectories) && <Button variant="primary" disabled={busy || !result} onClick={() => result && onSelect(result.path)}>选择此目录</Button>}</>}>
    <Field label="Linux 目录"><div className="input-with-button"><input autoFocus aria-label="Linux 目录" className="mono" value={path} onChange={event => setPath(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && path.startsWith('/')) navigate(path) }} /><Button disabled={!path.startsWith('/') || busy} onClick={() => navigate(path)}>前往</Button></div></Field>
    <div className="wsl-browser-toolbar"><Button small icon={ArrowUp} disabled={busy || !result?.parent || result.parent === result.path} onClick={() => result?.parent && navigate(result.parent)}>上级目录</Button><span>{result ? `${result.entries.length} 项` : busy ? '正在读取…' : ''}</span></div>
    {error && <ErrorNotice message={error} retry={() => navigate(requestedPath)} />}
    <div className="wsl-browser-entries" role="region" aria-label="Linux 目录内容" aria-busy={busy}>{result?.entries.map(entry => <button type="button" key={entry.path} disabled={kind === 'directory' && entry.type === 'file'} onClick={() => entry.type === 'directory' ? navigate(entry.path) : onSelect(entry.path)}>
      {entry.type === 'directory' ? <Folder size={17} /> : <File size={17} />}<span>{entry.name}</span><small>{entry.type === 'directory' ? '目录' : entry.size_bytes == null ? '大小未统计' : formatBytes(entry.size_bytes)}</small>
    </button>)}{result && !result.entries.length && <p className="field-help">此目录为空。</p>}</div>
  </Modal>
}
