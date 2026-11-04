import { useEffect, useRef, useState } from 'react'
import { Download, FolderOpen } from 'lucide-react'
import { apiFetch, errorMessage, pickPath, request } from './api'
import { Badge, Button, ErrorNotice, Field, IconButton, Modal, Select } from './components'
import { useAction } from './hooks'
import type { Backend, PageProps, RuntimeInstallation, RuntimeInstallPlan, RuntimePlatform, RuntimeVersion, RuntimeVersionPage } from './types'
import { backendPlatforms, runtimePlatform } from './platforms'
import { runtimeLabel } from './runtime-presentation'

export interface RuntimeInstallSelection {
  backend: Backend
  platform?: RuntimePlatform
  source?: RuntimeInstallation
  options?: Record<string, unknown>
  version?: string
  retryOf?: string
}

export default function RuntimeInstallDialog({ backend, platform, source, options: initialOptions = {}, version: initialVersion, retryOf, onClose, ...props }: PageProps & RuntimeInstallSelection & { onClose: () => void }) {
  const selectedPlatform = platform || (source ? runtimePlatform(source, props.state.backends) : backendPlatforms(backend)[0])
  const [versions, setVersions] = useState<RuntimeVersion[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [version, setVersion] = useState(initialVersion || String(initialOptions.version || ''))
  const [variant, setVariant] = useState(String(initialOptions.variant || initialOptions.flavor || ''))
  const [python, setPython] = useState(String(initialOptions.python || ''))
  const [startupPrefetch, setStartupPrefetch] = useState(typeof initialOptions.startup_expert_prefetch === 'boolean' ? initialOptions.startup_expert_prefetch : !!(source?.startup_expert_prefetch?.available || source?.metadata.startup_expert_prefetch))
  const [catalogBusy, setCatalogBusy] = useState(false)
  const [catalogError, setCatalogError] = useState('')
  const [planning, setPlanning] = useState(false)
  const [planError, setPlanError] = useState('')
  const catalogAbort = useRef<AbortController | null>(null)
  const catalogSequence = useRef(0)
  const planAbort = useRef<AbortController | null>(null)
  const planSequence = useRef(0)
  const action = useAction(props.notify, props.refresh)
  const selected = versions.find(item => item.version === version)
  const validVariant = selected?.variants.some(item => item.id === variant)
  const canPlan = !!selected && selected.status !== 'unavailable' && !!validVariant
  const installLabel = planning && !action.busy ? '准备安装清单…' : action.busy ? (source ? '开始更新…' : '开始安装…') : source ? '开始更新' : '开始安装'

  async function loadVersions(cursor: string | null = null, append = false) {
    catalogAbort.current?.abort()
    const controller = new AbortController()
    catalogAbort.current = controller
    const sequence = ++catalogSequence.current
    setCatalogBusy(true); setCatalogError('')
    try {
      const query = new URLSearchParams({ backend: backend.id })
      if (cursor) query.set('cursor', cursor)
      const response = await apiFetch(`/api/runtimes/versions?${query}`, { signal: controller.signal })
      const result = await response.json() as RuntimeVersionPage
      if (controller.signal.aborted || sequence !== catalogSequence.current) return
      setVersions(previous => append ? [...previous, ...result.versions.filter(item => !previous.some(existing => existing.version === item.version))] : result.versions)
      setNextCursor(result.next_cursor)
    } catch (error) {
      if (!controller.signal.aborted && sequence === catalogSequence.current) setCatalogError(errorMessage(error))
    } finally { if (!controller.signal.aborted && sequence === catalogSequence.current) setCatalogBusy(false) }
  }

  useEffect(() => {
    void loadVersions()
    return () => { catalogAbort.current?.abort(); planAbort.current?.abort(); ++catalogSequence.current; ++planSequence.current }
  }, [backend.id])

  function clearPlan() {
    planAbort.current?.abort(); planAbort.current = null; ++planSequence.current
    setPlanError(''); setPlanning(false)
  }

  function close() {
    clearPlan()
    onClose()
  }

  async function install() {
    if (!canPlan || action.busy || planAbort.current) return
    const controller = new AbortController()
    planAbort.current = controller
    const sequence = ++planSequence.current
    setPlanning(true); setPlanError('')
    try {
      const options = { version, variant, ...(backend.id === 'strata' ? { startup_expert_prefetch: startupPrefetch } : {}), ...(backend.python_option && python.trim() ? { python: python.trim() } : {}) }
      const response = await apiFetch('/api/runtimes/plan', { method: 'POST', signal: controller.signal, body: JSON.stringify({ backend: backend.id, options, ...(source ? { update_from: source.id } : {}), ...(retryOf ? { retry_of: retryOf } : {}) }) })
      const plan = await response.json() as RuntimeInstallPlan
      if (controller.signal.aborted || sequence !== planSequence.current) return
      const result = await action.run(() => request('POST', '/api/runtimes/install', { plan_id: plan.id }), source ? '更新已开始，原环境保留' : '环境安装已开始')
      if (result.ok) onClose()
    } catch (error) { if (!controller.signal.aborted && sequence === planSequence.current) setPlanError(errorMessage(error)) }
    finally { if (!controller.signal.aborted && sequence === planSequence.current) { planAbort.current = null; setPlanning(false) } }
  }

  return <Modal title={`${source ? '更新' : '安装'} ${backend.label}`} description={source ? '新建独立环境，完成后可切换加载配置；原环境保留。' : undefined} wide dismissible={!action.busy} onClose={close} footer={<><Button onClick={close} disabled={action.busy}>取消</Button><Button icon={Download} variant="primary" busy={action.busy || planning} disabled={!canPlan} onClick={() => void install()}>{installLabel}</Button></>}>
    {source && <div className="runtime-source-note"><span>当前环境</span><strong>{source.backend === 'strata' ? runtimeLabel(source) : source.name}</strong>{source.backend !== 'strata' && <Badge>{source.version || '版本未知'}</Badge>}</div>}
    <div className="runtime-install-fields"><Field label="官方版本"><Select disabled={action.busy} value={version} onChange={event => { setVersion(event.target.value); setVariant(''); clearPlan() }}><option value="">{catalogBusy && !versions.length ? '读取版本…' : '选择版本'}</option>{version && !selected && <option value={version} disabled>{version}（当前目录未提供）</option>}{versions.map(item => <option key={item.version} value={item.version} disabled={item.status === 'unavailable'}>{item.label}{item.recommended ? ' · 推荐' : ''}{item.status === 'unverified' ? ' · 未验证' : item.status === 'unavailable' ? ' · 不可安装' : ''}</option>)}</Select></Field><Field label="运行类型"><Select disabled={!selected || selected.status === 'unavailable' || action.busy} value={variant} onChange={event => { setVariant(event.target.value); clearPlan() }}><option value="">选择类型</option>{selected?.variants.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</Select></Field></div>
    {(nextCursor || selected) && <div className="runtime-catalog-meta">{nextCursor && <Button small variant="ghost" busy={catalogBusy} disabled={action.busy} onClick={() => void loadVersions(nextCursor, true)}>加载更多版本</Button>}{selected && <Badge tone={selected.status === 'available' ? 'green' : selected.status === 'unverified' ? 'amber' : 'red'}>{selected.status === 'available' ? '可安装' : selected.status === 'unverified' ? '兼容性未验证' : '当前不可安装'}</Badge>}</div>}
    {catalogError && <ErrorNotice message={catalogError} retry={() => void loadVersions(versions.length ? nextCursor : null, versions.length > 0)} />}
    {!catalogBusy && !catalogError && !versions.length && <p className="runtime-subtle">暂无可用版本。</p>}
    {selected?.reason && <p className={selected.status === 'available' ? 'runtime-subtle' : 'warning-note'}>{selected.reason}</p>}
    {backend.id === 'strata' && <label className="runtime-install-prefetch check-option"><input type="checkbox" checked={startupPrefetch} disabled={action.busy} onChange={event => { setStartupPrefetch(event.target.checked); clearPlan() }} /><strong>准备启动预取</strong></label>}
    {backend.python_option && <Field label="用于创建环境的 Python" help="留空使用应用默认 Python。"><div className="input-with-button"><input value={python} disabled={action.busy} onChange={event => { setPython(event.target.value); clearPlan() }} placeholder="python.exe 路径" /><IconButton icon={FolderOpen} label="浏览 Python 路径" disabled={action.busy} onClick={() => { clearPlan(); void action.run(() => pickPath('file')).then(result => { if (result.ok && result.data) setPython(result.data) }) }} /></div></Field>}
    {selectedPlatform === 'wsl2' && <p className="runtime-subtle">使用共享的 ModLoader 发行版，各环境目录独立。</p>}
    {planError && <ErrorNotice message={planError} retry={() => void install()} />}
  </Modal>
}
