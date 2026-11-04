import { useEffect, useRef, useState } from 'react'
import { ArrowRight, ArrowRightLeft, Cpu, Download, FolderOpen, Pencil, Plus, RefreshCw, ScanLine, Square, Terminal, Trash2 } from 'lucide-react'
import type { Backend, Job, PageProps, RuntimeInstallation, RuntimePlatform } from '../types'
import { errorMessage, isActive, openPath, request, requestProgress } from '../api'
import { Badge, Button, DiagnosticNotice, Empty, Field, IconButton, Modal, PageHeader, Select, StateBadge } from '../components'
import HardwareResources from '../ResourceCards'
import { useAction } from '../hooks'
import { runtimeDisplayName, runtimeLabel } from '../runtime-presentation'
import RuntimeDiscovery from '../RuntimeDiscovery'
import CustomRuntimeDialog from '../CustomRuntimeDialog'
import RuntimeInstallDialog, { type RuntimeInstallSelection } from '../RuntimeInstallDialog'
import RuntimeInstallJobs from '../RuntimeInstallJobs'
import { RemoveRuntimeDialog, SwitchRuntimeDialog } from '../RuntimeManagementDialogs'
import { backendPlatforms, desktopPath, modelSupportsRuntime, platformLabels, runtimeDistribution, runtimeLocation, runtimePlatform } from '../platforms'
import { PlatformPathInput, WslDistributionField, WslDistributionStatus } from '../WslPlatform'
import OperationStatus, { type OperationProgress } from '../OperationStatus'
import RuntimeStartupPrefetch from '../RuntimeStartupPrefetch'
import '../runtime-management.css'

const descriptions: Record<string, { subtitle: string; detail: string; symbol: string }> = {
  llama_cpp: { subtitle: 'GGUF · CPU / CUDA', detail: '轻量运行环境，提供本地 OpenAI 兼容接口。', symbol: 'L' },
  strata: { subtitle: 'GGUF · CUDA · MoE', detail: '面向专家混合模型的本地推理环境。', symbol: 'S' },
  transformers: { subtitle: 'Safetensors · Python / PyTorch', detail: '使用完整模型目录和 Transformers 服务。', symbol: 'T' },
  vllm: { subtitle: 'Safetensors · WSL2 / CUDA', detail: '在独立 WSL2 环境中运行 vLLM 服务。', symbol: 'V' },
}

function importKind(backend?: Backend) {
  return backend?.import_kind || (backend?.id === 'strata' ? 'directory' : backend?.id === 'vllm' ? 'wsl' : 'file')
}

export default function Runtimes({ active = true, ...props }: PageProps & { active?: boolean }) {
  const [importBackend, setImportBackend] = useState('')
  const [importPlatform, setImportPlatform] = useState<RuntimePlatform>('windows')
  const [importOpen, setImportOpen] = useState(false)
  const [customEditor, setCustomEditor] = useState<RuntimeInstallation | null>(null)
  const [installation, setInstallation] = useState<RuntimeInstallSelection | null>(null)
  const [remove, setRemove] = useState<RuntimeInstallation | null>(null)
  const [switching, setSwitching] = useState<{ source: RuntimeInstallation; target?: string } | null>(null)
  const [detectOpen, setDetectOpen] = useState(false)
  const [refreshingId, setRefreshingId] = useState<string | null>(null)
  const [refreshProgress, setRefreshProgress] = useState<OperationProgress | null>(null)
  const [refreshStartedAt, setRefreshStartedAt] = useState(Date.now())
  const refreshController = useRef<AbortController | null>(null)
  const action = useAction(props.notify, props.refresh)
  const busy = action.busy || refreshingId !== null
  useEffect(() => () => refreshController.current?.abort(), [])
  const { hardware } = props.state
  const activeInstances = props.state.instances.filter(instance => isActive(instance.state))

  async function refreshRuntime(runtime: RuntimeInstallation) {
    const controller = new AbortController()
    refreshController.current?.abort(); refreshController.current = controller
    setRefreshingId(runtime.id); setRefreshProgress(null); setRefreshStartedAt(Date.now())
    try {
      await requestProgress('POST', `/api/runtimes/${runtime.id}/refresh`, undefined, { signal: controller.signal, onProgress: value => { if (!controller.signal.aborted) setRefreshProgress(value) } })
      if (controller.signal.aborted) return
      await props.refresh()
      if (!controller.signal.aborted) props.notify(runtime.backend === 'custom' ? '引擎配置检查通过' : '环境版本已刷新', 'success')
    } catch (error) { if (!controller.signal.aborted) props.notify(errorMessage(error), 'error') }
    finally { if (refreshController.current === controller) { refreshController.current = null; setRefreshingId(null) } }
  }

  function retry(job: Job) {
    if (job.operation === 'prepare_startup_prefetch') {
      const runtime = props.state.runtimes.find(item => item.id === job.runtime_id)
      if (!runtime) { props.notify('原运行环境已移除，无法重试准备。', 'error'); return }
      void action.run(() => request('POST', `/api/runtimes/${runtime.id}/prepare-startup-prefetch`), '启动预取准备已开始')
      return
    }
    const backend = props.state.backends.find(item => item.id === (job.backend || job.request?.backend))
    if (!backend || backend.installable === false) { props.notify('此安装记录的后端当前不可安装。', 'error'); return }
    const sourceId = job.update_from || job.request?.update_from
    const source = sourceId ? props.state.runtimes.find(runtime => runtime.id === sourceId && runtime.managed) : undefined
    if (sourceId && !source) props.notify('原环境已移除，本次将作为新环境安装。', 'info')
    setInstallation({ backend, source, options: job.request?.options, version: job.version || job.request?.version, retryOf: job.id })
  }

  return <div className="runtime-management">
    <PageHeader eyebrow="ENGINES & RESOURCES" title="运行环境"><Button icon={ScanLine} onClick={() => setDetectOpen(true)}>自动搜索环境</Button><Button variant="primary" icon={Plus} onClick={() => { setImportPlatform('windows'); setImportBackend(props.state.backends.find(backend => backendPlatforms(backend).includes('windows'))?.id || ''); setImportOpen(true) }}>导入运行环境</Button></PageHeader>
    <HardwareResources hardware={hardware} />
    {hardware.gpu_error != null && <DiagnosticNotice summary="GPU 采样失败" detail={hardware.gpu_error} />}
    {(['windows', 'wsl2'] as const).map(platform => <section className="runtime-platform" key={platform} data-platform={platform} aria-label={platformLabels[platform]}>
    <div className="section-heading"><h3>{platformLabels[platform]}</h3><span>{props.state.runtimes.filter(runtime => runtimePlatform(runtime, props.state.backends) === platform).length} 个环境</span></div>
    {platform === 'wsl2' && <WslDistributionStatus active={active} runtimes={props.state.runtimes.filter(runtime => runtimePlatform(runtime, props.state.backends) === 'wsl2')} />}
    <div className="backend-grid">{props.state.backends.filter(backend => backendPlatforms(backend).includes(platform) || props.state.runtimes.some(runtime => runtime.backend === backend.id && runtimePlatform(runtime, props.state.backends) === platform)).map(backend => {
      const fallback = descriptions[backend.id]
      const runtimes = props.state.runtimes.filter(runtime => runtime.backend === backend.id && runtimePlatform(runtime, props.state.backends) === platform)
      return <div className="panel backend-card" key={backend.id} data-backend-id={backend.id}><div className="backend-heading"><span className={`backend-symbol ${backend.id}`}>{backend.symbol || fallback?.symbol || backend.label.charAt(0)}</span><div><h3>{backend.label}</h3><span>{backend.subtitle || fallback?.subtitle || backend.id}</span></div><Badge dot tone={runtimes.length ? 'green' : 'neutral'}>{runtimes.length ? `${runtimes.length} 个环境` : backend.id === 'custom' ? '未添加' : '未安装'}</Badge><div className="backend-actions">{backend.id !== 'custom' && backendPlatforms(backend).includes(platform) && <Button small icon={FolderOpen} onClick={() => { setImportPlatform(platform); setImportBackend(backend.id); setImportOpen(true) }}>导入已有</Button>}{backend.installable !== false && backendPlatforms(backend).includes(platform) && <Button small icon={Download} onClick={() => setInstallation({ backend, platform })}>{runtimes.length ? '安装新环境' : '安装环境'}</Button>}</div></div>
        {runtimes.length ? <div className="runtime-list">{runtimes.map(runtime => {
          const instances = activeInstances.filter(instance => instance.runtime_id === runtime.id)
          const profiles = props.state.profiles.filter(profile => profile.runtime_id === runtime.id)
          const switchTargets = runtimes.filter(item => item.id !== runtime.id && profiles.every(profile => { const model = props.state.models.find(model => model.id === profile.model_id); return !model || modelSupportsRuntime(model, item, props.state.backends) }))
          return <div className="runtime-row" key={runtime.id} data-runtime-id={runtime.id}><div className="runtime-row-info"><div className="runtime-row-title"><strong title={runtimeDisplayName(runtime)}>{runtimeLabel(runtime)}</strong>{instances.length > 0 && <Badge tone="green">使用中</Badge>}</div><div className="runtime-row-path mono" title={runtime.path}>{runtime.path}</div>
            {refreshingId === runtime.id && <OperationStatus progress={refreshProgress} startedAt={refreshStartedAt} fallback={runtime.backend === 'custom' ? '检查自定义引擎配置…' : '检查运行入口与版本…'} />}
            <div className="runtime-row-meta"><div className="runtime-usage"><span>{runtime.managed ? '应用安装' : '外部导入'}{runtime.backend === 'custom' ? ' · 自定义模板' : ''}</span>{platform === 'wsl2' && <span>{runtimeDistribution(runtime)}</span>}<span title={profiles.map(profile => profile.name).join('、')}>{profiles.length} 个加载配置</span>{instances.length > 0 && <span title={instances.map(instance => instance.alias).join('、')}>{instances.length} 个运行实例</span>}</div>
            <details className="runtime-usage-details"><summary>详情与引用</summary><dl><dt>{runtime.backend === 'strata' ? '引擎版本' : '版本'}</dt><dd>{runtime.version || '未知'}</dd><dt>路径</dt><dd className="mono">{runtime.path}</dd></dl>{profiles.length > 0 && <p>加载配置：{profiles.map(profile => profile.name).join('、')}</p>}{instances.length > 0 && <p>运行实例：{instances.map(instance => instance.alias).join('、')}</p>}</details></div><RuntimeStartupPrefetch {...props} runtime={runtime} /></div>
            <div className="runtime-row-actions">{runtime.backend === 'custom' && <IconButton icon={Pencil} label="编辑模板" disabled={busy || instances.length > 0} onClick={() => setCustomEditor(runtime)} />}<IconButton icon={RefreshCw} label={refreshingId === runtime.id ? '正在检查环境' : runtime.backend === 'custom' ? '检查配置' : '刷新版本'} aria-busy={refreshingId === runtime.id} disabled={busy} onClick={() => void refreshRuntime(runtime)} /><IconButton icon={FolderOpen} label="打开路径" disabled={busy} onClick={() => void action.run(() => openPath(desktopPath(runtime.path, runtimeLocation(runtime, props.state.backends))))} />{runtime.managed && <><Button small variant="ghost" icon={Download} disabled={busy || backend.installable === false} onClick={() => setInstallation({ backend, source: runtime, platform })}>更新环境</Button><Button small variant="ghost" icon={ArrowRightLeft} disabled={busy || !profiles.length || !switchTargets.length} title={!profiles.length ? '没有引用此环境的加载配置' : !switchTargets.length ? '需要同一后端的其他环境' : undefined} onClick={() => setSwitching({ source: runtime })}>切换配置</Button></>}<Button small variant="ghost" icon={Trash2} disabled={busy || instances.length > 0} title={instances.length ? '请先停止使用此环境的运行实例' : undefined} onClick={() => setRemove(runtime)}>{runtime.managed ? '移除 / 卸载' : '移除登记'}</Button></div>
          </div>
        })}</div> : backend.id === 'custom' && <div className="runtime-list"><div className="runtime-not-installed">导入运行环境 → 自定义引擎</div></div>}</div>
    })}</div></section>)}
    {!props.state.backends.length && <div className="panel"><Empty icon={Cpu} title="等待运行后端信息" /></div>}
    <RuntimeInstallJobs {...props} active={active} onRetry={retry} onSwitch={(source, target) => setSwitching({ source, target })} />
    <div className="panel instances-panel"><div className="panel-heading"><div><h3>正在运行 <span className="heading-count">{activeInstances.length}</span></h3></div></div>{activeInstances.length ? <div className="instance-table"><div className="instance-table-head"><span>模型 / API 名称</span><span>后端</span><span>状态</span><span>活动请求</span><span>操作</span></div>{activeInstances.map(instance => <div className="instance-table-row" key={instance.id}><div><strong>{instance.alias}</strong><span className="mono">{instance.platform === 'wsl2' ? `${instance.distribution || 'WSL2'} · Linux PID ${instance.guest_pid || '—'} · 宿主 ` : ''}PID {instance.pid || '—'} · 端口 {instance.port}</span></div><span>{props.state.backends.find(backend => backend.id === instance.backend)?.label || instance.backend}</span><StateBadge state={instance.state} /><span className="mono">{instance.active_requests}</span><div className="table-actions"><IconButton icon={Terminal} label="日志" onClick={() => props.openLogs(instance)} /><Button small icon={Square} onClick={() => void props.stopInstance(instance)} disabled={instance.state === 'stopping'}>停止</Button></div></div>)}</div> : <div className="inline-empty"><Cpu size={18} /><span>暂无模型运行</span><Button small variant="ghost" onClick={() => props.navigate('models')}>前往模型库 <ArrowRight size={13} /></Button></div>}</div>
    {importOpen && <ImportRuntime backend={importBackend} platform={importPlatform} onClose={() => setImportOpen(false)} {...props} />}
    {customEditor && <CustomRuntimeDialog runtime={customEditor} onClose={() => setCustomEditor(null)} {...props} />}
    {installation && <RuntimeInstallDialog {...props} {...installation} onClose={() => setInstallation(null)} />}
    {remove && <RemoveRuntimeDialog runtime={remove} onClose={() => setRemove(null)} {...props} />}
    {switching && <SwitchRuntimeDialog source={switching.source} initialTarget={switching.target} onClose={() => setSwitching(null)} {...props} />}
    {detectOpen && <RuntimeDiscovery {...props} onClose={() => setDetectOpen(false)} />}
  </div>
}

function ImportRuntime({ backend: initialBackend, platform: initialPlatform, onClose, ...props }: PageProps & { backend: string; platform: RuntimePlatform; onClose: () => void }) {
  const [backendId, setBackendId] = useState(initialBackend)
  const [platform, setPlatform] = useState(initialPlatform)
  const [distribution, setDistribution] = useState('')
  const [path, setPath] = useState('')
  const [python, setPython] = useState('')
  const [progress, setProgress] = useState<OperationProgress | null>(null)
  const [startedAt, setStartedAt] = useState(Date.now())
  const action = useAction(props.notify, props.refresh)
  const backend = props.state.backends.find(item => item.id === backendId)
  const kind = importKind(backend)
  const pythonOption = backend?.python_option ?? backendId === 'strata'
  const location = { platform, ...(platform === 'wsl2' ? { distribution } : {}) }
  const fallbackHint = platform === 'wsl2' ? '填写所选发行版中的程序或 Python 路径，不填 wsl.exe。' : kind === 'directory' ? '运行环境所在目录。' : '运行环境可执行文件。'
  function changePlatform(value: RuntimePlatform) {
    setPlatform(value); setPath(''); setPython('')
    if (!backend || !backendPlatforms(backend).includes(value)) setBackendId(props.state.backends.find(item => backendPlatforms(item).includes(value))?.id || '')
  }
  async function submit() {
    const metadata = { ...(pythonOption && python.trim() ? { python: python.trim() } : {}), ...(platform === 'wsl2' ? { distribution } : {}) }
    setProgress(null); setStartedAt(Date.now())
    const result = await action.run(() => requestProgress('POST', '/api/runtimes/import', { backend: backendId, platform, path: path.trim(), metadata }, { onProgress: setProgress }), '运行环境已导入')
    if (result.ok) onClose()
  }
  if (backendId === 'custom') return <CustomRuntimeDialog platform={platform} onPlatformChange={changePlatform} onClose={onClose} onBackendChange={setBackendId} {...props} />
  return <Modal title="导入已有运行环境" dismissible={!action.busy} onClose={onClose} footer={<><Button onClick={onClose} disabled={action.busy}>取消</Button><Button variant="primary" busy={action.busy} disabled={!backend || !path.trim() || platform === 'wsl2' && !distribution} onClick={() => void submit()}>{action.busy ? '验证并导入…' : '导入环境'}</Button></>}>
    <Field label="运行平台"><Select value={platform} disabled={action.busy} onChange={event => changePlatform(event.target.value as RuntimePlatform)}>{Object.entries(platformLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></Field>
    <Field label="运行后端"><Select disabled={action.busy} value={backendId} onChange={event => { setBackendId(event.target.value); setPath(''); setPython('') }}><option value="">选择后端</option>{props.state.backends.filter(item => backendPlatforms(item).includes(platform)).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</Select></Field>
    {platform === 'wsl2' && <WslDistributionField value={distribution} onChange={value => { setDistribution(value); setPath(''); setPython('') }} disabled={action.busy} />}
    <Field label="环境路径" help={platform === 'wsl2' ? fallbackHint : backend?.import_hint || fallbackHint}><PlatformPathInput label="环境路径" autoFocus disabled={action.busy || !backend} location={location} kind={kind === 'directory' ? 'directory' : 'file'} value={path} onChange={setPath} placeholder={platform === 'wsl2' ? '/opt/venv/bin/python' : '本地路径'} /></Field>
    {pythonOption && <Field label="Python 路径" help="留空自动检测。"><PlatformPathInput label="Python 路径" disabled={action.busy} location={location} kind="file" value={python} onChange={setPython} placeholder={platform === 'wsl2' ? '/opt/venv/bin/python' : 'python.exe 路径'} /></Field>}
    {action.busy && <OperationStatus progress={progress} startedAt={startedAt} fallback="检查运行入口与版本…" />}
    {platform === 'wsl2' && <p className="field-help">外部发行版仅登记和检查，文件由用户管理。</p>}
  </Modal>
}
