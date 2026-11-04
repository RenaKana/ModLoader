import { useEffect, useRef, useState } from 'react'
import { ArrowRightLeft, Trash2 } from 'lucide-react'
import { errorMessage, isActive, request } from './api'
import { Badge, Button, ErrorNotice, Field, Modal, Select } from './components'
import { runtimeDisplayName, runtimeOptionLabel } from './runtime-presentation'
import { useAction } from './hooks'
import type { LoadProfile, PageProps, RuntimeInstallation, RuntimeSwitchPreview } from './types'
import { modelSupportsRuntime, runtimePlatform } from './platforms'

function switchSelectionKey(targetId: string, profiles: LoadProfile[], target?: RuntimeInstallation) {
  return JSON.stringify({ targetId, profiles: profiles.map(profile => [profile.id, profile.revision]),
    target: target && [target.path, target.version, target.startup_expert_prefetch] })
}

function PrefetchSwitchNotice({ preview }: { preview: RuntimeSwitchPreview }) {
  const prefetch = preview.startup_expert_prefetch
  if (!prefetch || prefetch.status === 'unchanged') return null
  return <p className={prefetch.status === 'choice_required' ? 'warning-note' : 'runtime-subtle'}>{prefetch.message || ({ preserved: '启动预取将保留', choice_required: '目标环境缺少启动预取', disabled: '将关闭启动预取' })[prefetch.status]}</p>
}

export function RemoveRuntimeDialog({ runtime, onClose, ...props }: PageProps & { runtime: RuntimeInstallation; onClose: () => void }) {
  const [deleteFiles, setDeleteFiles] = useState(false)
  const [replacement, setReplacement] = useState('')
  const [error, setError] = useState('')
  const [migration, setMigration] = useState<{ key: string; value: RuntimeSwitchPreview } | null>(null)
  const [checkingMigration, setCheckingMigration] = useState(false)
  const [disableFor, setDisableFor] = useState<string | null>(null)
  const action = useAction(props.notify, props.refresh)
  const references = props.state.profiles.filter(profile => profile.runtime_id === runtime.id)
  const instances = props.state.instances.filter(instance => instance.runtime_id === runtime.id && isActive(instance.state))
  const replacements = props.state.runtimes.filter(item => item.id !== runtime.id && item.backend === runtime.backend && references.every(profile => { const model = props.state.models.find(model => model.id === profile.model_id); return !model || modelSupportsRuntime(model, item, props.state.backends) }))
  const installRoot = typeof runtime.metadata.install_root === 'string' ? runtime.metadata.install_root : runtime.path
  const wsl = runtimePlatform(runtime, props.state.backends) === 'wsl2'
  const target = replacements.find(item => item.id === replacement)
  const migrationKey = switchSelectionKey(replacement, references, target)
  const migrationPreview = migration?.key === migrationKey ? migration.value : null
  const checkPrefetch = runtime.backend === 'strata' && references.length > 0
  const needsChoice = migrationPreview?.startup_expert_prefetch?.status === 'choice_required'
  const disablePrefetch = disableFor === migrationKey
  const allowed = !instances.length && (!references.length || !!target && (!checkPrefetch || !!migrationPreview && (!needsChoice || disablePrefetch)))

  useEffect(() => {
    if (!checkPrefetch || !target) { setCheckingMigration(false); return }
    const controller = new AbortController()
    setCheckingMigration(true); setError('')
    void request<RuntimeSwitchPreview>('POST', `/api/runtimes/${runtime.id}/switch-preview`, { target_id: replacement, profile_ids: references.map(profile => profile.id) }, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setMigration({ key: migrationKey, value }) })
      .catch(cause => { if (!controller.signal.aborted) setError(errorMessage(cause)) })
      .finally(() => { if (!controller.signal.aborted) setCheckingMigration(false) })
    return () => controller.abort()
  }, [checkPrefetch, migrationKey, runtime.id])

  async function remove() {
    setError('')
    const query = new URLSearchParams()
    if (runtime.managed && deleteFiles) query.set('delete_files', 'true')
    if (references.length && replacement) query.set('replacement_id', replacement)
    if (needsChoice && disablePrefetch) query.set('disable_startup_expert_prefetch', 'true')
    const result = await action.run(async () => {
      try { return await request('DELETE', `/api/runtimes/${runtime.id}${query.size ? `?${query}` : ''}`) }
      catch (cause) { setError(errorMessage(cause)); throw cause }
    }, deleteFiles ? '运行环境已卸载' : '运行环境登记已移除')
    if (result.ok) onClose()
  }

  return <Modal title={`${runtime.managed ? '移除' : '移除登记'} ${runtimeDisplayName(runtime)}？`} description={runtime.managed ? '仅移除登记时保留安装文件。' : '原始文件保留。'} onClose={onClose} dismissible={!action.busy} footer={<><Button onClick={onClose} disabled={action.busy}>取消</Button><Button icon={Trash2} variant="danger" busy={action.busy} disabled={!allowed} onClick={() => void remove()}>{references.length ? `迁移配置并${deleteFiles ? '卸载' : '移除登记'}` : deleteFiles ? '卸载环境' : '仅移除登记'}</Button></>}>
    <div className="confirmation-path mono">{runtime.managed ? installRoot : runtime.path}</div>
    {runtime.managed && <label className="check-option runtime-remove-files"><input type="checkbox" checked={deleteFiles} disabled={action.busy} onChange={event => { setDeleteFiles(event.target.checked); setError('') }} /><div><strong>同时卸载并删除环境文件</strong><span>{wsl ? '仅删除此环境目录，保留共享的 ModLoader WSL 发行版及其他环境。' : '删除后需重新安装，模型文件不受影响。'}</span></div></label>}
    <div className="runtime-reference-heading"><h4>引用的加载配置</h4><Badge>{references.length}</Badge></div>
    {references.length > 0 && <ul className="runtime-reference-list">{references.map(profile => <li key={profile.id}><strong>{profile.name}</strong><span>{profile.alias}</span></li>)}</ul>}
    {!!references.length && <Field label="迁移到同一后端的其他环境"><Select value={replacement} disabled={action.busy} onChange={event => { setReplacement(event.target.value); setDisableFor(null); setError('') }}><option value="">选择替代环境</option>{replacements.map(item => <option value={item.id} key={item.id}>{runtimeOptionLabel(item, replacements)}</option>)}</Select></Field>}
    {checkingMigration && <p className="runtime-subtle" role="status">检查迁移配置…</p>}
    {migrationPreview && <PrefetchSwitchNotice preview={migrationPreview} />}
    {needsChoice && <label className="check-option"><input type="checkbox" checked={disablePrefetch} disabled={action.busy} onChange={event => setDisableFor(event.target.checked ? migrationKey : null)} /><strong>关闭预取并迁移</strong></label>}
    {!!references.length && !replacements.length && <p className="warning-note">先安装或导入同一后端的其他环境，再迁移并移除此环境。</p>}
    {!!instances.length && <div className="warning-note">请先停止使用此环境的实例：{instances.map(instance => instance.alias).join('、')}。</div>}
    {error && <ErrorNotice message={error} />}
  </Modal>
}

export function SwitchRuntimeDialog({ source, initialTarget, onClose, ...props }: PageProps & { source: RuntimeInstallation; initialTarget?: string; onClose: () => void }) {
  const references = props.state.profiles.filter(profile => profile.runtime_id === source.id)
  const targets = props.state.runtimes.filter(runtime => runtime.id !== source.id && runtime.backend === source.backend)
  const [target, setTarget] = useState(initialTarget || '')
  const [selected, setSelected] = useState(() => references.map(profile => profile.id))
  const [checkedPreview, setCheckedPreview] = useState<{ key: string; value: RuntimeSwitchPreview; disable: boolean } | null>(null)
  const [error, setError] = useState('')
  const action = useAction(props.notify)
  const profiles = references.filter(profile => selected.includes(profile.id))
  const compatibleTargets = targets.filter(runtime => profiles.every(profile => { const model = props.state.models.find(model => model.id === profile.model_id); return !model || modelSupportsRuntime(model, runtime, props.state.backends) }))
  const canSubmit = profiles.length > 0 && compatibleTargets.some(runtime => runtime.id === target)
  const selectionKey = switchSelectionKey(target, profiles, targets.find(runtime => runtime.id === target))
  const latestKey = useRef(selectionKey)
  latestKey.current = selectionKey
  const checked = checkedPreview?.key === selectionKey ? checkedPreview : null
  const preview = checked?.value
  const needsChoice = preview?.startup_expert_prefetch?.status === 'choice_required'
  const disablesPrefetch = needsChoice || checked?.disable

  function changed() { setCheckedPreview(null); setError('') }

  async function submit() {
    setError('')
    const committing = !!preview
    const key = selectionKey
    const body = { target_id: target, profile_ids: profiles.map(profile => profile.id), ...(disablesPrefetch ? { disable_startup_expert_prefetch: true } : {}) }
    const result = await action.run(async () => {
      try {
        if (!committing || needsChoice) {
          const value = await request<RuntimeSwitchPreview>('POST', `/api/runtimes/${source.id}/switch-preview`, body)
          if (latestKey.current !== key) return { committed: false }
          if (!committing || value.startup_expert_prefetch?.status === 'choice_required') return { committed: false, value }
        }
        if (latestKey.current !== key) return { committed: false }
        await request('POST', `/api/runtimes/${source.id}/switch`, body)
        await props.refresh()
        return { committed: true }
      } catch (cause) { setCheckedPreview(null); setError(errorMessage(cause)); throw cause }
    })
    if (!result.ok) return
    if (result.data.committed) { props.notify('所选加载配置已切换，下次加载时使用新环境', 'success'); onClose() }
    else if (result.data.value && latestKey.current === key) setCheckedPreview({ key, value: result.data.value, disable: !!disablesPrefetch })
  }

  return <Modal title={`切换 ${runtimeDisplayName(source)} 的加载配置`} onClose={onClose} dismissible={!action.busy} footer={<><Button onClick={onClose} disabled={action.busy}>{disablesPrefetch ? '保留原环境' : '取消'}</Button><Button icon={ArrowRightLeft} variant="primary" busy={action.busy} disabled={!canSubmit} onClick={() => void submit()}>{preview ? disablesPrefetch ? '关闭预取并切换' : `确认切换 ${profiles.length} 个配置` : '检查切换'}</Button></>}>
    <Field label="目标运行环境"><Select value={target} disabled={action.busy} onChange={event => { setTarget(event.target.value); changed() }}><option value="">选择同一后端的兼容环境</option>{target && !compatibleTargets.some(runtime => runtime.id === target) && <option value={target} disabled>此环境无法访问所选模型</option>}{compatibleTargets.map(runtime => <option value={runtime.id} key={runtime.id}>{runtimeOptionLabel(runtime, compatibleTargets)}</option>)}</Select></Field>
    <div className="runtime-reference-heading"><h4>选择加载配置</h4><span>{profiles.length} / {references.length}</span></div>
    {references.length ? <div className="runtime-profile-options">{references.map(profile => <label className="check-option" key={profile.id}><input type="checkbox" checked={selected.includes(profile.id)} disabled={action.busy} onChange={event => { setSelected(previous => event.target.checked ? [...previous, profile.id] : previous.filter(id => id !== profile.id)); changed() }} /><div><strong>{profile.name}</strong><span>{profile.alias}</span></div></label>)}</div> : <p className="runtime-subtle">没有可切换的加载配置。</p>}
    <p className="runtime-subtle">运行中实例不受影响，下次加载时生效。原环境保留。</p>
    {preview && <div className="runtime-switch-preview"><h4>将切换 {preview.profiles.length} 个加载配置</h4><ul>{preview.profiles.map(profile => <li key={profile.id}>{profile.name}</li>)}</ul><PrefetchSwitchNotice preview={preview} />{preview.warnings?.map((warning, index) => <p className="warning-note" key={index}>{warning}</p>)}</div>}
    {error && <ErrorNotice message={error} />}
  </Modal>
}
