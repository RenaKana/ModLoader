import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowDownUp, ArrowRight, Braces, Check, Minus, ChevronDown, ChevronRight, Copy, File, FileCode2, FolderOpen, HardDrive, Image, Layers3, ListFilter, LocateFixed, MessageSquare, Pencil, Play, Plus, Search, Square, Terminal, Trash2, X, Zap } from 'lucide-react'
import type { AccelerationStatus, LoadProfile, ModelAsset, ModelLocation, PageProps, ParameterField, PreparationStatus, RuntimePlatform } from '../types'
import { copy, errorMessage, isActive, isJobActive, isReady, openPath, request, requestProgress } from '../api'
import { Badge, Button, Empty, ErrorNotice, Field, IconButton, KeyValue, Modal, PageHeader, StateBadge, DiagnosticNotice, formatBytes } from '../components'
import { Select } from '../components'
import GpuLayersInput, { gpuInputError } from '../GpuLayersInput'
import { runtimeDisplayName, runtimeOptionLabel } from '../runtime-presentation'
import { useAction } from '../hooks'
import '../import.css'
import BulkRemoveModels from '../BulkRemoveModels'
import '../bulk-models.css'
import ResourcePreview from '../ResourcePreview'
import { PreparationDetails, PreparationSummary, useProfilePreflight } from '../ProfilePreparation'
import ProfileAcceleration from '../ProfileAcceleration'
import NumericParameter, { numberError } from '../NumericParameter'
import { generationDefaultText, parameterDefaultText } from '../parameter-defaults'
import { runtimeGenerationFields } from '../custom-engine'
import { backendPlatforms, desktopPath, locationLabel, modelSizeKnown, modelSupportsRuntime, platformLabels, runtimeLocation } from '../platforms'
import { PlatformBrowseButton, PlatformPathInput, WslDistributionField } from '../WslPlatform'
import '../custom-engine.css'
import ModelStorage from '../ModelStorage'
import { gpuKey, gpuLabel, gpuList } from '../resource-metrics'
import '../profile-gpu.css'
import ModelLoadingStatus from '../ModelLoadingStatus'
import OperationStatus, { type OperationProgress } from '../OperationStatus'
import '../chat-model-loading.css'
import { useLongPressReorder } from '../useLongPressReorder'
import { mergeVisibleOrder, orderedItems, readLibraryOrder } from '../library-order'
import LibraryDragPreview from '../LibraryDragPreview'
import RuntimeStartupPrefetch from '../RuntimeStartupPrefetch'
import StartupParameters from '../StartupParameters'
import { useStartupParameters } from '../useStartupParameters'

type FolderImportResult = { root: string; imported: ModelAsset[]; existing: ModelAsset[]; errors: { path: string; error: string }[]; skipped: { path: string; reason: string }[] }
const modelSorts = [
  ['name-asc', '名称 A → Z'], ['name-desc', '名称 Z → A'],
  ['size-desc', '文件从大到小'], ['size-asc', '文件从小到大'],
  ['format', '按模型格式'], ['running', '运行优先'], ['original', '登记顺序'], ['custom', '自定义顺序'],
] as const
const modelCollator = new Intl.Collator('zh-CN', { numeric: true, sensitivity: 'base' })
const modelSortKey = 'modloader.models.sort'
const modelOrderKey = 'modloader.models.order'
const profileOrderKey = 'modloader.profiles.order'
// Require a deliberate horizontal drag, so ordinary clicks and vertical scrolling keep browsing.
const selectionSwipeDistance = 48
const engineCapabilities = ['sglang', 'ninfer', 'sparkinfer']
const gpuParameterKeys = ['devices', 'split_mode', 'tensor_split', 'main_gpu', 'cuda_visible_devices', 'tensor_parallel_size', 'pipeline_parallel_size', 'gpu']
const secondaryParameterKeys = ['configuration', 'tokenizer', 'pack', 'expert_profile', 'startup_timeout', 'startup_expert_prefetch']
const pagedExpertKeys = ['enable_paged_experts', 'paged_experts_num_resident', 'moe_runner_backend', 'disable_shared_experts_fusion']
const accelerationModes: Record<string, Array<{ value: string; label: string }>> = {
  sglang: [{ value: 'none', label: '关闭' }, { value: 'mtp', label: '原生 MTP / NEXTN' }],
  ninfer: [{ value: 'none', label: '关闭' }, { value: 'mtp', label: '原生 MTP' }],
  sparkinfer: [{ value: 'none', label: '关闭' }, { value: 'dflash2', label: 'DFlash2' }, { value: 'dspark', label: 'DSpark' }],
}
const preparationParameterKeys: Record<string, string[]> = {
  ninfer: ['conversion_recipe', 'conversion_source', 'conversion_device', 'prepare_vision', 'prepare_mtp', 'prepare_proposal', 'chat_template', 'generation_config', 'tokenizer_directory', 'tokenizer_json'],
  sparkinfer: ['tokenizer'],
}

export default function Models(props: PageProps & { active?: boolean }) {
  const { state, refresh, notify, stopInstance, openLogs, navigate } = props
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [multiSelect, setMultiSelect] = useState(false)
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set())
  const [bulkRemove, setBulkRemove] = useState<ModelAsset[] | null>(null)
  const [swiping, setSwiping] = useState<{ id: string; offset: number } | null>(null)
  const [modelOrder, setModelOrder] = useState(() => readLibraryOrder(modelOrderKey))
  const [profileOrder, setProfileOrder] = useState(() => readLibraryOrder(profileOrderKey))
  const [sort, setSort] = useState(() => {
    try { const saved = localStorage.getItem(modelSortKey); return modelSorts.some(([value]) => value === saved) ? saved! : 'name-asc' } catch { return 'name-asc' }
  })
  const list = useRef<HTMLDivElement>(null)
  const profileList = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const panel = list.current?.parentElement
    const layout = panel?.parentElement
    if (!panel || !layout) return
    const resize = () => {
      if (!panel.getClientRects().length) return
      // Anchor the height budget in document space. Scrolling must not enlarge
      // this panel and extend the page when the selected details are shorter.
      const top = Math.max(layout.getBoundingClientRect().top + window.scrollY, parseFloat(getComputedStyle(panel).top) || 0)
      panel.style.setProperty('--library-available-height', `${Math.max(0, window.innerHeight - top - 18)}px`)
    }
    const observer = new ResizeObserver(resize)
    observer.observe(panel)
    const header = document.querySelector('.page-header')
    if (header) observer.observe(header)
    const headerSpace = document.querySelector('.topbar-space')
    if (headerSpace) observer.observe(headerSpace)
    window.addEventListener('resize', resize)
    window.addEventListener('scroll', resize, { passive: true })
    resize()
    return () => { observer.disconnect(); window.removeEventListener('resize', resize); window.removeEventListener('scroll', resize) }
  }, [])
  const [importKind, setImportKind] = useState<'file' | 'directory' | 'config' | null>(null)
  const [editor, setEditor] = useState<{ model: ModelAsset; profile?: LoadProfile; duplicate?: boolean; initialTab?: string } | null>(null)
  const [remove, setRemove] = useState<ModelAsset | null>(null)
  const [filesOpen, setFilesOpen] = useState(false)
  const [deleteProfile, setDeleteProfile] = useState<LoadProfile | null>(null)
  const [starting, setStarting] = useState<Record<string, number>>({})
  const [savingQuickLoad, setSavingQuickLoad] = useState<Record<string, boolean>>({})
  const quickLoadSaving = useRef(new Set<string>())
  const action = useAction(notify, refresh)
  useEffect(() => {
    if (!state.models.some(model => model.id === selectedId)) {
      const transfer = [...(state.migrations || [])].reverse().find(job => job.request?.model_id === selectedId)
      const target = typeof transfer?.model_id === 'string' && state.models.some(model => model.id === transfer.model_id) ? transfer.model_id : null
      if (target || !transfer || !isJobActive(transfer)) setSelectedId(target || state.models[0]?.id || null)
    }
    setCheckedIds(previous => {
      const existing = new Set(state.models.map(model => model.id))
      const next = new Set([...previous].filter(id => existing.has(id)))
      return next.size === previous.size ? previous : next
    })
  }, [state.models, state.migrations, selectedId])
  const selected = state.models.find(model => model.id === selectedId)
  const modelWarnings = selected?.warnings.filter(warning => ![
    '发现视觉投影文件；在加载配置中选择与模型匹配的文件后启用图片输入',
    '已检查 NInfer v3 容器、文件长度及续卷身份；完整模型组件与推理兼容性仍需官方检查工具和引擎确认。',
  ].includes(warning)) || []
  const orderedProfiles = useMemo(() => orderedItems(state.profiles, profileOrder), [state.profiles, profileOrder])
  const profiles = selected ? orderedProfiles.filter(profile => profile.model_id === selected.id) : []
  const duplicateNames = useMemo(() => {
    const names = new Set<string>(), duplicates = new Set<string>()
    for (const model of state.models) {
      if (names.has(model.name)) duplicates.add(model.name)
      names.add(model.name)
    }
    return duplicates
  }, [state.models])
  const activeModels = useMemo(() => new Map(state.instances.filter(instance => isActive(instance.state)).map(instance => [instance.model_id, instance])), [state.instances])
  const migratingModels = new Set((state.migrations || []).filter(isJobActive).map(job => job.request?.model_id))
  const sortedModels = useMemo(() => {
    if (sort === 'custom') return orderedItems(state.models, modelOrder)
    const models = [...state.models]
    const byName = (a: ModelAsset, b: ModelAsset) => modelCollator.compare(a.name, b.name) || modelCollator.compare(a.path, b.path) || a.id.localeCompare(b.id)
    if (sort !== 'original') models.sort((a, b) => {
      if (sort === 'size-asc' || sort === 'size-desc') return Number(modelSizeKnown(b)) - Number(modelSizeKnown(a)) || (sort === 'size-asc' ? a.size_bytes - b.size_bytes : b.size_bytes - a.size_bytes) || byName(a, b)
      if (sort === 'format') return modelCollator.compare(a.format, b.format) || byName(a, b)
      if (sort === 'running') return Number(activeModels.has(b.id)) - Number(activeModels.has(a.id)) || byName(a, b)
      return sort === 'name-desc' ? -byName(a, b) : byName(a, b)
    })
    return models
  }, [state.models, activeModels, sort, modelOrder])
  const filtered = useMemo(() => {
    const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean)
    return sortedModels.filter(model => {
      const text = `${model.name} ${model.architecture || ''} ${model.quantization || ''} ${model.format} ${model.path} ${locationLabel(model.location)}`.toLocaleLowerCase()
      return terms.every(term => text.includes(term)) && (filter === 'all' || (filter === 'running' ? activeModels.has(model.id) : model.format.toLowerCase() === filter))
    })
  }, [sortedModels, activeModels, query, filter])
  const formats = [...new Set(state.models.map(model => model.format.toLowerCase()))].sort(modelCollator.compare)
  const selectedVisible = filtered.some(model => model.id === selectedId)
  const selectable = filtered.filter(model => !activeModels.has(model.id) && !migratingModels.has(model.id))
  const checkedVisible = selectable.filter(model => checkedIds.has(model.id)).length
  const hiddenChecked = [...checkedIds].filter(id => !filtered.some(model => model.id === id)).length
  function toggleBatchSelection(id: string) {
    setCheckedIds(previous => {
      if ((activeModels.has(id) || migratingModels.has(id)) && !previous.has(id)) return previous
      const next = new Set(previous)
      if (next.has(id)) next.delete(id); else next.add(id)
      return next
    })
  }
  function finishSelection() { setMultiSelect(false); setCheckedIds(new Set()) }
  useEffect(() => { if (list.current) list.current.scrollTop = 0 }, [query, filter])
  useEffect(() => { try { localStorage.setItem(modelSortKey, sort) } catch { /* Sorting remains available for this session. */ } }, [sort])
  function saveOrder(key: string, ids: string[]) {
    try { localStorage.setItem(key, JSON.stringify(ids)) }
    catch { notify('排序未能保存，重启后可能丢失', 'error') }
  }
  const reorderDisabled = props.active === false || !!(editor || importKind || remove || bulkRemove || deleteProfile)
  const modelReorder = useLongPressReorder({
    items: filtered, container: list, disabled: reorderDisabled || multiSelect, reset: `${query}\n${filter}\n${sort}`,
    onReorder: next => {
      const ids = mergeVisibleOrder(sortedModels, next)
      setModelOrder(ids); saveOrder(modelOrderKey, ids); setSort('custom')
    },
    onMoveBeforeHold: (id, dx, dy) => {
      const distance = -dx, vertical = Math.abs(dy)
      if (vertical > selectionSwipeDistance && vertical > Math.abs(distance)) { setSwiping(null); return }
      if (distance > 8 && distance > vertical * 1.4) setSwiping({ id, offset: Math.min(distance, selectionSwipeDistance) })
      if (distance >= selectionSwipeDistance && distance > vertical * 1.4) {
        setSwiping(null); setMultiSelect(true)
        if (!activeModels.has(id) && !migratingModels.has(id)) setCheckedIds(new Set([id]))
      }
    },
    onGestureEnd: () => setSwiping(null),
  })
  const profileReorder = useLongPressReorder({
    items: profiles, container: profileList, disabled: reorderDisabled, reset: selectedId || '', pageScroll: true,
    onReorder: next => {
      const ids = mergeVisibleOrder(orderedProfiles, next)
      setProfileOrder(ids); saveOrder(profileOrderKey, ids)
    },
  })
  const draggedModel = modelReorder.drag && filtered.find(model => model.id === modelReorder.drag!.id)
  const draggedProfile = profileReorder.drag && profiles.find(profile => profile.id === profileReorder.drag!.id)
  function locateSelected() {
    const item = list.current?.querySelector<HTMLElement>(`.model-item[data-model-id="${CSS.escape(selectedId || '')}"]`)
    if (item && list.current) {
      const bounds = item.getBoundingClientRect(), viewport = list.current.getBoundingClientRect()
      list.current.scrollTop += bounds.top - viewport.top - (viewport.height - bounds.height) / 2
      item.focus({ preventScroll: true })
    }
  }

  async function start(profile: LoadProfile) {
    if (starting[profile.id]) return
    setStarting(previous => ({ ...previous, [profile.id]: Date.now() }))
    await action.run(() => request('POST', `/api/profiles/${profile.id}/start`))
    setStarting(previous => { const next = { ...previous }; delete next[profile.id]; return next })
  }

  async function toggleQuickLoad(profile: LoadProfile) {
    if (quickLoadSaving.current.has(profile.id)) return
    quickLoadSaving.current.add(profile.id)
    setSavingQuickLoad(previous => ({ ...previous, [profile.id]: true }))
    try {
      const result = await action.run(() => request('PATCH', `/api/profiles/${profile.id}`, { chat_quick_load: !profile.chat_quick_load }))
      return result.ok
    } finally {
      quickLoadSaving.current.delete(profile.id)
      setSavingQuickLoad(previous => { const next = { ...previous }; delete next[profile.id]; return next })
    }
  }

  return <>
    <PageHeader eyebrow="YOUR LOCAL MODELS" title="模型库">
      <Button icon={FileCode2} onClick={() => setImportKind('config')}>导入启动配置</Button>
      <Button icon={FolderOpen} onClick={() => setImportKind('directory')}>导入文件夹</Button>
      <Button variant="primary" icon={Plus} onClick={() => setImportKind('file')}>导入模型</Button>
    </PageHeader>
    <div className="library-layout">
      <div className="panel model-library">
        <div className="library-toolbar"><div className="search-input"><Search size={16} /><input aria-label="搜索本地模型" placeholder="搜索名称、量化或路径" value={query} onChange={event => setQuery(event.target.value)} />{query && <IconButton icon={X} label="清空搜索" onClick={() => setQuery('')} />}</div>
          <div className="library-controls"><label className="filter-control"><ListFilter size={14} /><Select aria-label="筛选模型" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">全部模型</option><option value="running">正在运行</option>{formats.map(format => <option key={format} value={format}>{format.toUpperCase()}</option>)}</Select></label><label className="filter-control"><ArrowDownUp size={14} /><Select aria-label="模型排序" value={sort} onChange={event => { setSort(event.target.value); if (list.current) list.current.scrollTop = 0 }}>{modelSorts.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></label></div></div>
        <div className="library-caption"><span>{filtered.length === state.models.length ? `${filtered.length} 个模型` : `${filtered.length} / ${state.models.length} 个模型`}</span><div className="library-list-actions">{selectedVisible && <IconButton icon={LocateFixed} label="定位当前模型" onClick={locateSelected} />}<Button small variant="ghost" title={multiSelect ? '退出多选并清空勾选' : '也可向左拖动模型条目进入多选'} disabled={!state.models.length && !multiSelect} onClick={() => multiSelect ? finishSelection() : setMultiSelect(true)}>{multiSelect ? '完成选择' : '多选'}</Button></div></div>
        {multiSelect && <div className="library-batch-toolbar"><label><input type="checkbox" aria-label="全选当前结果" checked={!!selectable.length && checkedVisible === selectable.length} ref={input => { if (input) input.indeterminate = checkedVisible > 0 && checkedVisible < selectable.length }} disabled={!selectable.length} onChange={event => {
          const checked = event.target.checked
          setCheckedIds(previous => { const next = new Set(previous); for (const model of selectable) { if (checked) next.add(model.id); else next.delete(model.id) } return next })
        }} />全选当前结果</label>{hiddenChecked > 0 && <span>另有 {hiddenChecked} 项在其他筛选结果中</span>}</div>}
        {selected && !selectedVisible && <div className="library-selection-note" role="status">当前模型已被筛选隐藏</div>}
        <div className="model-list" role="region" aria-label="模型列表" ref={list} {...modelReorder.listProps} onKeyDown={event => {
          if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
          const items = Array.from(list.current?.querySelectorAll<HTMLButtonElement>('.model-item') || [])
          const index = items.indexOf(event.target as HTMLButtonElement)
          if (index < 0) return
          event.preventDefault()
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : Math.max(0, Math.min(items.length - 1, index + (event.key === 'ArrowDown' ? 1 : -1)))
          items[next]?.focus()
        }}>{modelReorder.rendered.map((model, index) => {
          const instance = activeModels.get(model.id)
          const checked = checkedIds.has(model.id)
          const highlighted = multiSelect ? checked : selectedId === model.id
          return <div className={`model-list-row ${checked ? 'checked' : ''} ${modelReorder.drag?.id === model.id ? 'is-dragging' : ''}`} data-reorder-id={model.id} key={model.id}>{multiSelect && <input className="model-row-check" type="checkbox" aria-label={`勾选 ${model.name}`} title={instance ? '请先停止该模型，再移除登记' : `勾选 ${model.name}`} disabled={(!!instance || migratingModels.has(model.id)) && !checked} checked={checked} onChange={() => toggleBatchSelection(model.id)} />}<button data-model-id={model.id} aria-pressed={highlighted} tabIndex={selectedId === model.id || (!selectedVisible && index === 0) ? 0 : -1} className={`model-item ${highlighted ? 'selected' : ''}`} style={swiping?.id === model.id ? { transform: `translateX(-${swiping.offset}px)` } : undefined}
            onPointerDown={event => modelReorder.pointerDown(model.id, event)}
            onClick={() => {
              if (multiSelect) toggleBatchSelection(model.id)
              else { setSelectedId(model.id); setFilesOpen(false) }
            }}>
            <span className={`model-symbol ${model.modalities.includes('image') ? 'vision' : ''}`}>{model.modalities.includes('image') ? <Image size={18} /> : <Layers3 size={18} />}</span><span className="model-item-content"><span className="model-item-name" title={model.name}><span>{modelNameParts(model.name)[0]}</span><span>{modelNameParts(model.name)[1]}</span></span><span className="model-item-meta">{model.format.toUpperCase()}<span>·</span>{!modelSizeKnown(model) ? '大小未统计' : formatBytes(model.size_bytes)}{quantizationLabel(model) && <><span>·</span>{quantizationLabel(model)}</>}{model.location?.platform === 'wsl2' && <><span>·</span>{model.location.distribution}</>}</span>{duplicateNames.has(model.name) && <span className="model-item-path" title={model.path}>{model.path}</span>}{instance && <StateBadge state={instance.state} />}</span><ChevronRight className="model-item-arrow" size={14} />
          </button></div>
        })}
        {!filtered.length && <Empty icon={state.models.length ? Search : Layers3} title={state.models.length ? '没有匹配的模型' : '模型库还是空的'}>{state.models.length ? <Button small onClick={() => { setQuery(''); setFilter('all') }}>清除筛选</Button> : <Button small icon={Search} onClick={() => navigate('discover')}>发现模型</Button>}</Empty>}</div>
        <div className="library-bottom">{multiSelect ? <><span>已选 {checkedIds.size} 个</span><Button small variant="danger" icon={Trash2} disabled={!checkedIds.size} onClick={() => setBulkRemove(state.models.filter(model => checkedIds.has(model.id)))}>移除登记</Button></> : <><HardDrive size={14} /><span>本地文件</span><Button variant="ghost" small icon={FolderOpen} onClick={() => { if (state.settings.model_directory) void action.run(() => openPath(state.settings.model_directory!)) }} disabled={!state.settings.model_directory}>打开目录</Button></>}</div>
      </div>
      {selected ? <div className="model-detail">
        <div className="panel model-overview"><div className="model-detail-heading"><div><div className="detail-eyebrow">{selected.format.toUpperCase()}<span className="text-divider" />{selected.architecture || '架构未知'}</div><h2>{selected.name}</h2><div className="inline-tags"><Badge>{quantizationLabel(selected) || '精度未知'}</Badge>{selected.modalities.includes('image') && <Badge tone="purple"><Image size={14} aria-label="图像输入" /></Badge>}<Badge>{locationLabel(selected.location)}</Badge></div></div><IconButton icon={Trash2} label="移除模型" disabled={migratingModels.has(selected.id)} onClick={() => setRemove(selected)} /></div>
          <div className="model-facts"><KeyValue label="文件大小" value={!modelSizeKnown(selected) ? '未统计' : formatBytes(selected.size_bytes)} /><KeyValue label="来源" value={modelSourceLabel(selected)} /></div>
          <ModelStorage key={selected.id} model={selected} onSelect={setSelectedId} {...props}>
            <div className="path-strip"><FolderOpen size={15} /><span className="mono" title={selected.path}>{selected.path}</span><IconButton icon={Copy} label="复制模型路径" onClick={() => void action.run(() => copy(selected.path), '模型路径已复制')} /><IconButton icon={ArrowRight} label="打开模型目录" onClick={() => void action.run(() => openPath(desktopPath(selected.path, selected.location)))} /></div>
          </ModelStorage>
          {!!modelWarnings.length && <div className="warning-note">{modelWarnings.join('；')}</div>}
        </div>
        <div className="panel profile-panel"><div className="panel-heading"><div><h3>加载配置 <span className="heading-count">{profiles.length}</span></h3></div><Button small icon={Plus} disabled={migratingModels.has(selected.id)} onClick={() => setEditor({ model: selected })}>新建配置</Button></div>
          {profiles.length ? <div className="profile-list" ref={profileList} {...profileReorder.listProps}>{profileReorder.rendered.map(profile => {
            const instances = state.instances.filter(item => item.profile_id === profile.id && isActive(item.state))
            const instance = instances[0]
            const last = state.instances.filter(item => item.profile_id === profile.id).at(-1)
            const backend = state.backends.find(item => item.id === profile.backend)
            const runtime = state.runtimes.find(item => item.id === profile.runtime_id)
            const changed = instance && instance.profile_revision !== profile.revision
            return <div className={`profile-card ${profileReorder.drag?.id === profile.id ? 'is-dragging' : ''}`} data-reorder-id={profile.id} key={profile.id} onPointerDown={event => profileReorder.pointerDown(profile.id, event)}><div className="profile-card-top"><div className="profile-name-row"><span className="profile-icon"><Braces size={18} /></span><div><h4>{profile.name}</h4><span className="profile-subtitle" title={runtime ? runtimeDisplayName(runtime) : undefined}>{runtime ? runtimeOptionLabel(runtime, state.runtimes) : `${backend?.label || profile.backend} · 运行环境未指定`}</span></div></div><div className="inline-tags">{instance ? <StateBadge state={instance.state} /> : last?.state === 'failed' ? <StateBadge state="failed" /> : <Badge dot>未加载</Badge>}{changed && <Badge tone="amber">配置待重载</Badge>}{instance ? <Button small icon={Square} disabled={instance.state === 'stopping'} onClick={() => void stopInstance(instance)}>{instance.state === 'preparing' ? '取消准备' : '停止'}</Button> : <Button small variant="primary" icon={Play} busy={!!starting[profile.id]} disabled={!runtime || migratingModels.has(selected.id)} onClick={() => void start(profile)}>{starting[profile.id] ? '检查启动条件' : last?.state === 'failed' ? '重新加载' : '加载模型'}</Button>}</div></div>
              {instances.slice(1).map(extra => <div className="profile-card-top" key={extra.id}><span className="profile-subtitle">{extra.alias} · {extra.port}</span><div className="inline-tags"><StateBadge state={extra.state} /><Button small icon={Square} disabled={extra.state === 'stopping'} onClick={() => void stopInstance(extra)}>停止</Button><Button small icon={Terminal} onClick={() => openLogs(extra)}>日志</Button>{isReady(extra.state) && <Button small icon={MessageSquare} onClick={() => navigate('chat', extra.id)}>测试对话</Button>}</div><ModelLoadingStatus instance={extra} /></div>)}
              <ProfileConfigSummary profile={profile} active={Boolean(runtime && props.active && !editor)} revision={profilePreflightRevision(state, selected.id, profile.id)} instance={instance} />
              {(instance?.error || (!instance && last?.error)) && <DiagnosticNotice summary="加载失败" detail={instance?.error || last?.error} />}
              <ModelLoadingStatus instance={instance} pendingSince={instance ? undefined : starting[profile.id]} />
              {!instance && runtime && <ResourcePreview compact draft={{ model_id: selected.id, runtime_id: profile.runtime_id, backend: profile.backend, parameters: profile.parameters, extra_args: profile.extra_args }} hardware={state.hardware} active={props.active && !editor} />}
              <div className="profile-actions"><div><QuickLoadToggle enabled={!!profile.chat_quick_load} disabled={!!savingQuickLoad[profile.id] || migratingModels.has(selected.id)} onToggle={() => toggleQuickLoad(profile)} /><IconButton icon={Pencil} label="编辑配置" disabled={migratingModels.has(selected.id)} onClick={() => setEditor({ model: selected, profile })} /><IconButton icon={Copy} label="复制配置" disabled={migratingModels.has(selected.id)} onClick={() => setEditor({ model: selected, profile, duplicate: true })} /><Button small variant="ghost" icon={Terminal} disabled={migratingModels.has(selected.id)} onClick={() => setEditor({ model: selected, profile, initialTab: 'startup' })}>启动参数</Button><IconButton icon={Trash2} label={`删除配置 ${profile.name}`} disabled={!!instance || migratingModels.has(selected.id)} onClick={() => setDeleteProfile(profile)} /></div><div>{instance ? <><Button small icon={Terminal} onClick={() => openLogs(instance)}>日志</Button>{isReady(instance.state) && <Button small icon={MessageSquare} onClick={() => navigate('chat', instance.id)}>测试对话</Button>}</> : last && <Button small icon={Terminal} variant="ghost" onClick={() => openLogs(last)}>日志</Button>}</div></div>
            </div>
          })}</div> : null}
        </div>
        <div className="panel files-panel"><button className="disclosure-heading" onClick={() => setFilesOpen(!filesOpen)}><span><File size={16} />模型文件 <span className="heading-count">{selected.files.length}</span></span><ChevronDown size={17} className={filesOpen ? 'rotated' : ''} /></button>{filesOpen && <div className="file-list">{selected.format.toLowerCase() === 'ninfer' && selected.quantization && <KeyValue label="张量格式" value={selected.quantization} />}{selected.files.length ? selected.files.map(file => <div key={file}><File size={14} /><span className="mono">{file}</span></div>) : <p className="muted">未读取到文件清单。</p>}</div>}</div>
      </div> : <div className="panel selection-empty"><Empty icon={FolderOpen} title="从一个本地模型开始"><div className="import-choices"><button onClick={() => setImportKind('file')}><File size={19} /><span>模型文件</span><small>GGUF 等单文件模型</small><ChevronRight size={15} /></button><button onClick={() => setImportKind('directory')}><FolderOpen size={19} /><span>模型文件夹</span><small>递归导入所有层级的模型</small><ChevronRight size={15} /></button><button onClick={() => setImportKind('config')}><FileCode2 size={19} /><span>启动配置</span><small>导入已有加载参数</small><ChevronRight size={15} /></button></div></Empty></div>}
    </div>
    {modelReorder.drag && draggedModel && <LibraryDragPreview drag={modelReorder.drag} icon={draggedModel.modalities.includes('image') ? <Image size={18} /> : <Layers3 size={18} />} name={draggedModel.name} />}
    {profileReorder.drag && draggedProfile && <LibraryDragPreview drag={profileReorder.drag} icon={<Braces size={18} />} name={draggedProfile.name} />}
    {importKind && <ImportModel kind={importKind} onClose={() => setImportKind(null)} {...props} />}
    {editor && <ProfileEditor key={`${editor.profile?.id || `new-${editor.model.id}`}-${!!editor.duplicate}`} model={editor.model} profile={editor.profile} duplicate={editor.duplicate} initialTab={editor.initialTab} onClose={() => setEditor(null)} {...props} />}
    {remove && <RemoveModel model={remove} onClose={() => setRemove(null)} {...props} />}
    {bulkRemove && <BulkRemoveModels {...props} models={bulkRemove} onComplete={finishSelection} onClose={() => setBulkRemove(null)} />}
    {deleteProfile && <Modal title="删除加载配置？" description={`将删除「${deleteProfile.name}」，模型文件会保留。`} onClose={() => setDeleteProfile(null)} footer={<><Button onClick={() => setDeleteProfile(null)}>取消</Button><Button variant="danger" icon={Trash2} busy={action.busy} onClick={async () => { const result = await action.run(() => request('DELETE', `/api/profiles/${deleteProfile.id}`), '加载配置已删除'); if (result.ok) setDeleteProfile(null) }}>删除配置</Button></>}><KeyValue label="API 名称" value={deleteProfile.alias} mono /></Modal>}
  </>
}

function modelSourceLabel(model: ModelAsset) {
  let source = model.source
  while (source?.kind === 'storage_migration' && source.original_source && typeof source.original_source === 'object') source = source.original_source as Record<string, unknown>
  return String(source?.repo_id || ({ local: '本地导入', hf: 'Hugging Face', modelscope: 'ModelScope', direct: '直链下载', storage_migration: '本地导入' } as Record<string, string>)[String(source?.kind)] || source?.kind || '本地导入')
}

function modelNameParts(name: string) {
  name = name.replace(/-\d+-of-\d+$/, '')
  const separator = name.includes('-VERY-') ? name.lastIndexOf('-VERY-') : name.lastIndexOf('-')
  return separator > 0 ? [name.slice(0, separator + 1), name.slice(separator + 1)] : [name, '']
}

function quantizationLabel(model: ModelAsset) {
  if (model.format.toLowerCase() !== 'ninfer') return model.quantization
  const formats = Array.isArray(model.metadata.ninfer_tensor_formats)
    ? model.metadata.ninfer_tensor_formats.filter((value): value is string => typeof value === 'string')
    : (model.quantization || '').split(' / ').filter(Boolean)
  if (!formats.length) return null
  if (formats.includes('nvfp4')) return formats.length > 1 ? 'NVFP4 混合精度' : 'NVFP4'
  const integers = formats.filter(value => /^q\d+_g\d+_/.test(value))
  if (integers.length > 1) return '混合整数量化'
  if (formats.some(value => value.startsWith('fp8_'))) return formats.length > 1 ? 'FP8 混合精度' : 'FP8'
  if (integers.length === 1) return integers[0].split('_')[0].toUpperCase() + (formats.length > 1 ? ' 混合精度' : '')
  return formats.length > 1 ? '混合精度' : formats[0].toUpperCase()
}

function profilePreflightRevision(state: PageProps['state'], modelId: string, profileId?: string) {
  return state.profiles.filter(item => item.model_id === modelId).map(item => `${item.id}:${item.revision}`).join('|')
    + (state.instances.filter(item => item.profile_id === profileId).at(-1)?.state || '')
}

function ProfileConfigSummary({ profile, active, revision, instance }: { profile: LoadProfile; active: boolean; revision: string; instance?: PageProps['state']['instances'][number] }) {
  const preflight = useProfilePreflight(profile, active, revision)
  return <><div className="profile-config-summary"><span>API <code>{profile.alias}</code></span>{profileSummary(profile, preflight.status).map(item => <span key={item}>{item}</span>)}{instance && <span>端口 {instance.port}<span className="text-divider" />{instance.active_requests} 个请求</span>}</div>
    {!instance && profile.backend === 'strata' && <PreparationSummary {...preflight} />}
    {!instance && engineCapabilities.includes(profile.backend) && <EnginePreparationSummary {...preflight} />}</>
}

function profileSummary(profile: LoadProfile, status?: PreparationStatus) {
  const p = profile.parameters
  const items = []
  if (p.context_length != null) {
    const automatic = profile.backend === 'vllm' && p.context_length === -1
    const modelDefault = p.context_length === 0 || profile.backend === 'sparkinfer' && typeof p.context_length === 'number' && p.context_length < 0
    items.push(`上下文 ${automatic ? '自动' : modelDefault ? '模型默认' : Number(p.context_length).toLocaleString('zh-CN')}`)
  }
  if (p.gpu_layers != null) items.push(`GPU ${p.gpu_layers === -1 || p.gpu_layers === 'all' ? '全部' : p.gpu_layers === 'auto' ? '自动' : `${p.gpu_layers} 层`}`)
  if (p.cache_type_k || p.cache_type_v) items.push(`KV ${p.cache_type_k ?? '默认'} / ${p.cache_type_v ?? '默认'}`)
  else if (p.kv_cache) items.push(`KV ${p.kv_cache}`)
  if (engineCapabilities.includes(profile.backend)) {
    if (status?.acceleration?.enabled) items.push(status.acceleration.label)
  } else if (profile.backend === 'strata' && status?.drafting?.supported) {
    const mode = status.drafting.effective_mode
    if (mode !== 'off') items.push(mode === 'dflash2' ? 'DFlash2' : 'MTP')
  } else if (p.mtp_enabled === true || p.mtp_enabled === undefined && status?.mtp?.enabled) items.push('MTP')
  if (p.vision_enabled === true || p.vision_enabled === undefined && (status?.vision_enabled ?? !!p.mmproj)) items.push('图像')
  return items
}

function ImportModel({ kind, onClose, ...props }: PageProps & { kind: 'file' | 'directory' | 'config'; onClose: () => void }) {
  const [path, setPath] = useState('')
  const [platform, setPlatform] = useState<RuntimePlatform>('windows')
  const [distribution, setDistribution] = useState('')
  const [customModel, setCustomModel] = useState(false)
  const [customKind, setCustomKind] = useState<'file' | 'directory'>('file')
  const [name, setName] = useState('')
  const [folderResult, setFolderResult] = useState<FolderImportResult | null>(null)
  const [importProgress, setImportProgress] = useState<OperationProgress | null>(null)
  const [importStartedAt, setImportStartedAt] = useState(0)
  const action = useAction(props.notify, props.refresh)
  const location: ModelLocation = { platform, ...(platform === 'wsl2' ? { distribution } : {}) }
  function changeLocation(platform: RuntimePlatform, distribution: string) { setPlatform(platform); setDistribution(distribution); setPath(''); setFolderResult(null) }
  async function save() {
    if (kind === 'directory') {
      setFolderResult(null); setImportProgress(null); setImportStartedAt(Date.now())
      const result = await action.run(() => requestProgress<FolderImportResult>('POST', '/api/models/import-folder', { path, location }, { onProgress: setImportProgress }))
      if (result.ok) {
        setFolderResult(result.data)
      }
      return
    }
    const result = await action.run(() => request('POST', customModel ? '/api/models/import-custom' : kind === 'config' ? '/api/config/import' : '/api/models/import', kind === 'config' ? { path } : { path, location, ...(name.trim() ? { name: name.trim() } : {}) }), kind === 'config' ? '启动配置已导入' : '模型已导入')
    if (result.ok) onClose()
  }
  return <Modal title={kind === 'config' ? '导入启动配置' : kind === 'directory' ? '导入模型文件夹' : '导入模型文件'} description={kind === 'config' ? undefined : kind === 'directory' ? '包含子目录，原文件保留。' : '原文件保留。'} onClose={onClose} dismissible={!action.busy} footer={<><Button disabled={action.busy} onClick={onClose}>{folderResult ? '完成' : '取消'}</Button><Button variant="primary" icon={Plus} busy={action.busy} disabled={!path.trim() || kind !== 'config' && platform === 'wsl2' && !distribution} onClick={() => void save()}>{kind === 'directory' ? action.busy ? '正在扫描并导入…' : folderResult ? '重新扫描' : '扫描并导入' : '导入'}</Button></>}>
    {kind !== 'config' && <><Field label="模型位置"><Select value={platform} disabled={action.busy} onChange={event => changeLocation(event.target.value as RuntimePlatform, '')}>{Object.entries(platformLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></Field>{platform === 'wsl2' && <WslDistributionField value={distribution} onChange={value => changeLocation(platform, value)} disabled={action.busy} />}</>}
    {kind === 'file' && <><label className="custom-engine-check"><input type="checkbox" checked={customModel} disabled={action.busy} onChange={event => setCustomModel(event.target.checked)} />按自定义模型路径登记</label>{customModel && <><p className="field-help">不检查格式或权重完整性。</p><Field label="路径类型"><Select value={customKind} disabled={action.busy} onChange={event => setCustomKind(event.target.value as 'file' | 'directory')}><option value="file">模型文件</option><option value="directory">模型目录</option></Select></Field></>}</>}
    <Field label={customModel ? '模型路径' : kind === 'directory' ? '模型文件夹路径' : '文件路径'}><PlatformPathInput autoFocus label={customModel ? '模型路径' : kind === 'directory' ? '模型文件夹路径' : '文件路径'} disabled={action.busy} location={kind === 'config' ? undefined : location} kind={customModel ? customKind : kind === 'directory' ? 'directory' : 'file'} value={path} onChange={value => { setPath(value); setFolderResult(null) }} placeholder={platform === 'wsl2' ? kind === 'directory' || customModel && customKind === 'directory' ? '/home/user/models' : '/home/user/models/model.gguf' : kind === 'directory' ? 'D:\\Models' : kind === 'config' ? '启动配置文件路径' : 'D:\\Models\\model.gguf'} /></Field>{kind === 'file' && <Field label="显示名称"><input disabled={action.busy} value={name} onChange={event => setName(event.target.value)} placeholder="默认使用文件名称" /></Field>}
    {kind === 'directory' && action.busy && <OperationStatus progress={importProgress} startedAt={importStartedAt} fallback="正在扫描模型目录" />}
    {kind === 'directory' && !folderResult && !action.busy && <p className="field-help">支持 GGUF、NInfer 和完整 HF 目录；跳过已导入模型。</p>}
    {folderResult && <div className="folder-import-result" aria-live="polite">
      <div className="folder-import-counts"><Badge tone="green">新增 {folderResult.imported.length} 个</Badge><Badge>已存在 {folderResult.existing.length} 个</Badge>{folderResult.errors.length > 0 && <Badge tone="amber">未导入 {folderResult.errors.length} 项</Badge>}{folderResult.skipped.length > 0 && <Badge>已跳过 {folderResult.skipped.length} 项</Badge>}</div>
      {!folderResult.imported.length && !folderResult.existing.length && !folderResult.errors.length && <p className="muted">未找到可导入的 GGUF、NInfer 工件或完整 HF 模型。</p>}
      {folderResult.imported.length > 0 && <details className="folder-import-details"><summary>已导入的模型（{folderResult.imported.length}）</summary>{folderResult.imported.map(model => <div className="folder-import-entry" key={model.id}><strong>{model.name}</strong><span className="mono">{model.path}</span></div>)}</details>}
      {folderResult.existing.length > 0 && <details className="folder-import-details"><summary>已登记的模型（{folderResult.existing.length}）</summary>{folderResult.existing.map(model => <div className="folder-import-entry" key={model.id}><strong>{model.name}</strong><span className="mono">{model.path}</span></div>)}</details>}
      {folderResult.errors.length > 0 && <details className="folder-import-details" open><summary>未导入的项目（{folderResult.errors.length}）</summary>{folderResult.errors.map((item, index) => <div className="folder-import-entry" key={`${item.path}-${index}`}><span className="mono">{item.path}</span><span className="error-text">{item.error}</span></div>)}</details>}
      {folderResult.skipped.length > 0 && <details className="folder-import-details"><summary>跳过的辅助文件或目录（{folderResult.skipped.length}）</summary>{folderResult.skipped.map((item, index) => <div className="folder-import-entry" key={`${item.path}-${index}`}><span className="mono">{item.path}</span><span>{item.reason}</span></div>)}</details>}
    </div>}
  </Modal>
}

function ProfileEditor({ model, profile, duplicate = false, initialTab = 'load', onClose, ...props }: PageProps & { model: ModelAsset; profile?: LoadProfile; duplicate?: boolean; initialTab?: string; onClose: () => void }) {
  const compatibleRuntimes = props.state.runtimes.filter(runtime => modelSupportsRuntime(model, runtime, props.state.backends))
  const compatibleBackends = props.state.backends.filter(backend => model.location?.platform !== 'wsl2' || backendPlatforms(backend).includes('wsl2') || compatibleRuntimes.some(runtime => runtime.backend === backend.id))
  const preferredBackend = ({ custom: 'custom', gguf: 'llama_cpp', ninfer: 'ninfer' } as Record<string, string>)[model.format.toLowerCase()] || 'transformers'
  const defaultBackend = profile?.backend || compatibleBackends.find(item => item.id === preferredBackend)?.id || compatibleBackends[0]?.id || ''
  const modelProfiles = props.state.profiles.filter(item => item.model_id === model.id)
  const aliasBase = duplicate && profile ? `${profile.alias}-copy` : model.name.replace(/[^\p{L}\p{N}_.-]+/gu, '-').replace(/^-+|-+$/g, '') || 'local-model'
  const aliases = new Set(props.state.profiles.map(item => item.alias))
  let defaultAlias = aliasBase
  let aliasSuffix = 2
  while (aliases.has(defaultAlias)) defaultAlias = `${aliasBase}-${aliasSuffix++}`
  let copyName = `${profile?.name || '配置'} · 副本`, copySuffix = 2
  while (modelProfiles.some(item => item.name === copyName)) copyName = `${profile?.name || '配置'} · 副本 ${copySuffix++}`
  const [name, setName] = useState(duplicate ? copyName : profile?.name || (modelProfiles.length ? `配置 ${modelProfiles.length + 1}` : '默认配置'))
  const [backend, setBackend] = useState(defaultBackend)
  const [runtimeId, setRuntimeId] = useState(profile?.runtime_id || compatibleRuntimes.find(runtime => runtime.backend === defaultBackend)?.id || '')
  const [alias, setAlias] = useState(!duplicate && profile ? profile.alias : defaultAlias)
  const [parameters, setParameters] = useState<Record<string, unknown>>(() => structuredClone(profile?.parameters || {}))
  const [generation, setGeneration] = useState<Record<string, unknown>>(() => structuredClone(profile?.generation || {}))
  const [extraArgs, setExtraArgs] = useState(profile?.extra_args || [])
  const [editorVersion, setEditorVersion] = useState(profile?.parameter_editor_version)
  const [advanced, setAdvanced] = useState(() => secondaryParameterKeys.some(key => parameters[key] != null && parameters[key] !== ''))
  const [gpuOpen, setGpuOpen] = useState(() => gpuParameterKeys.some(key => profile?.parameters[key] != null))
  const [gpuEdited, setGpuEdited] = useState(false)
  const [tab, setTab] = useState(initialTab)
  const [validation, setValidation] = useState('')
  const [savingDraft, setSavingDraft] = useState(false)
  const saving = useRef(false)
  const action = useAction(props.notify, props.refresh)
  const startup = useStartupParameters({ id: profile?.id, model_id: model.id, name, alias, backend, runtime_id: runtimeId,
    parameters, generation, extra_args: extraArgs, parameter_editor_version: editorVersion }, tab === 'startup', value => {
    setParameters(value.parameters); setGeneration(value.generation); setExtraArgs(value.extra_args)
    setEditorVersion(1); setValidation('')
  })
  async function changeTab(next: string) {
    if (tab === 'startup') await startup.flush()
    setTab(next)
  }
  const schema = compatibleBackends.find(item => item.id === backend)
  const runtimes = compatibleRuntimes.filter(runtime => runtime.backend === backend)
  const selectedRuntime = runtimes.find(runtime => runtime.id === runtimeId)
  const pathLocation = selectedRuntime ? runtimeLocation(selectedRuntime, props.state.backends) : model.location
  const preflight = useProfilePreflight({ id: profile?.id, name: profile?.name, model_id: model.id, runtime_id: runtimeId, backend, parameters, generation, extra_args: extraArgs }, true, profilePreflightRevision(props.state, model.id, profile?.id) + JSON.stringify(selectedRuntime?.startup_expert_prefetch))
  const loadFields = preflight.status?.parameters || schema?.parameters || []
  const engineAcceleration = engineCapabilities.includes(backend)
  const deepseekStrata = backend === 'strata' && (model.architecture === 'deepseek41' || model.metadata['general.architecture'] === 'deepseek41')
  const accelerationFields = engineAcceleration ? preflight.status?.acceleration?.fields || [] : []
  const preparationFields = engineAcceleration ? preflight.status?.preparation_fields || [] : []
  const pagedFields = backend === 'sglang' && loadFields.some(field => field.key === 'enable_paged_experts')
    ? loadFields.filter(field => pagedExpertKeys.includes(field.key)) : []
  const paging = preflight.status?.paged_experts
  const pagingFromExtras = Boolean(paging?.normalized_parameters && paging.normalized_extra_args
    && JSON.stringify(preflight.checkedDraft?.extra_args) === JSON.stringify(extraArgs)
    && JSON.stringify(paging.normalized_extra_args) !== JSON.stringify(extraArgs))
  const pagingOverrides = pagingFromExtras ? Object.fromEntries(pagedExpertKeys
    .filter(key => Object.hasOwn(paging!.normalized_parameters!, key)).map(key => [key, paging!.normalized_parameters![key]])) : {}
  const pagingParameters = { ...parameters, ...pagingOverrides }
  const pagingEnabled = pagingParameters.enable_paged_experts ?? paging?.enabled ?? false
  const accelerationKeys = engineAcceleration ? ['acceleration_mode', ...accelerationFields.map(field => field.key)] : []
  const capabilityKeys = [...accelerationKeys, ...pagedFields.map(field => field.key), ...(backend === 'strata' ? ['strata_drafting', 'dflash2_path'] : []), ...(backend === 'ninfer' ? ['vision_enabled'] : []), ...(preparationParameterKeys[backend] || []), ...preparationFields.map(field => field.key)]
  const unknownKeys = Object.keys(parameters).filter(key => !['vision_enabled', 'strata_resource_id', 'mtp_enabled', 'mtp_path', 'mtp_tokens', 'resident_experts', ...capabilityKeys].includes(key) && !loadFields.some(field => field.key === key))
  const generationFields = preflight.status?.generation_parameters || runtimeGenerationFields(schema?.generation_parameters || [], selectedRuntime)
  const unsupportedGeneration = backend === 'custom' ? Object.keys(generation).filter(key => generation[key] != null && !generationFields.some(field => field.key === key)) : []
  const gpuField = loadFields.find(field => field.key === 'gpu_layers')
  const legacyGpuAll = !gpuEdited && backend === profile?.backend && parameters.gpu_layers === profile?.parameters.gpu_layers && typeof parameters.gpu_layers === 'number' && gpuField?.maximum != null && parameters.gpu_layers > gpuField.maximum
  const secondaryFields = loadFields.filter(field => secondaryParameterKeys.includes(field.key) && !capabilityKeys.includes(field.key))
  const gpuFields = backend === 'custom' ? [] : loadFields.filter(field => gpuParameterKeys.includes(field.key)).sort((a, b) => gpuParameterKeys.indexOf(a.key) - gpuParameterKeys.indexOf(b.key))
  const inheritedGpu = backend === 'strata' && typeof parameters.strata_config === 'object' && parameters.strata_config !== null ? (parameters.strata_config as Record<string, unknown>).gpu : undefined
  const mainFields = loadFields.filter(field => field.key !== 'mmproj' && !secondaryFields.includes(field) && !gpuFields.includes(field) && !capabilityKeys.includes(field.key))
  const changed = !!profile && (name !== profile.name || alias !== profile.alias || backend !== profile.backend || runtimeId !== profile.runtime_id || JSON.stringify(parameters) !== JSON.stringify(profile.parameters) || JSON.stringify(generation) !== JSON.stringify(profile.generation) || JSON.stringify(extraArgs) !== JSON.stringify(profile.extra_args))
  const activeProfile = !!profile && !duplicate && props.state.instances.some(item => item.profile_id === profile.id && isActive(item.state))
  const projectorVision = ['llama_cpp', 'strata'].includes(backend) && !deepseekStrata
  const visionSupported = projectorVision || backend === 'ninfer'
  const visionEnabled = parameters.vision_enabled === undefined ? preflight.status?.vision_enabled ?? !!parameters.mmproj : parameters.vision_enabled === true
  const visionPath = typeof parameters.mmproj === 'string' ? parameters.mmproj : preflight.status?.mmproj || ''
  const visionCandidates = [...new Set([...(preflight.status?.mmproj_candidates || model.metadata.mmproj_candidates as string[] || []), ...(visionPath ? [visionPath] : [])])]
  const parameterPlaceholder = (field: ParameterField) => ['engram', 'engram_meta', 'dspark'].includes(field.key) ? preflight.status?.dependencies?.[field.key] || '自动查找本地文件' : ['tokenizer', 'pack', 'expert_profile'].includes(field.key) ? '自动' : field.key === 'configuration' ? '可选' : undefined
  function setValue(target: 'parameters' | 'generation', key: string, value: unknown) {
    setValidation('')
    if (target === 'parameters' && key === 'gpu_layers') setGpuEdited(true)
    const setter = target === 'parameters' ? setParameters : setGeneration
    setter(previous => { const next = { ...previous }; if (value === undefined) delete next[key]; else next[key] = value; return next })
  }
  function changeBackend(value: string) {
    setValidation('')
    setBackend(value); setRuntimeId(compatibleRuntimes.find(runtime => runtime.backend === value)?.id || '')
    if (value !== backend) { setParameters({}); setGpuEdited(false) }
  }
  function setPagedValue(key: string, value: unknown) {
    setValidation('')
    setParameters(previous => {
      const next = { ...previous, ...pagingOverrides }
      if (value === undefined) delete next[key]; else next[key] = value
      return next
    })
    if (pagingFromExtras) setExtraArgs(paging!.normalized_extra_args!)
  }
  function changeVision(enabled: boolean) {
    setParameters(previous => ({ ...previous, vision_enabled: enabled, ...(projectorVision && enabled && !visionPath && visionCandidates.length === 1 ? { mmproj: visionCandidates[0] } : {}) }))
  }
  async function save() {
    if (saving.current) return
    saving.current = true; setSavingDraft(true)
    try {
    if (!name.trim() || !alias.trim() || !backend || !runtimeId) { setValidation('请填写配置名称、API 名称，并选择运行环境。'); return }
    if (!selectedRuntime) { setValidation(model.location?.platform === 'wsl2' ? '请选择与模型同一 WSL 发行版的运行环境。' : '请选择可用的运行环境。'); return }
    if (unsupportedGeneration.length) { setTab('generation'); setValidation('请移除此模板不支持的生成默认值，或在引擎模板中启用对应字段。'); return }
    const startupDraft = await startup.prepareSave()
    if (!startupDraft) { setTab('startup'); return }
    const savedParameters = startupDraft.parameters
    const enabledAcceleration = engineAcceleration && (savedParameters.acceleration_mode ?? preflight.status?.acceleration?.mode ?? 'none') !== 'none'
    const draftMode = savedParameters.strata_drafting ?? preflight.status?.drafting?.mode ?? 'legacy'
    const explicitDrafting = backend === 'strata' && draftMode !== 'legacy'
    const enabledMtp = explicitDrafting ? draftMode === 'mtp' || draftMode === 'auto' && preflight.status?.drafting?.effective_mode === 'mtp'
      : savedParameters.mtp_enabled ?? preflight.status?.mtp?.enabled
    const checkedLoadFields = [...loadFields.filter(field => !accelerationKeys.includes(field.key)), ...(enabledAcceleration ? accelerationFields : []), ...(!engineAcceleration && enabledMtp && preflight.status?.mtp?.token_field ? [preflight.status.mtp.token_field] : [])]
    const invalidLoad = checkedLoadFields.filter(field => field.type === 'number').map(field => {
      if (backend === 'sglang' && field.key === 'paged_experts_num_resident') {
        const effective = pagingFromExtras ? pagingParameters : savedParameters
        return effective.enable_paged_experts ?? paging?.enabled ? numberError(field, effective[field.key]) : ''
      }
      if (field.key === 'gpu_layers') {
        if (gpuEdited) return gpuInputError(field, savedParameters[field.key])
        if (legacyGpuAll) return ''
      }
      return numberError(field, savedParameters[field.key])
    }).find(Boolean)
    const invalidGeneration = generationFields.map(field => numberError(field, startupDraft.generation[field.key])).find(Boolean)
    if (invalidLoad || invalidGeneration) { setTab(invalidLoad ? 'load' : 'generation'); setValidation(invalidLoad || invalidGeneration || ''); return }
    const savedVisionEnabled = savedParameters.vision_enabled === undefined ? preflight.status?.vision_enabled ?? !!savedParameters.mmproj : savedParameters.vision_enabled === true
    const savedVisionPath = savedParameters.mmproj ?? preflight.status?.mmproj
    if (projectorVision && savedVisionEnabled && !savedVisionPath) { setValidation('启用图像输入前请选择视觉投影器。'); return }
    if (!engineAcceleration && !explicitDrafting && savedParameters.mtp_enabled === true && preflight.status?.mtp?.needs_path && !(savedParameters.mtp_path ?? preflight.status.mtp.path)) { setValidation('启用 MTP 前请选择配套的本地文件。'); return }
    const body = { ...(duplicate && profile ? { copy_from: profile.id } : {}), model_id: model.id, name: name.trim(), backend, runtime_id: runtimeId, alias: alias.trim(), parameters: savedParameters, generation: startupDraft.generation, extra_args: startupDraft.extra_args, ...(startupDraft.parameter_editor_version ? { parameter_editor_version: startupDraft.parameter_editor_version } : {}) }
    const editing = profile && !duplicate
    const result = await action.run(() => request(editing ? 'PATCH' : 'POST', editing ? `/api/profiles/${profile.id}` : '/api/profiles', body), duplicate ? '配置副本已创建' : profile ? '加载配置已保存' : '加载配置已创建')
    if (result.ok) onClose()
    } finally { saving.current = false; setSavingDraft(false) }
  }
  return <Modal wide className="profile-editor" title={duplicate ? '复制加载配置' : profile ? '编辑加载配置' : '新建加载配置'} description={model.name} onClose={onClose} dismissible={!action.busy && !savingDraft} footer={<><span className="footer-note">{activeProfile && changed ? '保存后需重载运行实例' : ''}</span><Button onClick={onClose} disabled={action.busy || savingDraft}>取消</Button><Button variant="primary" busy={action.busy || savingDraft} onClick={() => void save()}>保存配置</Button></>}>
    {validation && <ErrorNotice message={validation} />}
    <fieldset className="profile-editor-fields" disabled={action.busy || savingDraft} aria-busy={action.busy || savingDraft}>
    <div className="form-grid"><Field label="配置名称"><input autoFocus value={name} onChange={event => setName(event.target.value)} /></Field><Field label="API 模型名称"><input value={alias} onChange={event => setAlias(event.target.value)} placeholder="模型调用名称" /></Field><Field label="运行后端"><Select value={backend} onChange={event => changeBackend(event.target.value)}><option value="" disabled>选择后端</option>{backend && !compatibleBackends.some(item => item.id === backend) && <option value={backend} disabled>{backend}（与模型位置不兼容）</option>}{compatibleBackends.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</Select></Field><Field label="运行环境"><Select value={runtimeId} onChange={event => setRuntimeId(event.target.value)}><option value="">选择运行环境</option>{runtimeId && !selectedRuntime && <option value={runtimeId} disabled>原环境不可用于此模型位置</option>}{runtimes.map(runtime => <option key={runtime.id} value={runtime.id}>{runtimeOptionLabel(runtime, runtimes)}</option>)}</Select></Field></div>
    {model.location?.platform === 'wsl2' && <p className="field-help">运行环境需位于 {model.location.distribution}</p>}
    {!runtimes.length && <div className="warning-note">没有可用的运行环境。<Button small variant="ghost" onClick={() => { onClose(); props.navigate('runtimes') }}>前往运行环境 <ArrowRight size={13} /></Button></div>}
    {selectedRuntime && <details className="runtime-details"><summary>环境详情</summary><dl><dt>平台</dt><dd>{locationLabel(runtimeLocation(selectedRuntime, props.state.backends))}</dd><dt>{selectedRuntime.backend === 'strata' ? '引擎版本' : '版本'}</dt><dd>{selectedRuntime.version || '未知'}</dd><dt>路径</dt><dd className="mono">{selectedRuntime.path}</dd></dl></details>}
    <div className="sub-tabs" role="tablist"><button role="tab" aria-selected={tab === 'load'} className={tab === 'load' ? 'active' : ''} onClick={() => void changeTab('load')}>加载参数</button><button role="tab" aria-selected={tab === 'generation'} className={tab === 'generation' ? 'active' : ''} onClick={() => void changeTab('generation')}>生成默认值</button><button role="tab" aria-selected={tab === 'startup'} className={tab === 'startup' ? 'active' : ''} onClick={() => void changeTab('startup')}>启动参数</button></div>
    {tab === 'load' && <>
      {backend === 'strata' && runtimeId && <PreparationDetails {...preflight} selected={typeof parameters.strata_resource_id === 'string' ? parameters.strata_resource_id : undefined} onSelect={value => setValue('parameters', 'strata_resource_id', value)} />}
      {runtimeId && engineAcceleration && <EnginePreparation status={preflight.status} checking={preflight.checking} progress={preflight.progress} startedAt={preflight.startedAt} location={pathLocation} parameters={parameters} onChange={(key, value) => setValue('parameters', key, value)} />}
      <div className={`parameter-grid load-parameters ${backend}-parameters`}>{engineAcceleration && preflight.checking && !preflight.status ? <p className="muted" role="status">正在读取加载参数…</p> : <>{mainFields.map(field => <ParameterInput location={pathLocation} key={`${backend}-${field.key}`} field={field} gpuEdited={gpuEdited} value={parameters[field.key]} defaultText={parameterPlaceholder(field)} onChange={value => setValue('parameters', field.key, value)} />)}{!loadFields.length && <p className="muted">此后端未提供可编辑的加载参数。</p>}</>}</div>
      {!!gpuFields.length && <details className="profile-gpu capability-section" open={gpuOpen} onToggle={event => setGpuOpen(event.currentTarget.open)}>
        <summary>GPU 分配</summary>
        <div className="parameter-grid load-parameters">{gpuFields.map(field => <div key={field.key} className={['devices', 'cuda_visible_devices', 'gpu'].includes(field.key) ? 'gpu-device-field' : undefined}>
          <ParameterInput field={field} value={Array.isArray(parameters[field.key]) ? (parameters[field.key] as unknown[]).join(',') : parameters[field.key]} defaultText={field.key === 'gpu' ? inheritedGpu != null ? `继承 ${String(inheritedGpu)}` : '继承配置' : undefined} onChange={value => setValue('parameters', field.key, value)} />
          {field.description && <p className="field-help">{field.description}</p>}
          {['devices', 'cuda_visible_devices', 'gpu'].includes(field.key) && <GpuChoices fieldKey={field.key} devices={gpuList(props.state.hardware)} value={parameters[field.key] ?? (field.key === 'gpu' ? inheritedGpu : undefined)} onChange={value => setValue('parameters', field.key, value)} />}
        </div>)}</div>
      </details>}
      {!!pagedFields.length && <section className="capability-section" aria-label="专家分页设置"><div className="parameter-grid">
        {pagedFields.filter(field => field.key === 'enable_paged_experts' || pagingEnabled).map(field => <ParameterInput key={field.key} field={field}
          value={pagingParameters[field.key]}
          defaultText={field.key === 'moe_runner_backend' && paging?.recommended_runner ? `默认 ${paging.recommended_runner}`
            : field.key === 'disable_shared_experts_fusion' && paging?.disable_shared_experts_fusion != null
              ? parameterDefaultText({ ...field, default: paging.disable_shared_experts_fusion }) : undefined}
          onChange={value => setPagedValue(field.key, value)} />)}
      </div></section>}
      {runtimeId && engineAcceleration && <EngineAcceleration backend={backend} location={pathLocation} parameters={parameters} status={preflight.status?.acceleration} reportedErrors={preflight.status?.status === 'error' ? [preflight.status.message, ...preflight.status.errors] : []} checking={preflight.checking} progress={preflight.progress} startedAt={preflight.startedAt} error={preflight.error} retry={preflight.retry} onChange={(key, value) => setValue('parameters', key, value)} />}
      {runtimeId && backend !== 'custom' && !engineAcceleration && !deepseekStrata && <ProfileAcceleration location={pathLocation} model={model} backend={backend} parameters={parameters} status={preflight.status?.mtp} drafting={preflight.status?.drafting} checking={preflight.checking} progress={preflight.progress} startedAt={preflight.startedAt} error={preflight.error} retry={preflight.retry} onChange={values => { setValidation(''); setParameters(previous => ({ ...previous, ...values })) }} />}
      {visionSupported && <section className="profile-vision capability-section" aria-label="图像输入设置"><label className="profile-vision-toggle"><input type="checkbox" checked={visionEnabled} onChange={event => changeVision(event.target.checked)} />图像输入</label>{visionEnabled && projectorVision && <div className="capability-fields"><Field label="视觉投影器"><div className="input-with-button"><Select aria-label="视觉投影器" value={visionPath} onChange={event => setValue('parameters', 'mmproj', event.target.value || undefined)}><option value="">选择视觉投影器</option>{visionCandidates.map(path => <option key={path} value={path}>{path.split(/[\\/]/).at(-1)}</option>)}</Select><PlatformBrowseButton location={pathLocation} kind="file" value={visionPath} onChange={path => setValue('parameters', 'mmproj', path)} /></div></Field>{!preflight.checking && preflight.status?.mmproj_error && <p className="error-text" role="alert">{preflight.status.mmproj_error}</p>}</div>}</section>}
      {backend === 'strata' && !deepseekStrata && <section className="capability-section resident-experts"><label className="resident-experts-toggle"><input type="checkbox" checked={parameters.resident_experts === undefined ? preflight.status?.mtp?.resident_experts ?? false : parameters.resident_experts === true} disabled={!preflight.status?.mtp && parameters.resident_experts === undefined} onChange={event => setValue('parameters', 'resident_experts', event.target.checked)} />专家常驻 RAM</label></section>}
      {!!secondaryFields.length && <><button className="advanced-toggle" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}><ChevronDown size={15} className={advanced ? 'rotated' : ''} />文件与启动选项</button>{advanced && <div className="parameter-grid conditional-fields">
        {secondaryFields.map(field => field.key === 'startup_expert_prefetch' ? <div key={`${backend}-${field.key}`}><ParameterInput location={pathLocation} field={field} value={parameters[field.key]} onChange={value => setValue('parameters', field.key, value)} />{selectedRuntime && <RuntimeStartupPrefetch {...props} key={selectedRuntime.id} runtime={selectedRuntime} />}</div> : <ParameterInput location={pathLocation} key={`${backend}-${field.key}`} field={field} value={parameters[field.key]} defaultText={parameterPlaceholder(field)} onChange={value => setValue('parameters', field.key, value)} />)}
      </div>}</>}
      {unknownKeys.length > 0 && <details className="retained-parameters"><summary>导入配置原始数据</summary>{unknownKeys.map(key => <div key={key}><code>{key}</code><pre>{typeof parameters[key] === 'object' ? JSON.stringify(parameters[key], null, 2) : String(parameters[key])}</pre><IconButton icon={X} label={`移除参数 ${key}`} onClick={() => setValue('parameters', key, undefined)} /></div>)}</details>}
    <ResourcePreview draft={{ model_id: model.id, runtime_id: runtimeId, backend, parameters, extra_args: extraArgs }} hardware={props.state.hardware} alreadyLoaded={!duplicate && props.state.instances.some(item => item.profile_id === profile?.id && isActive(item.state))} />
    </>}
    {tab === 'generation' && <>{unsupportedGeneration.length > 0 && <div className="custom-engine-unsupported"><span>模板未启用：{unsupportedGeneration.join('、')}</span><Button small onClick={() => { setGeneration(previous => Object.fromEntries(Object.entries(previous).filter(([key]) => !unsupportedGeneration.includes(key)))); setValidation('') }}>移除不支持的默认值</Button></div>}<div className="parameter-grid">{generationFields.map(field => <ParameterInput key={field.key} field={field} value={generation[field.key]} defaultText={generationDefaultText(field, { backend, generation: {}, extra_args: extraArgs })} onChange={value => setValue('generation', field.key, value)} />)}</div></>}
    {tab === 'startup' && <StartupParameters editor={startup} />}
    </fieldset>
  </Modal>
}

function EnginePreparationSummary({ status, checking, progress, startedAt, error, retry }: { status?: PreparationStatus; checking: boolean; progress?: OperationProgress; startedAt: number; error?: string; retry: () => void }) {
  if (checking) return <div className="preparation-summary"><OperationStatus progress={progress} startedAt={startedAt} fallback="正在检查模型准备文件" /></div>
  if (!error && (!status || status.status === 'ready') && status?.acceleration?.status !== 'unsupported') return null
  return <div className={`preparation-summary ${error || status?.status === 'error' || status?.status === 'choice_required' || status?.acceleration?.status === 'unsupported' ? 'error-text' : ''}`}>
    {error ? <DiagnosticNotice summary="模型预检失败" detail={error} retry={retry} /> : status?.status !== 'ready' ? status?.message : status?.acceleration?.message}
  </div>
}

function EnginePreparation({ status, checking, progress, startedAt, location, parameters, onChange }: {
  status?: PreparationStatus; checking: boolean; progress?: OperationProgress; startedAt: number; location?: ModelLocation; parameters: Record<string, unknown>; onChange: (key: string, value: unknown) => void
}) {
  const fields = status?.preparation_fields || []
  const problem = status?.status === 'error' ? <>
    <DiagnosticNotice summary={status.message || status.parameter_migration?.message || '模型预检未通过'} detail={status.errors.join('\n')} />
    {status.parameter_migration && <Button small variant="ghost" onClick={() => onChange(status.parameter_migration!.parameter, undefined)}>移除旧容量参数</Button>}
  </> : null
  if (!checking && !status?.kind && !fields.length) return problem
  const labels = { ready: '已就绪', needs_preparation: '加载时准备', choice_required: '需要选择', error: '需要处理' }
  const recipe = status?.recipes?.find(item => item.id === parameters.conversion_recipe)
  return <section className="engine-preparation capability-section" aria-label="模型准备">
    <div className="preparation-heading"><strong>模型准备</strong>{checking ? <OperationStatus progress={progress} startedAt={startedAt} fallback="正在检查模型准备文件" /> : <span className={status?.status === 'error' ? 'error-text' : ''}>{status ? labels[status.status] : ''}</span>}</div>
    {!checking && (problem || status?.status === 'choice_required' && status.message && <p className="field-help">{status.message}</p>)}
    {!!fields.length && <div className="parameter-grid engine-preparation-fields">{fields.map(field => {
      const described = field.key === 'conversion_recipe' && status?.recipes?.length ? { ...field, options: status.recipes.map(item => ({ value: item.id, label: item.label })) } : field
      return <div key={field.key}><ParameterInput field={described} value={parameters[field.key]} location={location} browseDirectories={['conversion_source', 'tokenizer'].includes(field.key)} defaultText={field.key === 'conversion_recipe' ? '选择转换方案' : field.type === 'boolean' && typeof field.default === 'boolean' ? `默认 ${field.default ? '开启' : '关闭'}` : undefined} onChange={value => onChange(field.key, value)} />{field.description && <p className="field-help">{field.description}</p>}{field.key === 'conversion_recipe' && recipe?.description && <p className="field-help">{recipe.description}</p>}</div>
    })}</div>}
    {status?.artifact && <div className="engine-artifact"><span>运行文件</span><code>{status.artifact}</code></div>}
  </section>
}

function EngineAcceleration({ backend, location, parameters, status, reportedErrors, checking, progress, startedAt, error, retry, onChange }: {
  backend: string; location?: ModelLocation; parameters: Record<string, unknown>; status?: AccelerationStatus
  reportedErrors: string[]
  checking: boolean; progress?: OperationProgress; startedAt: number; error?: string; retry: () => void; onChange: (key: string, value: unknown) => void
}) {
  const mode = typeof parameters.acceleration_mode === 'string' ? parameters.acceleration_mode : status?.mode || 'none'
  return <section className="engine-acceleration capability-section" aria-label="推测加速设置">
    <div className="engine-acceleration-mode"><Field label="推测加速"><Select value={mode} onChange={event => onChange('acceleration_mode', event.target.value)}>
      {mode && !accelerationModes[backend]?.some(item => item.value === mode) && <option value={mode} disabled>{mode}（已保存值）</option>}
      {accelerationModes[backend]?.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}
    </Select></Field>{checking ? <OperationStatus progress={progress} startedAt={startedAt} fallback="正在检查加速配置" /> : <span role="status">{error ? '检查失败' : status?.status === 'unsupported' ? '当前不可用' : status?.status === 'unverified' ? '待验证' : ''}</span>}</div>
    {!checking && (error ? <DiagnosticNotice summary="加速检查失败" detail={error} retry={retry} /> : mode !== 'none' && status?.message && !reportedErrors.includes(status.message) && <p className={status.status === 'unsupported' ? 'error-text' : 'field-help'}>{status.message}</p>)}
    {mode !== 'none' && !!status?.fields.length && <div className="parameter-grid engine-acceleration-fields">{status.fields.map(field => <div key={field.key}><ParameterInput field={field} location={location} browseDirectories={field.key === 'draft_model'} value={parameters[field.key]} onChange={value => onChange(field.key, value)} />{field.description && <p className="field-help">{field.description}</p>}</div>)}</div>}
  </section>
}

function GpuChoices({ fieldKey, devices, value, onChange }: { fieldKey: string; devices: Record<string, unknown>[]; value: unknown; onChange: (value: string | undefined) => void }) {
  const selected = (Array.isArray(value) ? value : String(value ?? '').split(',')).map(item => String(item).trim()).filter(Boolean)
  const choices = devices.map(gpu => ({ gpu, token: fieldKey === 'devices' ? gpu.device : fieldKey === 'gpu' ? gpu.index : gpu.gpu_id ?? gpu.id }))
    .filter(({ token }) => token != null && (fieldKey !== 'cuda_visible_devices' || /^(GPU|MIG)-/.test(String(token))))
  if (!choices.length) return null
  return <div className="gpu-device-choices" aria-label="本机 GPU">{choices.map(({ gpu, token }) => {
    const id = String(token), checked = selected.includes(id)
    return <button type="button" key={gpuKey(gpu)} aria-pressed={checked} title={id} onClick={() => {
      const next = checked ? selected.filter(item => item !== id) : [...selected.filter(item => item !== 'none' && item !== '-1'), id]
      onChange(next.length ? next.join(',') : undefined)
    }}>{gpuLabel(gpu)}{typeof gpu.memory_total === 'number' && <span>{formatBytes(gpu.memory_total)}</span>}</button>
  })}</div>
}

function ParameterInput({ field, value, location, browseDirectories, defaultText: defaultOverride, gpuEdited = false, onChange }: { field: ParameterField; value: unknown; location?: ModelLocation; browseDirectories?: boolean; defaultText?: string; gpuEdited?: boolean; onChange: (value: unknown) => void }) {
  const defaultText = defaultOverride ?? parameterDefaultText(field)
  if (value === null || value === '') value = undefined
  if (field.key === 'gpu_layers') return <GpuLayersInput field={field} value={value} edited={gpuEdited} onChange={onChange} />
  if (field.type === 'number') return <NumericParameter field={field} value={value} placeholder={defaultText} onChange={onChange} />
  const options = field.type === 'select' ? field.options.map(option => typeof option === 'object' && option !== null ? option as { value: unknown; label?: string } : { value: option, label: String(option) }) : []
  const savedOption = value !== undefined && value !== '' && !options.some(option => String(option.value) === String(value))
  return <Field label={field.label}>
    {field.type === 'boolean' ? <Select className={value === undefined ? 'parameter-default' : undefined} value={value === undefined ? '' : String(value)} onChange={event => onChange(event.target.value === '' ? undefined : event.target.value === 'true')}><option value="">{defaultText}</option><option value="true">开启</option><option value="false">关闭</option></Select>
      : field.type === 'select' ? <Select className={value === undefined ? 'parameter-default' : undefined} value={value === undefined ? '' : String(value)} onChange={event => { const selected = options.find(item => String(item.value) === event.target.value); onChange(event.target.value === '' ? undefined : selected?.value ?? event.target.value) }}><option value="">{defaultText}</option>{savedOption && <option value={String(value)}>{String(value)}（已保存值）</option>}{options.map(item => <option key={String(item.value)} value={String(item.value)}>{item.label || String(item.value)}</option>)}</Select>
        : field.type === 'file' ? <PlatformPathInput label={field.label} location={location} kind={!browseDirectories && ['tokenizer', 'pack', 'tokenizer_directory'].includes(field.key) ? 'directory' : 'file'} allowDirectories={browseDirectories} value={value === undefined ? '' : String(value)} placeholder={defaultText} onChange={path => onChange(path || undefined)} />
          : <input type="text" value={value === undefined ? '' : String(value)} placeholder={defaultText} onChange={event => onChange(event.target.value === '' ? undefined : event.target.value)} />}
  </Field>
}

function RemoveModel({ model, onClose, ...props }: PageProps & { model: ModelAsset; onClose: () => void }) {
  const [deleteFiles, setDeleteFiles] = useState(false)
  const [preview, setPreview] = useState<{ files: string[]; total_bytes: number } | null>(null)
  const [previewError, setPreviewError] = useState('')
  const [previewBusy, setPreviewBusy] = useState(false)
  const action = useAction(props.notify, props.refresh)
  async function toggle(value: boolean) {
    setDeleteFiles(value)
    if (!value || preview) return
    setPreviewBusy(true); setPreviewError('')
    try { setPreview(await request('POST', `/api/models/${model.id}/delete-preview`)) }
    catch (error) { setPreviewError(errorMessage(error)) }
    finally { setPreviewBusy(false) }
  }
  async function remove() {
    const result = await action.run(() => request('DELETE', `/api/models/${model.id}${deleteFiles ? '?delete_files=true' : ''}`), deleteFiles ? '模型登记及文件已删除' : '模型登记已移除')
    if (result.ok) onClose()
  }
  return <Modal title="移除模型？" description={`「${model.name}」及其加载配置将从模型库移除。`} onClose={onClose} footer={<><Button disabled={action.busy} onClick={onClose}>取消</Button><Button variant="danger" busy={action.busy} icon={Trash2} disabled={deleteFiles && (!preview || previewBusy || !!previewError)} onClick={() => void remove()}>{deleteFiles ? '删除登记及模型文件' : '仅移除登记'}</Button></>}><div className="confirmation-path mono">{model.path}</div>{model.location?.platform !== 'wsl2' ? <label className="check-option danger-option"><input type="checkbox" checked={deleteFiles} onChange={event => void toggle(event.target.checked)} /><div><strong>同时删除磁盘中的模型文件</strong><span>文件删除无法通过本应用撤销。</span></div></label> : <p className="muted">WSL 中的模型文件保留。</p>}{deleteFiles && <>{previewBusy && <p className="muted">正在读取待删除文件…</p>}{previewError && <ErrorNotice message={previewError} />}{preview && <div className="delete-preview"><span>{preview.files.length} 项文件 · {formatBytes(preview.total_bytes)}</span><div>{preview.files.map(file => <p className="mono" key={file}>{file}</p>)}</div></div>}</>}</Modal>
}


function QuickLoadToggle({ enabled, disabled, onToggle }: { enabled: boolean; disabled: boolean; onToggle: () => Promise<boolean | undefined> }) {
  const [feedback, setFeedback] = useState<'added' | 'removed' | null>(null)
  useEffect(() => {
    if (!feedback) return
    const timer = window.setTimeout(() => setFeedback(null), 500)
    return () => window.clearTimeout(timer)
  }, [feedback])
  return <button type="button" className="icon-button chat-quick-load-toggle" aria-label={enabled ? '从对话快捷加载移除' : '添加到对话快捷加载'} aria-pressed={enabled} disabled={disabled} data-feedback={feedback || undefined}
    onClick={async () => { const next = enabled ? 'removed' : 'added'; setFeedback(null); if (await onToggle()) setFeedback(next) }}>
    <Zap className="quick-load-bolt" size={16} aria-hidden="true" />
    <span className="quick-load-confirm" aria-hidden="true">{feedback === 'removed' ? <Minus size={18} /> : <Check size={18} />}</span>
    <span className="quick-load-confirm-ring" aria-hidden="true" />
  </button>
}
