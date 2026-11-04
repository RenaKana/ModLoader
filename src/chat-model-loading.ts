import { useEffect, useRef, useState } from 'react'
import { errorMessage, isActive, isJobActive, isReady, request, requestProgress } from './api'
import type { OperationProgress } from './OperationStatus'
import type { AppState, RuntimeInstance } from './types'

export type ChatQuickLoadOption = {
  value: string
  label: string
  detail: string
  status: string
  loading?: boolean
  disabled?: boolean
  error?: string
}
type ProfileLoad = { pending?: boolean; progress?: OperationProgress; instance?: RuntimeInstance; error?: string }

export function chatQuickLoadOptions(state: AppState, loads: Record<string, ProfileLoad> = {}): ChatQuickLoadOption[] {
  return state.profiles.filter(profile => profile.chat_quick_load).flatMap(profile => {
    const instances = state.instances.filter(item => item.profile_id === profile.id)
    const load = loads[profile.id]
    const active = instances.find(item => isActive(item.state)) || load?.instance
    const model = state.models.find(item => item.id === profile.model_id)
    const runtime = state.runtimes.find(item => item.id === profile.runtime_id)
    const migrating = state.migrations?.some(job => job.request?.model_id === profile.model_id && isJobActive(job))
    const busy = !!load?.pending || !!active && isActive(active.state) && !isReady(active.state)
    const last = instances.at(-1)
    return [{ value: profile.id, label: profile.name, detail: state.backends.find(item => item.id === profile.backend)?.label || profile.backend,
      status: active && isReady(active.state) && !load?.pending ? '已加载' : load?.pending ? load.progress?.message || '检查启动条件'
        : active?.state === 'stopping' ? '正在停止'
          : active?.state === 'preparing' ? '准备运行文件'
            : busy ? '加载中' : !model ? '模型不存在' : !runtime ? '运行环境不存在' : migrating ? '模型正在迁移'
              : load?.error || last?.state === 'failed' ? '重新加载' : '加载',
      loading: busy && active?.state !== 'stopping', disabled: busy || !model || !runtime || !!migrating,
      error: !busy ? load?.error || (last?.state === 'failed' ? last.error || last.message || '加载失败' : undefined) : undefined,
    }]
  })
}

export function useChatModelLoading(state: AppState, refresh: () => Promise<void>, onReady: (id: string) => void, resetKey?: string, selectedInstanceId?: string) {
  const [loads, setLoads] = useState<Record<string, ProfileLoad>>({})
  const [unloadingIds, setUnloadingIds] = useState<string[]>([])
  const [unloadError, setUnloadError] = useState('')
  const busy = useRef(false)
  const currentResetKey = useRef(resetKey)
  currentResetKey.current = resetKey
  const currentLoads = useRef(loads)
  const target = useRef<{ profileId: string; instanceId?: string; resetKey?: string } | null>(null)
  const onReadyRef = useRef(onReady)
  onReadyRef.current = onReady

  function updateLoads(update: (previous: Record<string, ProfileLoad>) => Record<string, ProfileLoad>) {
    currentLoads.current = update(currentLoads.current)
    setLoads(currentLoads.current)
  }
  function cancelAutoSelect() { target.current = null }

  useEffect(() => {
    const intent = target.current
    if (intent && intent.resetKey !== resetKey) cancelAutoSelect()
    else if (intent?.instanceId) {
      const instance = state.instances.find(item => item.id === intent.instanceId)
      if (instance && isReady(instance.state)) {
        cancelAutoSelect()
        onReadyRef.current(instance.id)
      } else if (instance && !isActive(instance.state)) cancelAutoSelect()
    }
    const completed = Object.entries(currentLoads.current).filter(([, load]) => load.instance && state.instances.some(item => item.id === load.instance!.id))
    if (completed.length) updateLoads(previous => {
      const next = { ...previous }
      for (const [id] of completed) delete next[id]
      return next
    })
  }, [state.instances, resetKey])

  async function unloadInstance(id: string) {
    if (busy.current) return
    busy.current = true
    setUnloadError(''); setUnloadingIds([id])
    try {
      await request<RuntimeInstance>('POST', `/api/instances/${encodeURIComponent(id)}/stop`, {})
      await refresh()
    } catch (error) { setUnloadError(errorMessage(error)) }
    finally { busy.current = false; setUnloadingIds([]) }
  }

  async function loadProfile(profileId: string, parallel = false) {
    if (busy.current) return
    const profile = state.profiles.find(item => item.id === profileId && item.chat_quick_load)
    if (!profile) return
    const active = state.instances.find(item => item.profile_id === profileId && isReady(item.state))
    if (active && !parallel) { cancelAutoSelect(); onReadyRef.current(active.id); return }
    const intent = { profileId, instanceId: undefined as string | undefined, resetKey }
    target.current = intent
    busy.current = true
    setUnloadError('')
    updateLoads(previous => ({ ...previous, [profileId]: { pending: true } }))
    try {
      const selected = state.instances.find(item => item.id === selectedInstanceId && isActive(item.state))
      if (!parallel && state.settings.chat_unload_before_load !== false && selected) {
        setUnloadingIds([selected.id])
        updateLoads(previous => ({ ...previous, [profileId]: { pending: true, progress: { stage: 'stopping', message: '正在卸载当前实例' } } }))
        await request<RuntimeInstance>('POST', `/api/instances/${encodeURIComponent(selected.id)}/stop`, {})
        setUnloadingIds([])
      }
      const instance = await requestProgress<RuntimeInstance>('POST', `/api/profiles/${encodeURIComponent(profileId)}/start`, { allow_parallel: true }, {
        onProgress: progress => updateLoads(previous => ({ ...previous, [profileId]: { pending: true, progress } })),
      })
      intent.instanceId = instance.id
      if (currentResetKey.current !== intent.resetKey && target.current === intent) cancelAutoSelect()
      updateLoads(previous => ({ ...previous, [profileId]: { instance } }))
      await refresh()
    } catch (error) {
      updateLoads(previous => ({ ...previous, [profileId]: { error: errorMessage(error) } }))
      if (target.current === intent) cancelAutoSelect()
    } finally { busy.current = false; setUnloadingIds([]) }
  }

  return { quickLoadOptions: chatQuickLoadOptions(state, loads), loadProfile, cancelAutoSelect, unloadInstance, unloadingIds, unloadError }
}
