import { useEffect, useRef, useState } from 'react'
import { Check, CheckCircle2, Copy, Eye, EyeOff, FolderOpen, KeyRound, Link2, Loader2, Network, RotateCw, ShieldCheck, X } from 'lucide-react'
import type { PageProps } from '../types'
import { copy, errorMessage, pickPath, request } from '../api'
import { Badge, Button, ErrorNotice, Field, IconButton, Modal, PageHeader } from '../components'
import { Select } from '../components'
import CredentialField from '../CredentialField'
import WebServicesSettings, { webRouteSettingKeys, webSettingDefaults, type WebRouteSettingKey, type WebRouteUpdate, type WebSettingKey, type WebSettingsDraft, type WebTextSettingKey } from '../WebServicesSettings'

type SaveState = 'idle' | 'saving' | 'saved' | 'error'
type EditableSettings = { api_port: string; api_history_enabled: boolean; api_history_directory: string; api_history_retention_days: string; model_directory: string; proxy_mode: string; proxy_url: string; chat_unload_before_load: boolean; chat_tool_details: 'compact' | 'full'; close_to_tray: boolean; font_scale: string } & WebSettingsDraft
const settingKeys = ['api_port', 'api_history_enabled', 'api_history_directory', 'api_history_retention_days', 'model_directory', 'proxy_mode', 'proxy_url', 'close_to_tray', 'chat_unload_before_load', 'chat_tool_details', 'font_scale', ...Object.keys(webSettingDefaults) as WebSettingKey[]] as const
// Leave time for the desktop's 8-second close handshake to report a failure.
const routeSaveTimeoutMs = 6000

function sameSettingValue(left: unknown, right: unknown) {
  if (Array.isArray(left) || Array.isArray(right)) return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index])
  return String(left) === String(right)
}

export default function Settings({ active, ...props }: PageProps & { active: boolean }) {
  const [draft, setDraft] = useState<EditableSettings>({ api_port: '', api_history_enabled: true, api_history_directory: '', api_history_retention_days: '0', model_directory: '', proxy_mode: 'system', proxy_url: '', close_to_tray: true, chat_unload_before_load: true, chat_tool_details: 'compact', font_scale: '1', ...webSettingDefaults })
  const draftRef = useRef(draft)
  draftRef.current = draft
  const dirty = useRef(new Set<keyof EditableSettings>())
  const focused = useRef(new Set<keyof EditableSettings>())
  const savingFields = useRef(new Map<keyof EditableSettings, Promise<boolean>>())
  const routeSaving = useRef<Promise<boolean> | undefined>(undefined)
  const routeNeedsSave = useRef(false)
  const [routeBusy, setRouteBusy] = useState(false)
  const failedFields = useRef(new Map<keyof EditableSettings, { value: unknown; message: string }>())
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [saveError, setSaveError] = useState('')
  const [failedUpdate, setFailedUpdate] = useState<Record<string, unknown> | null>(null)
  const [key, setKey] = useState('')
  const [keyVisible, setKeyVisible] = useState(false)
  const [keyBusy, setKeyBusy] = useState(false)
  const [editKey, setEditKey] = useState(false)
  const [snippet, setSnippet] = useState('curl')
  const [directoryBusy, setDirectoryBusy] = useState(false)
  const [historyRetrying, setHistoryRetrying] = useState(false)
  const [historyRetryError, setHistoryRetryError] = useState('')

  useEffect(() => {
    setDraft(previous => {
      const next = { ...previous }
      for (const field of settingKeys) {
        const incoming = props.state.settings[field]
        if (incoming === undefined) continue
        const value = Array.isArray(previous[field]) ? (Array.isArray(incoming) ? incoming as string[] : previous[field])
          : (field === 'close_to_tray' || field === 'chat_unload_before_load' || field === 'api_history_enabled') ? Boolean(incoming) : String(incoming)
        if (dirty.current.has(field) && !savingFields.current.has(field) && sameSettingValue(previous[field], value)) dirty.current.delete(field)
        if (!dirty.current.has(field) && !focused.current.has(field) && !savingFields.current.has(field)) (next[field] as string | boolean | string[]) = value
      }
      return next
    })
  }, [props.state.settings])

  function change<K extends keyof EditableSettings>(field: K, value: EditableSettings[K]) {
    dirty.current.add(field)
    const next = { ...draftRef.current, [field]: value }
    draftRef.current = next
    setDraft(next)
  }

  function save(update: Record<string, unknown>): Promise<boolean> {
    setSaveState('saving'); setSaveError('')
    const fields = Object.keys(update) as (keyof EditableSettings)[]
    const routeUpdate = fields.some(field => webRouteSettingKeys.includes(field as WebRouteSettingKey))
    for (const field of fields) failedFields.current.delete(field)
    const pending: Promise<boolean> = Promise.resolve().then(async () => {
      try {
        await request('PATCH', '/api/settings', update, routeUpdate ? { signal: AbortSignal.timeout(routeSaveTimeoutMs) } : {})
        for (const field of fields) if (savingFields.current.get(field) === pending) {
          failedFields.current.delete(field)
          // Keep route drafts until refreshed settings acknowledge the latest order.
          if (!webRouteSettingKeys.includes(field as WebRouteSettingKey) && sameSettingValue(draftRef.current[field], update[field])) dirty.current.delete(field)
        }
        // Persistence is complete at the PATCH response; telemetry refresh must
        // not hold the close handshake open after a successful route save.
        if (routeUpdate) void props.refresh()
        else await props.refresh()
        return true
      } catch (error) {
        const message = error instanceof DOMException && error.name === 'TimeoutError' ? '路由保存超时，请重试。' : errorMessage(error)
        for (const field of fields) if (savingFields.current.get(field) === pending) failedFields.current.set(field, { value: update[field], message })
        return false
      }
    })
    for (const field of fields) savingFields.current.set(field, pending)
    void pending.finally(() => {
      for (const field of fields) if (savingFields.current.get(field) === pending) savingFields.current.delete(field)
      const failures = [...failedFields.current]
      setSaveError([...new Set(failures.map(([, failure]) => failure.message))].join('；'))
      setFailedUpdate(failures.length ? Object.fromEntries(failures.map(([field, failure]) => [field, failure.value])) : null)
      setSaveState(savingFields.current.size ? 'saving' : failures.length ? 'error' : 'saved')
    })
    return pending
  }

  function changeWebRoute(update: WebRouteUpdate) {
    for (const field of Object.keys(update) as WebRouteSettingKey[]) dirty.current.add(field)
    const next = { ...draftRef.current, ...update }
    draftRef.current = next
    setDraft(next)
    routeNeedsSave.current = true
    if (!savingFields.current.size && !failedFields.current.size) setSaveState('idle')
  }

  function flushWebRoutes(): Promise<boolean> {
    if (routeSaving.current) return routeSaving.current
    if (!routeNeedsSave.current) return Promise.resolve(true)
    const update = Object.fromEntries(webRouteSettingKeys.filter(field => dirty.current.has(field)).map(field => {
      const value = draftRef.current[field]
      return [field, Array.isArray(value) ? value.filter(provider => provider !== '') : value]
    })) as WebRouteUpdate
    if (!Object.keys(update).length) { routeNeedsSave.current = false; return Promise.resolve(true) }
    setRouteBusy(true)
    const pending = save(update).then(success => {
      if (success) {
        // The fieldset is disabled while committing. Keep other settings edits.
        const next = { ...draftRef.current, ...update }
        draftRef.current = next
        setDraft(next)
        routeNeedsSave.current = false
      }
      return success
    }).finally(() => {
      routeSaving.current = undefined
      setRouteBusy(false)
    })
    routeSaving.current = pending
    return pending
  }

  const flushRoutes = useRef(flushWebRoutes)
  flushRoutes.current = flushWebRoutes
  useEffect(() => {
    if (!active) void flushRoutes.current().then(success => {
      if (!success) props.notify('路由保存失败，请返回设置重试。', 'error')
    })
  }, [active, props.notify])
  useEffect(() => {
    const beforeClose = window.modloader?.onBeforeClose(async () => {
      if (!await flushRoutes.current()) throw new Error('路由保存失败')
    })
    const hidden = window.modloader?.onWindowHidden?.(() => {
      void flushRoutes.current().then(success => {
        if (!success) props.notify('路由保存失败，请返回设置重试。', 'error')
      })
    })
    return () => { beforeClose?.(); hidden?.() }
  }, [props.notify])

  function retryFailedSettings() {
    if (!failedUpdate) return
    if (Object.keys(failedUpdate).some(field => webRouteSettingKeys.includes(field as WebRouteSettingKey))) void flushWebRoutes()
    const otherFields = Object.fromEntries(Object.entries(failedUpdate).filter(([field]) => !webRouteSettingKeys.includes(field as WebRouteSettingKey)))
    if (Object.keys(otherFields).length) void save(otherFields)
  }

  async function waitForWebSettings(fields: WebTextSettingKey[]) {
    const results = await Promise.all(fields.map(field => savingFields.current.get(field)))
    return results.every(result => result !== false) && fields.every(field => !dirty.current.has(field))
  }

  function blur(field: keyof EditableSettings) {
    focused.current.delete(field)
    if (!dirty.current.has(field)) return
    const value = draftRef.current[field]
    if (field === 'api_port') {
      const port = Number(value)
      if (!Number.isInteger(port) || port < 1 || port > 65535) { setSaveError('端口须为 1–65535 的整数。'); setSaveState('error'); setFailedUpdate(null); return }
      void save({ api_port: port })
    } else if ((field === 'model_directory' || field === 'api_history_directory') && !String(value).trim()) {
      setSaveError(field === 'model_directory' ? '模型目录不能为空。' : '记录目录不能为空。'); setSaveState('error'); setFailedUpdate(null)
    } else void save({ [field]: value })
  }

  function textProps(field: 'api_port' | 'api_history_directory' | 'model_directory' | 'proxy_url' | WebTextSettingKey) {
    return { value: draft[field], onChange: (event: React.ChangeEvent<HTMLInputElement>) => change(field, event.target.value), onFocus: () => focused.current.add(field), onBlur: () => blur(field), onKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => { if (event.key === 'Enter') event.currentTarget.blur() } }
  }

  async function chooseDirectory(field: 'model_directory' | 'api_history_directory' = 'model_directory') {
    setDirectoryBusy(true)
    try { const result = await pickPath('directory'); if (result) { change(field, result); await save({ [field]: result }) } }
    catch (error) { props.notify(errorMessage(error), 'error') }
    finally { setDirectoryBusy(false) }
  }

  async function retryHistory() {
    setHistoryRetrying(true); setHistoryRetryError('')
    try { await request('POST', '/api/request-history/retry'); await props.refresh() }
    catch (error) { setHistoryRetryError(errorMessage(error)) }
    finally { setHistoryRetrying(false) }
  }

  async function getKey() {
    if (key) return key
    const result = await request<{ key: string }>('GET', '/api/settings/api-key')
    setKey(result.key)
    return result.key
  }

  async function revealKey() {
    if (keyVisible) { setKeyVisible(false); return }
    setKeyBusy(true)
    try { await getKey(); setKeyVisible(true) }
    catch (error) { props.notify(errorMessage(error), 'error') }
    finally { setKeyBusy(false) }
  }

  async function copyKey() {
    setKeyBusy(true)
    try { await copy(await getKey()); props.notify('API Key 已复制') }
    catch (error) { props.notify(errorMessage(error), 'error') }
    finally { setKeyBusy(false) }
  }

  const visibleSaveState = routeNeedsSave.current && saveState === 'saved' ? 'idle' : saveState
  const port = props.state.settings.api_port || 12340
  const apiUrl = String(props.state.settings.api_url || `http://127.0.0.1:${port}/v1`)
  const historyStatus = props.state.settings.api_history_status
  const alias = props.state.instances.find(instance => instance.state === 'ready')?.alias || props.state.profiles[0]?.alias || 'your-model-name'
  const samples: Record<string, string> = {
    curl: `curl ${apiUrl}/chat/completions \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"${alias}","messages":[{"role":"user","content":"你好"}],"stream":true}'`,
    python: `from openai import OpenAI\n\nclient = OpenAI(\n    base_url="${apiUrl}",\n    api_key="YOUR_API_KEY",\n)\n\nresponse = client.chat.completions.create(\n    model="${alias}",\n    messages=[{"role": "user", "content": "你好"}],\n)\nprint(response.choices[0].message.content)`,
  }

  return <>
    <PageHeader eyebrow="WORKSPACE PREFERENCES" title="设置"><div className={`save-indicator ${visibleSaveState}`} role="status">{visibleSaveState === 'saving' ? <Loader2 size={14} className="spin" /> : visibleSaveState === 'saved' ? <CheckCircle2 size={14} /> : null}{visibleSaveState === 'saving' ? '正在保存' : visibleSaveState === 'saved' ? '已保存' : visibleSaveState === 'error' ? '保存失败' : '自动保存'}</div></PageHeader>
    {saveError && <ErrorNotice message={saveError} retry={failedUpdate ? retryFailedSettings : undefined} />}
    <div className="settings-layout"><div className="settings-main">
      <div className="panel settings-card"><div className="settings-card-heading"><span className="settings-icon"><Link2 size={18} /></span><div><h3>本地 API</h3><p>OpenAI Chat Completions</p></div></div><div className="settings-card-body"><Field label="Base URL"><div className="copyable-value"><code>{apiUrl}</code><IconButton icon={Copy} label="复制 Base URL" onClick={() => void copy(apiUrl).then(() => props.notify('Base URL 已复制')).catch(error => props.notify(errorMessage(error), 'error'))} /></div></Field><div className="settings-two-cols"><Field label="API 端口" help="重启应用生效。"><input {...textProps('api_port')} inputMode="numeric" placeholder="12340" /></Field><Field label="监听地址"><div className="read-only-field mono">127.0.0.1 <ShieldCheck size={14} /></div></Field></div>
        {typeof props.state.settings.api_error === 'string' && props.state.settings.api_error && <div className="warning-note">{props.state.settings.api_error}</div>}
        <Field as="div" label="API Key"><div className="api-key-control"><code>{keyVisible && key ? key : '••••••••••••••••••••••••'}</code><IconButton icon={keyVisible ? EyeOff : Eye} label={keyVisible ? '隐藏 API Key' : '查看 API Key'} disabled={keyBusy} onClick={() => void revealKey()} /><IconButton icon={Copy} label="复制 API Key" disabled={keyBusy} onClick={() => void copyKey()} /><Button small onClick={() => setEditKey(true)}>更换</Button></div></Field>
        <section className="api-history-settings" aria-label="调用记录设置">
        <div className="setting-toggle-row"><div><strong>保存调用记录</strong></div><button role="switch" aria-checked={draft.api_history_enabled} aria-label="保存调用记录" className={`toggle ${draft.api_history_enabled ? 'on' : ''}`} disabled={saveState === 'saving' || historyRetrying} onClick={() => { const value = !draft.api_history_enabled; change('api_history_enabled', value); void save({ api_history_enabled: value }) }}><span /></button></div>
        <Field className="api-history-retention" label="保留时间"><Select aria-label="调用记录保留时间" value={draft.api_history_retention_days} disabled={saveState === 'saving' || historyRetrying} onChange={event => { change('api_history_retention_days', event.target.value); void save({ api_history_retention_days: Number(event.target.value) }) }}><option value="7">7 天</option><option value="30">30 天</option><option value="0">永久</option></Select></Field>
        <Field label="记录目录"><div className="input-with-button"><input {...textProps('api_history_directory')} className="mono" disabled={saveState === 'saving' || historyRetrying} placeholder="等待目录信息" /><IconButton icon={FolderOpen} label="浏览记录目录" disabled={directoryBusy || saveState === 'saving' || historyRetrying} onClick={() => void chooseDirectory('api_history_directory')} /></div></Field>
        <div className="setting-toggle-row" role="status" aria-label="调用记录保存状态"><div><span>{historyStatus?.state === 'saving' ? '正在写入调用记录' : historyStatus?.state === 'error' ? '调用记录保存失败' : historyStatus?.state === 'saved' ? '调用记录已保存' : '仅保留本次运行记录'}</span></div>{(historyStatus?.state === 'error' || historyRetryError || historyStatus?.cleanup_error) && <Button small busy={historyRetrying} disabled={saveState === 'saving'} onClick={() => void retryHistory()}>{historyStatus?.state === 'error' || historyRetryError ? '重试写入' : '重试清理'}</Button>}</div>
        {(historyRetryError || historyStatus?.error) && <ErrorNotice message={historyRetryError || historyStatus!.error!} />}
        {historyStatus?.cleanup_error && <ErrorNotice message={historyStatus.cleanup_error} />}
        </section>
      </div></div>
      <div className="panel settings-card"><div className="settings-card-heading"><span className="settings-icon"><FolderOpen size={18} /></span><div><h3>文件与应用</h3></div></div><div className="settings-card-body"><Field label="模型下载目录" help="仅用于新下载，已有模型位置不变。"><div className="input-with-button"><input {...textProps('model_directory')} className="mono" placeholder="等待目录信息" /><IconButton icon={FolderOpen} label="浏览下载目录" disabled={directoryBusy} aria-busy={directoryBusy} onClick={() => void chooseDirectory()} /></div></Field><div className="font-setting-row"><div><strong>文字大小</strong></div><Select aria-label="文字大小" value={draft.font_scale} disabled={saveState === 'saving'} onChange={event => { change('font_scale', event.target.value); void save({ font_scale: Number(event.target.value) }) }}><option value="0.9">90%</option><option value="1">100%</option><option value="1.15">115%</option><option value="1.3">130%</option>{!['0.9', '1', '1.15', '1.3'].includes(draft.font_scale) && <option value={draft.font_scale}>{Math.round(Number(draft.font_scale) * 100)}%</option>}</Select></div><div className="setting-toggle-row"><div><strong>关闭到系统托盘</strong><span>模型继续运行</span></div><button role="switch" aria-checked={draft.close_to_tray} aria-label="关闭到系统托盘" className={`toggle ${draft.close_to_tray ? 'on' : ''}`} disabled={saveState === 'saving'} onClick={() => { const value = !draft.close_to_tray; change('close_to_tray', value); void save({ close_to_tray: value }) }}><span /></button></div></div></div>
      <div className="panel settings-card"><div className="settings-card-heading"><span className="settings-icon"><Network size={18} /></span><div><h3>网络代理</h3><p>搜索、下载与环境安装</p></div></div><div className="settings-card-body"><div className="proxy-options">{[{ value: 'system', title: '系统代理' }, { value: 'direct', title: '直接连接' }, { value: 'manual', title: '手动代理' }].map(mode => <label className={draft.proxy_mode === mode.value ? 'selected' : ''} key={mode.value}><input type="radio" name="proxy_mode" value={mode.value} checked={draft.proxy_mode === mode.value} disabled={saveState === 'saving'} onChange={() => { change('proxy_mode', mode.value); void save({ proxy_mode: mode.value }) }} /><strong>{mode.title}</strong></label>)}</div>{draft.proxy_mode === 'manual' && <Field label="代理地址" help="本地 HTTP(S) 地址，不含用户名和密码。"><input {...textProps('proxy_url')} className="mono" placeholder="http://127.0.0.1:7890" /></Field>}</div></div>
      <section className="panel settings-card chat-settings-card" aria-label="对话">
        <div className="settings-card-heading"><h3>对话</h3></div>
        <div className="settings-card-body">
          <div className="chat-setting-row"><strong>快捷加载前卸载当前实例</strong><button role="switch" aria-checked={draft.chat_unload_before_load} aria-label="快捷加载前卸载当前实例" className={`toggle ${draft.chat_unload_before_load ? 'on' : ''}`} disabled={saveState === 'saving'} onClick={() => { const value = !draft.chat_unload_before_load; change('chat_unload_before_load', value); void save({ chat_unload_before_load: value }) }}><span /></button></div>
          <Field className="chat-setting-row" label="工具详情"><Select aria-label="工具详情" value={draft.chat_tool_details} onChange={event => { const value = event.target.value as 'compact' | 'full'; change('chat_tool_details', value); void save({ chat_tool_details: value }) }}><option value="compact">简洁</option><option value="full">完整</option></Select></Field>
        </div>
      </section>
      <WebServicesSettings {...props} beforeTest={waitForWebSettings} draft={draft} textProps={textProps} onRouteChange={changeWebRoute} routeBusy={routeBusy} />
      <div className="panel settings-card"><div className="settings-card-heading"><span className="settings-icon"><KeyRound size={18} /></span><div><h3>模型源凭据</h3></div><Badge><ShieldCheck size={11} />本机加密保存</Badge></div><div className="settings-card-body credentials-grid"><CredentialField source="hf" label="Hugging Face Token" configured={Boolean(props.state.settings.hf_configured || props.state.settings.hf_token_set)} {...props} /><CredentialField source="modelscope" label="ModelScope Token" configured={Boolean(props.state.settings.modelscope_configured || props.state.settings.modelscope_token_set)} {...props} /></div></div>
    </div><aside className="settings-aside"><div className="panel integration-card"><div className="panel-heading"><div><h3>API 示例</h3></div></div><div className="sub-tabs compact"><button className={snippet === 'curl' ? 'active' : ''} onClick={() => setSnippet('curl')}>cURL</button><button className={snippet === 'python' ? 'active' : ''} onClick={() => setSnippet('python')}>Python</button><IconButton icon={Copy} label="复制接入示例" onClick={() => void copy(samples[snippet]).then(() => props.notify('接入示例已复制')).catch(error => props.notify(errorMessage(error), 'error'))} /></div><pre className="integration-code">{samples[snippet]}</pre></div></aside></div>
    {editKey && <ChangeApiKey onClose={() => setEditKey(false)} onSaved={() => { setKey(''); setKeyVisible(false) }} {...props} />}
  </>
}


function ChangeApiKey({ onClose, onSaved, ...props }: PageProps & { onClose: () => void; onSaved: () => void }) {
  const [value, setValue] = useState('')
  const [visible, setVisible] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  async function save() {
    setBusy(true); setError('')
    try { await request('PATCH', '/api/settings', { api_key: value }); onSaved(); await props.refresh(); props.notify('API Key 已更换'); onClose() }
    catch (issue) { setError(errorMessage(issue)) }
    finally { setBusy(false) }
  }
  return <Modal title="更换本地 API Key" description="更换后，使用旧密钥的客户端需更新配置。" onClose={onClose} footer={<><Button onClick={onClose}>取消</Button><Button variant="primary" busy={busy} disabled={!value.trim()} onClick={() => void save()}>更换密钥</Button></>}><Field label="新 API Key"><div className="input-with-button"><input autoFocus type={visible ? 'text' : 'password'} autoComplete="off" value={value} onChange={event => setValue(event.target.value)} placeholder="输入新密钥" /><IconButton icon={visible ? EyeOff : Eye} label={visible ? '隐藏新密钥' : '显示新密钥'} onClick={() => setVisible(!visible)} /></div></Field>{error && <ErrorNotice message={error} />}</Modal>
}
