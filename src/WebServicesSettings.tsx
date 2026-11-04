import { useEffect, useRef, useState, type InputHTMLAttributes, type KeyboardEvent } from 'react'
import { ArrowDown, ArrowUp, Globe2, Plus, ShieldCheck, X } from 'lucide-react'
import type { PageProps, WebServiceCapability, WebServiceDefinition } from './types'
import type { WebReadResult, WebSearchResult } from './web-search'
import { errorMessage, requestProgress } from './api'
import { Badge, Button, Field, IconButton, Select } from './components'
import CredentialField, { type CredentialFieldHandle } from './CredentialField'
import './web-services-settings.css'

const webTextSettingDefaults = {
  searxng_url: '', searxng_image_engine: '',
  web_search_provider: 'searxng', web_image_provider: 'searxng', web_fetch_provider: 'local',
  deepseek_url: '', tavily_url: '', brave_search_url: '', serper_url: '', saucenao_url: '',
  openai_search_url: '', openai_search_model: '', openai_image_model: '', openai_read_model: '',
}
export const webSettingDefaults = {
  ...webTextSettingDefaults,
  web_search_fallbacks: [] as string[], web_image_fallbacks: [] as string[], web_fetch_fallbacks: [] as string[],
}
export type WebSettingKey = keyof typeof webSettingDefaults
export type WebTextSettingKey = keyof typeof webTextSettingDefaults
export type WebSettingsDraft = typeof webSettingDefaults
const purposes = [
  { field: 'web_search_provider', fallbackField: 'web_search_fallbacks', capability: 'search', label: '网页搜索' },
  { field: 'web_image_provider', fallbackField: 'web_image_fallbacks', capability: 'images', label: '关键词搜图' },
  { field: 'web_fetch_provider', fallbackField: 'web_fetch_fallbacks', capability: 'fetch', label: '网页读取' },
] as const
export type WebRouteSettingKey = typeof purposes[number]['field' | 'fallbackField']
export type WebRouteUpdate = Partial<Pick<WebSettingsDraft, WebRouteSettingKey>>
export const webRouteSettingKeys = purposes.flatMap(purpose => [purpose.field, purpose.fallbackField])
const capabilityNames: Record<WebServiceCapability, string> = { search: '网页搜索', images: '关键词搜图', fetch: '网页读取', reverse: '以图搜源' }
const openaiModelFields = { search: 'openai_search_model', images: 'openai_image_model', fetch: 'openai_read_model' } as const
const reverseImageTest = {
  image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=',
  filename: 'modloader-test.png',
}
type ReverseImageTestResult = { provider: string; elapsed_ms: number; matches: unknown[]; quota?: unknown }
type TestState = { busy: boolean; message: string; error?: boolean }

type Props = PageProps & {
  draft: WebSettingsDraft
  textProps: (field: WebTextSettingKey) => InputHTMLAttributes<HTMLInputElement>
  onRouteChange: (update: WebRouteUpdate) => void
  routeBusy: boolean
  beforeTest: (fields: WebTextSettingKey[]) => Promise<boolean>
}

function serviceLabel(service: WebServiceDefinition, capability: WebServiceCapability) {
  return capability === 'fetch' ? service.fetch_label || service.label : service.label
}

export default function WebServicesSettings({ draft, textProps, onRouteChange, routeBusy, beforeTest, ...props }: Props) {
  const [selected, setSelected] = useState('searxng')
  const [tests, setTests] = useState<Record<string, TestState>>({})
  const controllers = useRef(new Map<string, AbortController>())
  const tabs = useRef<HTMLDivElement>(null)
  const credentials = useRef(new Map<string, CredentialFieldHandle>())
  const services = props.state.settings.web_services ?? []
  const configurable = services.filter(service => service.url_setting)
  useEffect(() => () => controllers.current.forEach(controller => controller.abort()), [])
  if (!services.length) return null

  function navigateTabs(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const next = event.key === 'ArrowRight' ? (index + 1) % configurable.length
      : event.key === 'ArrowLeft' ? (index + configurable.length - 1) % configurable.length
        : event.key === 'Home' ? 0 : event.key === 'End' ? configurable.length - 1 : null
    if (next === null) return
    event.preventDefault()
    setSelected(configurable[next].id)
    tabs.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }

  async function testService(service: WebServiceDefinition, capability: WebServiceCapability) {
    if (controllers.current.has(service.id)) return
    const controller = new AbortController()
    controllers.current.set(service.id, controller)
    setTests(previous => ({ ...previous, [service.id]: { busy: true, message: `正在测试${capabilityNames[capability]}…` } }))
    try {
      const fields = [service.url_setting as WebTextSettingKey,
        ...(service.id === 'searxng' ? ['searxng_image_engine' as const] : []),
        ...(service.id === 'openai' && capability !== 'reverse' ? [openaiModelFields[capability]] : [])]
      if (!await beforeTest(fields)) throw new Error('请先修正并保存此渠道的配置，再重试测试。')
      if (service.credential_key && !await credentials.current.get(service.id)?.commit()) throw new Error('API Key 保存失败，请修正后重试测试。')
      if (controller.signal.aborted) return
      const result = capability === 'reverse'
        ? await requestProgress<ReverseImageTestResult>('POST', '/api/web-reverse-image', reverseImageTest, { signal: controller.signal })
        : await requestProgress<WebSearchResult | WebReadResult>('POST', '/api/web-services/test', {
          provider: service.id, capability,
          ...(capability === 'fetch' ? { url: 'https://docs.python.org/3/library/pathlib.html' } : { query: 'Python pathlib documentation' }),
        }, { signal: controller.signal })
      const isReverse = 'matches' in result
      const summary = !isReverse && 'kind' in result && result.kind === 'summary'
      const name = isReverse ? capabilityNames.reverse : summary ? '阅读摘要' : capabilityNames[capability]
      const detail = isReverse ? `${result.matches.length} 条匹配`
        : summary ? `${result.sources.length} 个来源`
          : 'sources' in result ? `${capability === 'images' ? ('images' in result ? result.images?.length ?? 0 : 0) : result.sources.length} 条结果` : `${result.total_lines} 行正文`
      if (!controller.signal.aborted) setTests(previous => ({ ...previous, [service.id]: {
        busy: false, message: `测试成功 · ${name} · ${detail} · ${(result.elapsed_ms / 1000).toLocaleString('zh-CN', { maximumFractionDigits: 2 })} s`,
      } }))
    } catch (issue) {
      if (!controller.signal.aborted) setTests(previous => ({ ...previous, [service.id]: { busy: false, error: true, message: errorMessage(issue) } }))
    } finally { controllers.current.delete(service.id) }
  }

  return <section className="panel settings-card web-services-card" aria-label="联网服务">
    <div className="settings-card-heading"><span className="settings-icon"><Globe2 size={18} /></span><div><h3>联网服务</h3></div><Badge><ShieldCheck size={11} />本机加密保存</Badge></div>
      <fieldset className="settings-card-body web-channel-routing" aria-label="高级路由" disabled={routeBusy}>
        <div className="web-channel-routes">
          {purposes.map(({ field, fallbackField, capability, label }) => {
          const available = services.filter(service => service.capabilities.includes(capability))
          const fallbacks = draft[fallbackField]
          return <div key={field} className="web-channel-route">
            <div className="web-route-heading"><h4>{label}</h4><IconButton icon={Plus} label={`添加${label}备用渠道`} onClick={() => onRouteChange({ [fallbackField]: [...fallbacks, ''] })} /></div>
              <Select aria-label={label} value={draft[field]} onChange={event => {
                const value = event.target.value
                onRouteChange({ [field]: value, [fallbackField]: fallbacks.filter(provider => provider !== value) })
                if (configurable.some(service => service.id === value)) setSelected(value)
              }}>{available.map(service => <option key={service.id} value={service.id}>{serviceLabel(service, capability)}</option>)}</Select>
            {fallbacks.length > 0 && <ol className="web-fallback-list" aria-label={`${label}备用顺序`}>
              {fallbacks.map((provider, index) => {
                const service = available.find(item => item.id === provider)
                const name = service ? serviceLabel(service, capability) : provider || `空渠道 ${index + 1}`
                const choices = available.filter(item => item.id !== draft[field] && (item.id === provider || !fallbacks.includes(item.id)))
                return <li key={index}>
                  <Select className={`web-fallback-select${provider ? ' filled' : ''}`} aria-label={`${label}备用渠道 ${index + 1}`} value={provider} onChange={event => {
                    const next = [...fallbacks]
                    next[index] = event.target.value
                    onRouteChange({ [fallbackField]: next })
                  }}>
                    <option value="">选择渠道</option>
                    {provider && !service && <option value={provider}>{provider}</option>}
                    {choices.map(item => <option key={item.id} value={item.id}>{serviceLabel(item, capability)}</option>)}
                  </Select>
                  <div className="web-fallback-actions">
                    {fallbacks.length > 1 && (index > 0 ? <IconButton icon={ArrowUp} label={`上移${label}备用渠道 ${name}`} onClick={() => {
                      const reordered = [...fallbacks]
                      ;[reordered[index - 1], reordered[index]] = [reordered[index], reordered[index - 1]]
                      onRouteChange({ [fallbackField]: reordered })
                    }} /> : <span className="web-fallback-action-gap" />)}
                    {fallbacks.length > 1 && (index < fallbacks.length - 1 ? <IconButton icon={ArrowDown} label={`下移${label}备用渠道 ${name}`} onClick={() => {
                      const reordered = [...fallbacks]
                      ;[reordered[index], reordered[index + 1]] = [reordered[index + 1], reordered[index]]
                      onRouteChange({ [fallbackField]: reordered })
                    }} /> : <span className="web-fallback-action-gap" />)}
                    <IconButton icon={X} label={`删除${label}备用渠道 ${name}`} onClick={() => onRouteChange({ [fallbackField]: fallbacks.filter((_, position) => position !== index) })} />
                  </div>
                </li>
              })}
            </ol>}
            </div>
          })}
        </div>
      </fieldset>
    <div ref={tabs} className="sub-tabs web-service-tabs" role="tablist" aria-label="渠道配置">
      {configurable.map((service, index) => <button key={service.id} type="button" role="tab" id={`web-service-tab-${service.id}`} aria-controls={`web-service-panel-${service.id}`} aria-selected={service.id === selected} tabIndex={service.id === selected ? 0 : -1} className={service.id === selected ? 'active' : ''} onClick={() => setSelected(service.id)} onKeyDown={event => navigateTabs(event, index)}>{service.label}</button>)}
    </div>
    {configurable.map(service => <div key={service.id} role="tabpanel" id={`web-service-panel-${service.id}`} aria-labelledby={`web-service-tab-${service.id}`} hidden={service.id !== selected} className="settings-card-body web-service-config">
      <Field label={`${service.label} 地址`}><input {...textProps(service.url_setting as WebTextSettingKey)} aria-label={`${service.label} 地址`} disabled={tests[service.id]?.busy} type="url" className="mono" /></Field>
      {service.id === 'openai' && <div className="web-openai-models">
        <Field label="搜索模型"><input {...textProps('openai_search_model')} aria-label="OpenAI 搜索模型" disabled={tests[service.id]?.busy} className="mono" /></Field>
        <Field label="搜图模型"><input {...textProps('openai_image_model')} aria-label="OpenAI 搜图模型" disabled={tests[service.id]?.busy} className="mono" /></Field>
        <Field label="阅读模型"><input {...textProps('openai_read_model')} aria-label="OpenAI 阅读模型" disabled={tests[service.id]?.busy} className="mono" /></Field>
      </div>}
      {service.id === 'searxng' && <Field label="图片引擎简称" help="bdi 为百度图片；留空使用 SearXNG 默认图片引擎。"><input {...textProps('searxng_image_engine')} disabled={tests[service.id]?.busy} className="mono" placeholder="服务默认" /></Field>}
      {service.credential_key && <CredentialField ref={handle => { if (handle) credentials.current.set(service.id, handle); else credentials.current.delete(service.id) }} disabled={tests[service.id]?.busy} source={service.credential_key} label={`${service.label} API Key`} kind="API Key" configured={Boolean(props.state.settings.web_credentials?.[service.credential_key])} clearDescription="使用此凭据的联网服务将无法继续调用。" {...props} />}
      <div className="web-service-tests">{service.capabilities.map(capability => <Button key={capability} small variant="ghost" aria-label={`测试 ${service.label} ${capabilityNames[capability]}`} disabled={tests[service.id]?.busy} onClick={() => void testService(service, capability)}>测试{capability === 'fetch' && service.fetch_label ? '阅读摘要' : capabilityNames[capability]}</Button>)}</div>
      {tests[service.id] && <p className={`web-service-result${tests[service.id].error ? ' error' : ''}`} role={tests[service.id].error ? 'alert' : 'status'}>{tests[service.id].message}</p>}
    </div>)}
  </section>
}
