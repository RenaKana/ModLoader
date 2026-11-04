import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import type { PageProps, RuntimeInstallation, RuntimePlatform } from './types'
import { request } from './api'
import { Button, ErrorNotice, Field, Modal, Select } from './components'
import { useAction } from './hooks'
import { customConfig, customGenerationOptions, defaultCustomGeneration } from './custom-engine'
import { backendPlatforms, platformLabels, runtimeDistribution, runtimePlatform } from './platforms'
import { PlatformPathInput, WslDistributionField } from './WslPlatform'
import './custom-engine.css'

export default function CustomRuntimeDialog({ runtime, platform: initialPlatform = 'windows', onPlatformChange, onClose, onBackendChange, ...props }: PageProps & {
  runtime?: RuntimeInstallation; platform?: RuntimePlatform; onPlatformChange?: (platform: RuntimePlatform) => void; onClose: () => void; onBackendChange?: (backend: string) => void
}) {
  const config = customConfig(runtime)
  const [platform, setPlatform] = useState(runtime ? runtimePlatform(runtime, props.state.backends) : initialPlatform)
  const [distribution, setDistribution] = useState(runtime && runtimePlatform(runtime, props.state.backends) === 'wsl2' ? runtimeDistribution(runtime) : '')
  const [name, setName] = useState(runtime?.name || '')
  const [path, setPath] = useState(runtime?.path || '')
  const [args, setArgs] = useState(config.args?.join('\n') || '')
  const [cwd, setCwd] = useState(config.cwd || '')
  const [env, setEnv] = useState(Object.entries(config.env || {}).map(([key, value]) => `${key}=${value}`).join('\n'))
  const [healthPath, setHealthPath] = useState(config.health_path || '/v1/models')
  const [chatPath, setChatPath] = useState(config.chat_path || '/v1/chat/completions')
  const [apiModel, setApiModel] = useState(config.api_model_name || '{alias}')
  const [generation, setGeneration] = useState(config.generation_parameters || defaultCustomGeneration)
  const [streamUsage, setStreamUsage] = useState(config.stream_usage || false)
  const [error, setError] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const action = useAction((message, tone) => tone === 'error' ? setError(message) : props.notify(message, tone), props.refresh)

  const location = { platform, ...(platform === 'wsl2' ? { distribution } : {}) }
  async function save() {
    setError('')
    const environment: Record<string, string> = {}
    for (const line of env.split(/\r?\n/).filter(line => line.trim())) {
      const separator = line.indexOf('=')
      const key = line.slice(0, separator).trim()
      if (separator < 1 || !key) { setAdvanced(true); setError('环境变量请按每行 NAME=value 填写。'); return }
      if (Object.hasOwn(environment, key)) { setAdvanced(true); setError(`环境变量 ${key} 重复，请保留一项。`); return }
      environment[key] = line.slice(separator + 1)
    }
    const body = { backend: 'custom', platform, name: name.trim(), path: path.trim(), metadata: { ...runtime?.metadata, ...(platform === 'wsl2' ? { distribution } : {}), custom: {
      args: args.split(/\r?\n/).filter(line => line.length > 0), cwd: cwd.trim(), env: environment,
      health_path: healthPath.trim(), chat_path: chatPath.trim(), api_model_name: apiModel,
      generation_parameters: generation, stream_usage: streamUsage,
    } } }
    const result = await action.run(() => request(runtime ? 'PATCH' : 'POST', runtime ? `/api/runtimes/${runtime.id}` : '/api/runtimes/import', body), runtime ? '引擎模板已保存' : '自定义引擎已添加')
    if (result.ok) onClose()
  }

  return <Modal wide className="custom-engine-editor" title={runtime ? '编辑自定义引擎' : '添加自定义引擎'} onClose={onClose} dismissible={!action.busy}
    footer={<><span className="footer-note">{runtime ? `${props.state.profiles.filter(profile => profile.runtime_id === runtime.id).length} 个关联配置下次加载时生效。` : '仅保存模板，加载时运行。'}</span><Button disabled={action.busy} onClick={onClose}>取消</Button><Button variant="primary" busy={action.busy} disabled={!name.trim() || !path.trim() || !args.trim() || platform === 'wsl2' && !distribution} onClick={() => void save()}>{runtime ? '保存模板' : '添加引擎'}</Button></>}>
    {error && <ErrorNotice message={error} />}
    <Field label="运行平台"><Select disabled={action.busy} value={platform} onChange={event => { const value = event.target.value as RuntimePlatform; setPlatform(value); setPath(''); setCwd(''); onPlatformChange?.(value) }}>{Object.entries(platformLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</Select></Field>
    {onBackendChange && <Field label="运行后端"><Select disabled={action.busy} value="custom" onChange={event => onBackendChange(event.target.value)}>{props.state.backends.filter(backend => backendPlatforms(backend).includes(platform)).map(backend => <option key={backend.id} value={backend.id}>{backend.label}</option>)}</Select></Field>}
    {platform === 'wsl2' && <WslDistributionField value={distribution} onChange={value => { setDistribution(value); setPath(''); setCwd('') }} disabled={action.busy} />}
    <Field label="引擎名称"><input autoFocus value={name} disabled={action.busy} onChange={event => setName(event.target.value)} /></Field>
    <Field label="启动程序" help={platform === 'wsl2' ? '所选发行版中的程序或 Python，不填 wsl.exe；需提供 OpenAI 兼容对话接口。' : '使用可信的本机 EXE 或 Python，需提供 OpenAI 兼容对话接口。'}><PlatformPathInput label="启动程序" location={location} kind="file" value={path} disabled={action.busy} onChange={setPath} placeholder={platform === 'wsl2' ? '/opt/my-engine/bin/python' : 'D:\\Engines\\my-engine\\python.exe'} /></Field>
    <Field label="启动参数模板" help="每行一项 argv，选项与值分行；路径不加引号。"><textarea className="mono custom-engine-args" rows={8} spellCheck={false} value={args} disabled={action.busy} onChange={event => { setArgs(event.target.value); setError('') }} placeholder={'--model\n{model_path}\n--host\n{host}\n--port\n{port}\n--ctx-size\n{context_length}'} /></Field>
    <div className="custom-engine-tokens"><span>必填占位符</span><code>{'{model_path}'}</code><code>{'{port}'}</code><span>可选</span><code>{'{host}'}</code><code>{'{context_length}'}</code><code>{'{alias}'}</code><code>{'{model_dir}'}</code></div>
    <p className="field-help">{'{host}'} 为 127.0.0.1，服务应仅监听本机。使用 {'{context_length}'} 时需在加载配置中填写上下文长度。</p>
    <button className="advanced-toggle" aria-expanded={advanced} disabled={action.busy} onClick={() => setAdvanced(!advanced)}><ChevronDown size={15} className={advanced ? 'rotated' : ''} />服务与环境选项</button>
    {advanced && <div className="custom-engine-advanced">
      <Field label="工作目录" help="留空使用启动程序所在目录。"><PlatformPathInput label="工作目录" location={location} kind="directory" value={cwd} disabled={action.busy} onChange={setCwd} placeholder={platform === 'wsl2' ? '/home/user/my-engine' : undefined} /></Field>
      <Field label="环境变量" help="每行 NAME=value；保存在本机数据库，勿填密钥或密码。"><textarea className="mono" rows={3} spellCheck={false} value={env} disabled={action.busy} onChange={event => setEnv(event.target.value)} placeholder="CUDA_VISIBLE_DEVICES=0" /></Field>
      <div className="form-grid"><Field label="就绪检查路径"><input value={healthPath} disabled={action.busy} onChange={event => setHealthPath(event.target.value)} /></Field><Field label="对话接口路径"><input value={chatPath} disabled={action.busy} onChange={event => setChatPath(event.target.value)} /></Field></div>
      <Field label="后端模型名称" help="model 字段；默认沿用 API 名称，可填固定值或 {model_path}。"><input className="mono" value={apiModel} disabled={action.busy} onChange={event => setApiModel(event.target.value)} /></Field>
      <fieldset className="custom-engine-generation"><legend>服务支持的生成参数</legend><div>{customGenerationOptions.map(([key, label]) => <label key={key}><input type="checkbox" disabled={action.busy} checked={generation.includes(key)} onChange={event => setGeneration(previous => event.target.checked ? [...previous, key] : previous.filter(item => item !== key))} />{label}</label>)}</div><p className="field-help">参数值在“生成默认值”中设置。</p></fieldset>
      <label className="custom-engine-check"><input type="checkbox" checked={streamUsage} disabled={action.busy} onChange={event => setStreamUsage(event.target.checked)} />支持流式用量统计（stream_options.include_usage）</label>
    </div>}
  </Modal>
}
