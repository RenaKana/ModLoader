import type { LoadProfile, ModelAsset, RuntimeInstance, RuntimeInstallation } from './types'

export type ReasoningEffort = 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max' | number
export interface ReasoningCapability { kind: 'qwen' | 'deepseek'; levels: Array<{ value: ReasoningEffort; label: string }> }

const qwen: ReasoningCapability = { kind: 'qwen', levels: [
  { value: 'none', label: '关闭' }, { value: 'low', label: '低' }, { value: 'medium', label: '中' }, { value: 'xhigh', label: '高' },
] }
const deepseek: ReasoningCapability = { kind: 'deepseek', levels: [
  { value: 'none', label: '关闭' }, { value: 'low', label: '低' }, { value: 'medium', label: '中' }, { value: 'high', label: '高' }, { value: 'max', label: '最高' },
] }

// These are the local Qwen3.8 and DeepSeek41 template contracts, not a claim
// that every model sharing qwen35 or every backend accepts the same levels.
export function chatReasoningCapability(instance?: RuntimeInstance, model?: ModelAsset, runtime?: RuntimeInstallation, profile?: LoadProfile): ReasoningCapability | undefined {
  if (!instance || !model) return
  const architecture = model.architecture || model.metadata['general.architecture']
  if (instance.backend === 'strata') {
    if (architecture === 'deepseek41') return deepseek
    if (architecture === 'qwen4exp') return qwen
  }
  const names = [model.name, model.metadata['general.name'], model.metadata['general.basename']].join(' ')
  const qwen38 = /qwen3[._-]?8\b/i.test(names)
  const flags = runtime?.metadata.supported_flags
  const overriddenTemplate = profile?.extra_args.some(arg => ['--no-jinja', '--chat-template', '--chat-template-file', '--in-prefix', '--in-suffix'].includes(arg.split('=')[0]))
  if (instance.backend === 'llama_cpp' && architecture === 'qwen35' && qwen38
    && Array.isArray(flags) && flags.includes('--reasoning-effort') && flags.includes('--jinja') && !overriddenTemplate) return qwen
  // The verified NInfer API accepts request-level effort even though its CLI has
  // no --reasoning-effort flag. The local Qwen3.8 artifact embeds the same
  // low/medium/xhigh template contract; do not infer it for other qwen3_5_text models.
  if (instance.backend === 'ninfer' && qwen38 && !overriddenTemplate) {
    if (model.format === 'ninfer' && architecture === 'qwen3_5_text') return qwen
    // Loading an HF source keeps its original ModelAsset after preparation.
    const recipe = profile?.parameters.conversion_recipe
    if (model.format === 'safetensors' && ['qwen3_5', 'qwen3_5_text', 'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration'].includes(String(architecture))
      && (recipe === 'qwen3_8_27b' || recipe === 'qwen3_8_27b_nvfp4') && !profile?.parameters.chat_template) return qwen
  }
}

export function effectiveReasoningEffort(capability: ReasoningCapability | undefined, value?: ReasoningEffort): ReasoningEffort | undefined {
  if (!capability || value === undefined) return
  if (capability.kind === 'qwen' && value === 'high') return 'xhigh'
  if (capability.kind === 'deepseek' && typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= 100) return value
  return capability.levels.some(level => level.value === value) ? value : undefined
}

export function reasoningLabel(value?: ReasoningEffort) {
  return ({ none: '关闭', low: '低', medium: '中', high: '高', xhigh: '高', max: '最高' } as const)[value as Exclude<ReasoningEffort, number>] || (typeof value === 'number' ? String(value) : '默认')
}

export function reasoningDefaultLabel(capability?: ReasoningCapability, profile?: LoadProfile) {
  const inherited = profile?.generation.reasoning_effort
  if (inherited !== undefined && inherited !== null) {
    const label = typeof inherited === 'number' || typeof inherited === 'string' ? reasoningLabel(inherited as ReasoningEffort) : String(inherited)
    return `配置 ${label === '默认' ? String(inherited) : label}`
  }
  if (profile?.backend === 'ninfer' && profile.extra_args.includes('--no-thinking')) return '启动配置 关闭'
  return capability ? '模板默认' : '引擎默认'
}

export function reasoningRequestParameters(capability: ReasoningCapability | undefined, value: ReasoningEffort | undefined, backend: string) {
  const effort = effectiveReasoningEffort(capability, value)
  // Default means omission: do not overwrite the loaded service's defaults.
  if (effort === undefined) return {}
  return { reasoning_effort: effort, ...(backend === 'llama_cpp' ? { chat_template_kwargs: { enable_thinking: effort !== 'none' } } : {}) }
}
