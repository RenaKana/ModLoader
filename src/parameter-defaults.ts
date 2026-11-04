import type { LoadProfile, ParameterField } from './types'

export function parameterDefaultText(field: ParameterField, source = '默认') {
  const value = field.default
  if (value === null || value === undefined) return source === '默认' ? '引擎默认' : source
  const option = field.type === 'select' ? field.options.find(option => typeof option === 'object' && option !== null && Object.is((option as { value?: unknown }).value, value)) as { label?: string } | undefined : undefined
  const label = typeof value === 'boolean' ? value ? '开启' : '关闭' : option?.label || (typeof value === 'object' ? JSON.stringify(value) : String(value))
  return `${source} ${label}`
}

// Display only: callers keep omitted settings omitted in saved profiles and requests.
export function generationDefaultText(field: ParameterField, profile?: Pick<LoadProfile, 'backend' | 'generation' | 'extra_args'>) {
  const inherited = profile?.generation[field.key]
  if (inherited !== null && inherited !== undefined) return parameterDefaultText({ ...field, default: inherited }, '配置')
  if (profile?.backend === 'ninfer') {
    const flags: Record<string, string> = { max_tokens: '--default-max-tokens', temperature: '--temperature', top_p: '--top-p', top_k: '--top-k', seed: '--seed' }
    const args = profile.extra_args
    const flag = flags[field.key]
    const index = args.map(arg => arg.split('=')[0]).lastIndexOf(flag)
    if (field.key === 'temperature' && args.includes('--greedy')) return '启动配置 0'
    if (index >= 0) {
      const raw = args[index].includes('=') ? args[index].slice(args[index].indexOf('=') + 1) : args[index + 1]
      return raw === undefined ? '启动配置' : `启动配置 ${raw}`
    }
    if (field.default == null && ['temperature', 'top_p', 'top_k'].includes(field.key)) return '模型生成配置'
  }
  return parameterDefaultText(field)
}
