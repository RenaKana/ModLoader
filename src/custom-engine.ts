import type { ParameterField, RuntimeInstallation } from './types'

export interface CustomEngineConfig {
  args: string[]
  cwd: string
  env: Record<string, string>
  health_path: string
  chat_path: string
  api_model_name: string
  generation_parameters: string[]
  stream_usage: boolean
}

export const customGenerationOptions = [
  ['temperature', 'Temperature'], ['top_p', 'Top P'], ['max_tokens', '最大输出 Token'],
  ['top_k', 'Top K'], ['repeat_penalty', '重复惩罚'], ['seed', '随机种子'],
] as const
export const defaultCustomGeneration = ['temperature', 'top_p', 'max_tokens']

export function customConfig(runtime?: RuntimeInstallation): Partial<CustomEngineConfig> {
  return runtime?.backend === 'custom' ? (runtime.metadata.custom as Partial<CustomEngineConfig> | undefined) || {} : {}
}

export function runtimeGenerationFields(fields: ParameterField[], runtime?: RuntimeInstallation, capabilities?: Record<string, unknown>) {
  fields = runtime?.generation_parameters ?? fields
  if (runtime?.backend !== 'custom') return fields
  const keys = Array.isArray(capabilities?.generation_parameters)
    ? capabilities.generation_parameters : customConfig(runtime).generation_parameters || defaultCustomGeneration
  return fields.filter(field => keys.includes(field.key))
}
