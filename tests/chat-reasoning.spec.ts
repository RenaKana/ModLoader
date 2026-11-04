import { test, expect } from '@playwright/test'
import { chatReasoningCapability, effectiveReasoningEffort, reasoningRequestParameters } from '../src/chat-reasoning'
import { emptyConversation } from '../src/chat-history'
import type { ModelAsset, RuntimeInstance, RuntimeInstallation, LoadProfile } from '../src/types'
import ninferState from './fixtures/ninfer-chat-state.json'

const instance = { backend: 'strata' } as RuntimeInstance
const model = { name: 'Qwen3.8-27B-NVFP4-MTP-HIGH', architecture: 'qwen35', metadata: {} } as ModelAsset
const runtime = { metadata: { supported_flags: ['--jinja', '--reasoning-effort'] } } as RuntimeInstallation

test('推理档位按已核实模型与后端选择，未知模型保持默认', () => {
  const qwen = chatReasoningCapability(instance, { ...model, architecture: 'qwen4exp' })!
  expect(qwen.levels.map(level => level.value)).toEqual(['none', 'low', 'medium', 'xhigh'])
  expect(chatReasoningCapability({ ...instance, backend: 'llama_cpp' }, model, runtime)).toEqual(qwen)
  expect(chatReasoningCapability({ ...instance, backend: 'llama_cpp' }, { ...model, name: '另一个 qwen35 模型' }, runtime)).toBeUndefined()
  expect(chatReasoningCapability({ ...instance, backend: 'llama_cpp' }, model, { ...runtime, metadata: {} })).toBeUndefined()
  expect(chatReasoningCapability({ ...instance, backend: 'llama_cpp' }, model, runtime, { extra_args: ['--chat-template-file=test.jinja'] } as LoadProfile)).toBeUndefined()
  expect(chatReasoningCapability({ ...instance, backend: 'custom' }, model, runtime)).toBeUndefined()
  expect(chatReasoningCapability(instance, undefined)).toBeUndefined()
  expect(effectiveReasoningEffort(qwen, 'high')).toBe('xhigh')
  expect(effectiveReasoningEffort(qwen, 'max')).toBeUndefined()
  expect(effectiveReasoningEffort(qwen, 50)).toBeUndefined()
})

test('默认不覆盖服务，关闭与自定义数值发送真实参数且不使用 token 预算', () => {
  const qwen = chatReasoningCapability(instance, { ...model, architecture: 'qwen4exp' })!
  const ds = chatReasoningCapability(instance, { ...model, architecture: 'deepseek41' })!
  expect(reasoningRequestParameters(qwen, undefined, 'strata')).toEqual({})
  expect(reasoningRequestParameters(undefined, 'low', 'strata')).toEqual({})
  expect(reasoningRequestParameters(qwen, 'max', 'strata')).toEqual({})
  expect(reasoningRequestParameters(qwen, 'xhigh', 'strata')).toEqual({ reasoning_effort: 'xhigh' })
  expect(reasoningRequestParameters(qwen, 'none', 'llama_cpp')).toEqual({ reasoning_effort: 'none', chat_template_kwargs: { enable_thinking: false } })
  expect(reasoningRequestParameters(qwen, 'low', 'llama_cpp')).toEqual({ reasoning_effort: 'low', chat_template_kwargs: { enable_thinking: true } })
  for (const effort of [1, 17, 50, 100, 'max'] as const) expect(reasoningRequestParameters(ds, effort, 'strata')).toEqual({ reasoning_effort: effort })
  for (const effort of [0, 101, 1.5, NaN, 'xhigh'] as const) expect(reasoningRequestParameters(ds, effort, 'strata')).toEqual({})
})

test('实际 NInfer qwen3_5_text 工件显示真实档位，不依赖不存在的 CLI effort 标志', () => {
  const { instance, model, runtime, profile } = ninferState as unknown as { instance: RuntimeInstance; model: ModelAsset; runtime: RuntimeInstallation; profile: LoadProfile }
  const capability = chatReasoningCapability(instance, model, runtime, profile)!
  expect(capability.levels.map(level => level.value)).toEqual(['none', 'low', 'medium', 'xhigh'])
  expect(chatReasoningCapability(instance, model, { ...runtime, version: 'unknown', metadata: {} }, profile)).toEqual(capability)
  expect(chatReasoningCapability(instance, { ...model, name: '另一个 Qwen3.5 模型' }, runtime, profile)).toBeUndefined()
  expect(chatReasoningCapability(instance, model, runtime, { ...profile, extra_args: ['--chat-template', '/custom/template.jinja'] })).toBeUndefined()
  const source = { ...model, name: 'unsloth--Qwen3.8-27B-NVFP4', format: 'safetensors', architecture: 'qwen3_5', metadata: { model_type: 'qwen3_5', architectures: ['Qwen3_5ForConditionalGeneration'] } }
  for (const recipe of ['qwen3_8_27b', 'qwen3_8_27b_nvfp4']) {
    const prepared = { ...profile, parameters: { conversion_recipe: recipe } }
    expect(chatReasoningCapability(instance, source, runtime, prepared)).toEqual(capability)
    expect(chatReasoningCapability(instance, source, runtime, { ...prepared, parameters: { ...prepared.parameters, chat_template: '/custom.jinja' } })).toBeUndefined()
  }
  expect(chatReasoningCapability(instance, { ...source, name: '未知 Qwen3.5' }, runtime, { ...profile, parameters: { conversion_recipe: 'qwen3_8_27b_nvfp4' } })).toBeUndefined()
  expect(chatReasoningCapability(instance, source, runtime, { ...profile, parameters: { conversion_recipe: 'qwen3_6_27b' } })).toBeUndefined()
  for (const effort of ['none', 'low', 'medium', 'xhigh'] as const) {
    expect(reasoningRequestParameters(capability, effort, 'ninfer')).toEqual({ reasoning_effort: effort })
  }
  expect(reasoningRequestParameters(capability, 'high', 'ninfer')).toEqual({ reasoning_effort: 'xhigh' })
  for (const effort of [undefined, 50, 'max'] as const) expect(reasoningRequestParameters(capability, effort, 'ninfer')).toEqual({})
})

test('旧预设保持默认，默认预设把强度复制到独立会话快照', () => {
  const preset = { id: 'preset', name: '推理', system: '系统', toolsDraft: '', created_at: '', updated_at: '' }
  expect(emptyConversation(null, preset).settings).not.toHaveProperty('reasoningEffort')
  const conversation = emptyConversation(null, { ...preset, reasoningEffort: 17 })
  expect(conversation.settings.reasoningEffort).toBe(17)
  conversation.settings.reasoningEffort = 'low'
  expect(preset).not.toHaveProperty('reasoningEffort')
})
