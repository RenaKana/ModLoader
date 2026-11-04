import type { ReasoningEffort } from './chat-reasoning'

export interface ChatPreset {
  id: string
  name: string
  system: string
  toolsDraft: string
  reasoningEffort?: ReasoningEffort
  toolConcurrency?: number
  created_at: string
  updated_at: string
}

export interface ChatPresetCatalog { items: ChatPreset[]; default_id: string | null }

export function validatePresetTools(toolsDraft: string) {
  if (!toolsDraft.trim()) return
  let tools: unknown
  try { tools = JSON.parse(toolsDraft) }
  catch { throw new Error('工具定义须为有效 JSON 数组。') }
  if (!Array.isArray(tools)) throw new Error('工具定义须为 JSON 数组。')
}
