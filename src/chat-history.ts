import type { ChatActivityEvent } from './chat-activity'
import type { ContextCompaction } from './chat-context'
import type { Timings, Usage } from './chat-response'
import type { ChatPreset } from './chat-presets'
import type { ReasoningEffort } from './chat-reasoning'
import type { ModelRequestRecord, TurnTotals } from './chat-turn-metrics'
import type { LocalResource } from './chat-local-resources'
import { DEFAULT_WEB_SEARCH_SETTINGS, type WebSearchMode, type WebSearchResult } from './web-search'

export interface Attachment { id: string; name: string; size: number; url: string }
export interface DocumentAttachment { id: string; name: string; size: number; text: string }
export interface ToolCall { id?: string; type?: string; function: { name: string; arguments: string } }
export interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  images: Attachment[]
  documents?: DocumentAttachment[]
  createdAt: number
  responseAt?: number
  status: 'completed' | 'generating' | 'stopped' | 'failed' | 'interrupted'
  reasoning?: string
  // Earlier tool-loop passes are display history, not the final model message.
  reasoningHistory?: string
  thinking?: boolean
  model?: string
  backend?: string
  profileId?: string
  modelId?: string
  stopped?: boolean
  error?: string
  elapsed?: number
  tools?: ToolCall[]
  activity?: ChatActivityEvent[]
  toolResults?: Array<{ call: ToolCall; content: string }>
  contextRevision?: number
  usage?: Usage
  timings?: Timings
  requests?: ModelRequestRecord[]
  turn?: TurnTotals
  resources?: LocalResource[]
  instanceId?: string
  contextCapacity?: number
  firstOutput?: number
  finishReason?: string
  webSearch?: WebSearchResult
}
export interface ModelSelection { profile_id: string; model_id: string; alias: string; instance_id?: string }
export interface ConversationSummary { id: string; title: string; created_at: string; updated_at: string; last_message_at?: number | null; last_sent_at?: number | null; pinned?: boolean; sort_order?: number; message_count: number }
export interface Conversation extends ConversationSummary {
  selection: ModelSelection | null
  draft: { text: string; images: Attachment[]; documents?: DocumentAttachment[] }
  settings: { system: string; temperature?: number; maxTokens?: number; stream: boolean; toolsDraft: string; presetId?: string; reasoningEffort?: ReasoningEffort; webSearch?: boolean; webSearchMode?: WebSearchMode; webSearchRounds?: number; toolConcurrency?: number; basicTools?: boolean; toolDirectory?: string; autoCompact?: boolean }
  messages: Message[]
  compaction?: ContextCompaction | null
}
export type ConversationPatch = Partial<Pick<Conversation, 'title' | 'selection' | 'draft' | 'settings' | 'messages' | 'compaction'>>

export function emptyConversation(selection: ModelSelection | null = null, preset?: ChatPreset): Conversation {
  const now = new Date().toISOString()
  return { id: crypto.randomUUID(), title: '新会话', created_at: now, updated_at: now, message_count: 0,
    selection, draft: { text: '', images: [] }, settings: { system: preset?.system ?? '', stream: true, toolsDraft: preset?.toolsDraft ?? '', ...DEFAULT_WEB_SEARCH_SETTINGS, ...(preset ? { presetId: preset.id, ...(preset.toolConcurrency === undefined ? {} : { toolConcurrency: preset.toolConcurrency }), ...(preset.reasoningEffort === undefined ? {} : { reasoningEffort: preset.reasoningEffort }) } : {}) }, messages: [] }
}
