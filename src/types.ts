import type { BackendRequestReport, ModelRequestRecord, RateTotal, StreamDecodeSample, TokenTotal, TurnTotals } from './chat-turn-metrics'
import type { ChatActivityEvent } from './chat-activity'

export type RuntimePlatform = 'windows' | 'wsl2'
export interface ModelLocation { platform: RuntimePlatform; distribution?: string }

export interface ModelAsset {
  id: string
  name: string
  path: string
  location?: ModelLocation
  format: string
  architecture?: string | null
  quantization?: string | null
  size_bytes: number
  files: string[]
  modalities: string[]
  metadata: Record<string, unknown>
  source: Record<string, unknown>
  managed: boolean
  warnings: string[]
}

export interface LoadProfile {
  id: string
  model_id: string
  name: string
  backend: string
  runtime_id: string
  alias: string
  parameters: Record<string, unknown>
  generation: Record<string, unknown>
  extra_args: string[]
  parameter_editor_version?: 1
  revision: number
  chat_quick_load?: boolean
}

export interface RuntimeInstallation {
  id: string
  backend: string
  platform?: RuntimePlatform
  name: string
  path: string
  version: string
  managed: boolean
  metadata: Record<string, unknown>
  generation_parameters?: ParameterField[]
  startup_expert_prefetch?: StartupExpertPrefetchCapability
}

export interface StartupExpertPrefetchCapability {
  status: 'ready' | 'missing' | 'mismatch' | 'unsupported'
  available: boolean
  preparable: boolean
  source: 'managed' | 'legacy' | null
  message: string
  engine_identity: Record<string, unknown>
  enabled?: boolean
}

export interface RuntimeInstance {
  id: string
  model_id: string
  profile_id: string
  profile_revision: number
  alias: string
  backend: string
  platform?: RuntimePlatform
  distribution?: string
  guest_pid?: number | null
  runtime_id: string
  state: string
  port: number
  pid?: number | null
  started_at: string
  error?: string | null
  active_requests: number
  capabilities: Record<string, unknown>
  phase?: string | null
  message?: string | null
}

export interface ParameterField {
  key: string
  label: string
  type: 'text' | 'number' | 'boolean' | 'select' | 'file'
  default: unknown
  options: unknown[]
  description: string
  advanced: boolean
  minimum?: number | null
  maximum?: number | null
  exclusive_minimum?: boolean
  integer?: boolean
  step?: number | null
  special_values?: (number | string)[]
}

export interface Backend {
  id: string
  supported_platforms?: RuntimePlatform[]
  label: string
  parameters: ParameterField[]
  generation_parameters?: ParameterField[]
  description?: string
  installable?: boolean
  subtitle?: string
  symbol?: string
  import_kind?: 'file' | 'directory' | 'wsl'
  import_hint?: string
  python_option?: boolean
}

export interface WslDistribution { name: string; state: string; version: number; managed: boolean }
export interface WslDistributions { available: boolean; distributions: WslDistribution[] }
export interface WslCheck { distribution: string; version: number; python: string | null; gpu: string | null; ok: boolean; errors: string[] }
export interface WslDirectory {
  distribution: string
  path: string
  parent: string | null
  entries: Array<{ name: string; path: string; type: 'directory' | 'file'; size_bytes: number | null }>
}

export interface RuntimeVersion {
  version: string
  label: string
  status: 'available' | 'unverified' | 'unavailable'
  recommended?: boolean
  reason?: string
  variants: Array<{ id: string; label: string }>
}

export interface RuntimeVersionPage {
  versions: RuntimeVersion[]
  next_cursor: string | null
}

export interface RuntimeInstallFile {
  id: string
  name: string
  purpose: string
  size_bytes: number | null
  cached: boolean
  downloaded_bytes?: number
  status?: string
}

export interface RuntimeManifest {
  id: string
  title: string
  files: RuntimeInstallFile[]
}

export interface RuntimeInstallPlan {
  id: string
  backend: string
  version: string
  variant: string
  options: Record<string, unknown>
  stage: RuntimeManifest
  dependency_notice?: string | null
}

export interface InstallationHistory {
  items: Job[]
  total: number
  page: number
  page_size: number
}

export interface RuntimeSwitchPreview {
  profiles: Array<{ id: string; name: string }>
  warnings: string[]
  startup_expert_prefetch?: {
    status: 'unchanged' | 'preserved' | 'choice_required' | 'disabled'
    affected_profiles: Array<{ id: string; name: string }>
    message: string
  }
}

export interface Job {
  id: string
  phase?: string
  can_pause?: boolean
  state?: string
  status?: string
  progress?: number
  total_bytes?: number | null
  completed_bytes?: number
  downloaded_bytes?: number
  error?: string | null
  name?: string
  label?: string
  repo_id?: string
  backend?: string
  source?: string
  message?: string
  operation?: 'install' | 'update' | 'prepare_startup_prefetch'
  version?: string
  update_from?: string | null
  runtime_id?: string | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  manifest?: RuntimeManifest | null
  stages?: RuntimeManifest[]
  file?: string
  current_file?: string
  file_downloaded_bytes?: number
  file_total_bytes?: number | null
  completed_files?: number
  total_files?: number
  request?: { backend?: string; options?: Record<string, unknown>; version?: string; update_from?: string; [key: string]: unknown }
  [key: string]: unknown
}

export type WebServiceCapability = 'search' | 'images' | 'fetch' | 'reverse'
export interface WebServiceDefinition {
  id: string
  label: string
  capabilities: WebServiceCapability[]
  url_setting: string | null
  credential_key: string | null
  fetch_label?: string
}

export interface Settings {
  api_history_enabled?: boolean
  api_history_directory?: string
  api_history_retention_days?: 0 | 7 | 30
  api_history_status?: { state: 'disabled' | 'saving' | 'saved' | 'error'; error: string | null; directory: string; cleanup_error?: string | null }
  chat_unload_before_load?: boolean
  chat_tool_details?: 'compact' | 'full'
  web_services?: WebServiceDefinition[]
  web_credentials?: Record<string, boolean>
  api_port?: number
  api_host?: string
  close_to_tray?: boolean
  proxy_mode?: 'system' | 'direct' | 'manual'
  proxy_url?: string
  model_directory?: string
  font_scale?: number
  data_directory?: string
  hf_token_set?: boolean
  modelscope_token_set?: boolean
  credentials?: Record<string, boolean>
  credential_flags?: Record<string, boolean>
  [key: string]: unknown
}

export interface TelemetryMetric {
  id: string
  label: string
  value: number | string | null
  unit?: string
  kind: 'gauge' | 'counter' | 'text' | 'timestamp'
  description?: string
  rate_basis?: 'decode' | 'prefill' | 'end_to_end' | null
}
export interface TelemetryGroup {
  id: string
  title: string
  metrics: TelemetryMetric[]
  request_status?: string
  request_key?: string | null
  client_request_id?: string | null
  client_call_id?: string | null
}
export interface ChatGenerationActivity {
  instanceId: string
  messageId: string
  phase: 'planning' | 'searching' | 'reading' | 'tools' | 'compacting' | 'generating' | 'completed' | 'cancelled' | 'failed'
  requests?: ModelRequestRecord[]
  turn?: TurnTotals
  toolCount?: number
  model?: string
  contextCapacity?: number
  stream?: StreamDecodeSample
  toolActivity?: Extract<ChatActivityEvent, { kind: 'tool' }>[]
}
export interface LiveGeneration {
  client_request_id?: string | null
  client_call_id?: string | null
  tokens_per_second: number | null
  window_s: number | null
  scope: 'model' | 'request'
  source: string
  backend_running: number | null
  backend_queued: number | null
  backend_paused?: number | null
  available_for_chat: boolean
  reason: string | null
  request_key?: string | null
  phase?: string | null
  average_tokens_per_second?: number | null
  average_basis?: 'backend_decode' | 'sampled_decode' | null
  context_tokens?: number | null
  context_capacity?: number | null
  context_capacity_basis?: string | null
  context_basis?: string | null
  prompt_tokens?: number | null
  generated_tokens?: number | null
  elapsed_s?: number | null
  first_output_s?: number | null
  request_elapsed_s?: number | null
  prefill_tokens_per_second?: number | null
  prefill_window_s?: number | null
  average_prefill_tokens_per_second?: number | null
  prefill_average_basis?: 'backend_prefill' | 'sampled_prefill' | null
  prefill_processed_tokens?: number | null
  prefill_total_tokens?: number | null
  prefill_cached_tokens?: number | null
}
export interface GenerationEvent {
  run_id: string
  instance_id: string
  observed_at: number
  live_generation: LiveGeneration | null
  accounting: GenerationAccounting | null
  finalized: GenerationAccounting[]
}
export interface GenerationRequestSnapshot extends BackendRequestReport {
  started_at: number
  streaming: boolean
  measurements: NonNullable<ModelRequestRecord['measurements']>
  upstream?: { status?: number; code?: string | number; reason?: string }
}
export interface GenerationAccounting {
  request: GenerationRequestSnapshot
  turn: { source: 'external' | 'internal'; key: string; turn_id: string | null; client_request_id: string | null
    request_keys: string[]; prior: RequestHistoryStats | null }
}
export type ExternalTurnState = 'running' | 'waiting' | 'completed' | 'recovered' | 'failed' | 'cancelled' | 'incomplete'
export interface ExternalRequestRecord extends BackendRequestReport {
  run_id: string
  instance_id: string
  alias: string
  backend: string
  revision: number
  aggregation: 'request' | 'turn'
  turn_id: string | null
  turn_state: ExternalTurnState | null
  labels: string[]
  tool_names?: string[]
  client_context?: { source: 'opencode'; session_id: string }
  upstream?: { status?: number; code?: string | number; reason?: string }
  date?: string
  streaming?: boolean
  record_incomplete?: boolean
  live?: LiveGeneration | null
}
export interface ExternalHistorySnapshot {
  run_id: string
  aggregation: 'request' | 'turn'
  turn_id: string | null
  state: ExternalTurnState
  requests: ExternalRequestRecord[]
}
export interface RequestHistoryRun {
  id: string
  started_at: number
  current: boolean
  request_count: number
  instances: { id: string; alias: string; backend: string }[]
}
export interface RequestHistoryRuns {
  current_run_id: string
  runs: RequestHistoryRun[]
  issues: string[]
}
export interface RequestHistoryInstance {
  source?: 'external' | 'internal'
  run_id?: string
  id: string | null
  conversation_id?: string
  alias: string
  backend: string
  started_at?: number
}
export interface RequestHistoryCoverage {
  legacy_message_count: number
  legacy_metrics_message_count: number
}
export interface RequestHistoryDates {
  current_run_id: string
  usage_revision?: string
  today: string
  dates: {
    date: string
    request_count: number
    instances: RequestHistoryInstance[]
  }[]
  issues: string[]
  coverage?: RequestHistoryCoverage
}
export interface RequestHistoryStats {
  request_count: number
  input: TokenTotal
  output: TokenTotal
  // Per-request input + output; reported counts requests with both counters.
  total: TokenTotal
  cached: TokenTotal
  cache_pair: { input: number | null; cached: number | null; reported: number }
  // Fraction of the paired input, not the entire input total.
  cache_ratio: number | null
  decode: RateTotal
  prefill: RateTotal
  engine?: RateTotal
  // Arithmetic mean in seconds; native TTFT takes precedence for each request.
  ttft: { value: number | null; reported: number; engine_reported: number; gateway_reported: number; client_reported?: number; unknown_reported?: number }
  status_counts: Record<BackendRequestReport['status'], number>
  record_incomplete_count: number
}
export interface RequestHistoryGroup extends RequestHistoryStats {
  source?: 'external'
  matched_request_count?: number
  run_id: string
  instance_id: string
  turn_id: string | null
  request_key: string
  aggregation: 'request' | 'turn'
  started_at: number
  ended_at: number | null
  elapsed_s: number
  status: ExternalTurnState
}
export interface InternalRequestHistoryGroup extends RequestHistoryStats {
  source: 'internal'
  matched_request_count?: number
  conversation_id: string
  message_id: string
  instance_id: string | null
  alias: string
  backend?: string
  started_at: number
  ended_at: number | null
  elapsed_s: number
  status: ExternalTurnState
  context_capacity?: number
  requests: ModelRequestRecord[]
}
export interface RequestHistoryPage {
  external_group_by?: 'request'
  run_id: string | null
  items: ExternalRequestRecord[]
  total: number
  groups?: RequestHistoryGroup[]
  internal_groups?: InternalRequestHistoryGroup[]
  group_total?: number
  summary?: RequestHistoryStats
  coverage?: RequestHistoryCoverage
  offset: number
  limit: number
  issues: string[]
}
export interface RequestHistoryUsageBucket extends RequestHistoryStats {
  start_at: number
  end_at: number
}
export interface RequestHistoryUsage {
  as_of: number
  revision: string
  summary: RequestHistoryStats
  coverage: RequestHistoryCoverage
  issues: string[]
  trend: {
    start_at: number | null
    end_at: number | null
    bucket_seconds: number
    next_refresh_at: number
    buckets: RequestHistoryUsageBucket[]
  }
}
export interface Telemetry {
  instance_id: string
  backend: string
  state: string
  updated_at: string | number
  groups: TelemetryGroup[]
  unavailable: string[]
  live_generation?: LiveGeneration | null
  external_history?: ExternalHistorySnapshot | null
}

export interface Hardware {
  cpu?: string | Record<string, unknown>
  memory_total?: number
  memory_available?: number
  memory_commit_available?: number | null
  ram_total_bytes?: number
  ram_available_bytes?: number
  gpus?: Array<Record<string, unknown>>
  sampled_at?: number
  cpu_per_core?: number[]
  cpu_temperature_c?: number | null
  cpu_temperature_source?: string | null
  cpu_temperature_reason?: string | null
  cpu_temperature_status?: 'disabled' | 'driver_missing' | 'permission_required' | 'starting' | 'ready' | 'error'
  cpu_power_w?: number | null
  cpu_power_source?: string | null
  cpu_power_reason?: string | null
  gpu_temperature_reason?: string | null
  cpu_info?: { name?: string; cores?: number; threads?: number; max_clock_mhz?: number; l2_bytes?: number; l3_bytes?: number }
  memory_modules?: Array<{ capacity?: number; type?: string; speed_mt_s?: number; slot?: string }>
  [key: string]: unknown
}

export interface AppState {
  models: ModelAsset[]
  profiles: LoadProfile[]
  runtimes: RuntimeInstallation[]
  instances: RuntimeInstance[]
  downloads: Job[]
  installations: Job[]
  migrations?: Job[]
  settings: Settings
  backends: Backend[]
  generation_parameters?: ParameterField[]
  hardware: Hardware
}

export interface SourceFile { path: string; size: number | null; role?: string; group_id?: string | null; reason?: string | null }
export interface SourceGroup { id: string; name: string; format: string; kind: 'model' | 'vision'; files: string[]; size: number | null; quantization?: string | null }
export interface SourceSearchItem {
  id: string
  name: string
  description: string
  url: string
  downloads: number | null
  gated: boolean | string | null
}
export interface SourceRepository {
  source: string
  repo_id: string
  revision: string
  name: string
  description: string
  url: string
  files: SourceFile[]
  metadata: Record<string, unknown>
  groups?: SourceGroup[]
}

export interface PreparationStatus {
  parameter_migration?: { parameter: string; message: string }
  drafting?: StrataDraftingStatus
  paged_experts?: {
    enabled: boolean
    num_resident?: unknown
    runner?: string | null
    recommended_runner?: string | null
    disable_shared_experts_fusion?: boolean
    status: 'disabled' | 'available' | 'unverified' | 'unsupported'
    message: string
    normalized_parameters?: Record<string, unknown>
    normalized_extra_args?: string[]
  }
  startup_expert_prefetch?: StartupExpertPrefetchCapability
  dependencies?: Record<string, string>
  status: 'ready' | 'needs_preparation' | 'choice_required' | 'error'
  source?: { kind: string; label: string; profile_id?: string } | null
  pack?: string | null
  tokenizer?: string | null
  expert_profile?: string | null
  candidates: Array<{ id: string; label: string; pack: string; tokenizer: string }>
  errors: string[]
  message: string
  mmproj_candidates?: string[]
  vision_enabled?: boolean
  mmproj?: string | null
  mmproj_error?: string
  mtp?: MtpStatus
  acceleration?: AccelerationStatus
  parameters?: ParameterField[]
  generation_parameters?: ParameterField[]
  kind?: string
  artifact?: string | null
  recipes?: Array<{ id: string; label: string; description?: string }>
  preparation_fields?: ParameterField[]
}

export interface StrataDraftingStatus {
  supported: boolean
  mode: 'legacy' | 'auto' | 'mtp' | 'dflash2' | 'off'
  effective_mode: 'mtp' | 'dflash2' | 'off'
  status: 'ready' | 'needs_preparation' | 'error'
  path?: string
  message?: string
  download_bytes?: number
}

export interface AccelerationStatus {
  mode: string
  enabled: boolean
  status: 'disabled' | 'unverified' | 'available' | 'unsupported'
  kind: string
  label: string
  requires_draft: boolean
  fields: ParameterField[]
  message: string
}

export interface MtpStatus {
  status: 'builtin' | 'external' | 'missing' | 'unsupported' | 'unverified'
  message: string
  diagnostic?: string
  enabled: boolean
  path: string
  path_kind: 'file' | 'directory'
  needs_path: boolean
  candidates: Array<{ path: string; label: string }>
  resident_experts: boolean
  backend_supported: boolean | null
  tokens?: number | null
  token_field?: ParameterField
}

export interface Connection { baseUrl: string; token: string }
export interface DesktopBridge {
  openMiniMonitor(): Promise<void>
  publishMiniMonitor(snapshot: import('./mini-monitor').MiniMonitorSnapshot): void
  getMonitorVisibility(): Promise<boolean>
  onMonitorVisibility(handler: (visible: boolean) => void): () => void
  onMiniMonitorAction(handler: (action: import('./mini-monitor').MiniMonitorAction) => void): () => void
  connection(): Promise<Connection>
  pickPath(kind: 'file' | 'directory'): Promise<string | null>
  openExternal(url: string): Promise<void>
  exportHtml(source: string, action: 'open' | 'save'): Promise<{ canceled: boolean }>
  exportMarkdown(source: string, title: string): Promise<{ canceled: boolean }>
  saveImage(source: string, name: string): Promise<{ canceled: boolean }>
  openPath(path: string): Promise<void>
  localResourceAction(id: string, action: 'open' | 'reveal'): Promise<{ handed_off: boolean }>
  copy(text: string): Promise<void>
  onBeforeClose(handler: () => Promise<void>): () => void
  onWindowHidden(handler: () => void): () => void
  onCloseCancelled(handler: () => void): () => void
}

export interface PageProps {
  state: AppState
  refresh: () => Promise<void>
  notify: (message: string, tone?: 'success' | 'error' | 'info') => void
  stopInstance: (instance: RuntimeInstance) => Promise<void>
  openLogs: (instance: RuntimeInstance) => void
  navigate: (page: string, instanceId?: string) => void
}

declare global {
  interface Window { modloader?: DesktopBridge; miniMonitor?: import('./mini-monitor').MiniMonitorBridge }
}
