import type { AppState, Connection } from './types'
import type { OperationProgress } from './OperationStatus'

export const REFRESH_INTERVAL_MS = 1000
// Match the backend telemetry cache while a generation is active.
export const GENERATION_REFRESH_INTERVAL_MS = 500

let connection: Promise<Connection> | null = null

export class ApiError extends Error {
  constructor(public status: number, message: string, public details: unknown = null) {
    super(message)
    this.name = 'ApiError'
  }
}

export function resetConnection() { connection = null }

export async function getConnection() {
  if (!connection) connection = window.modloader
    ? window.modloader.connection()
    : Promise.resolve({ baseUrl: '', token: '' })
  return connection
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  const { baseUrl, token } = await getConnection()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  let response: Response
  try {
    response = await fetch(`${baseUrl.replace(/\/$/, '')}${path}`, { ...init, headers })
  } catch (error) {
    if (error instanceof DOMException && (error.name === 'AbortError' || error.name === 'TimeoutError')) throw error
    throw new ApiError(0, '无法连接管理服务，请检查服务状态后重试。')
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    const detail = payload.detail ?? payload.error ?? payload.message
    const message = typeof detail === 'string' ? detail
      : typeof detail?.message === 'string' ? detail.message
        : typeof detail?.error === 'string' ? detail.error
          : Array.isArray(detail) ? detail.map(item => typeof item?.msg === 'string' ? item.msg : '').filter(Boolean).join('；') || `参数校验失败（${response.status}）`
            : `请求失败（${response.status}）`
    throw new ApiError(response.status, message, detail)
  }
  return response
}

export async function request<T = unknown>(method: string, path: string, body?: unknown, options: Pick<RequestInit, 'signal'> = {}): Promise<T> {
  const response = await apiFetch(path, { ...options, method, body: body === undefined ? undefined : JSON.stringify(body) })
  if (response.status === 204) return undefined as T
  return response.json()
}

export async function requestProgress<T>(method: string, path: string, body: unknown, options: {
  signal?: AbortSignal; onProgress?: (progress: OperationProgress) => void
} = {}): Promise<T> {
  const response = await apiFetch(path, { method, body: body === undefined ? undefined : JSON.stringify(body),
    signal: options.signal, headers: { Accept: 'application/x-ndjson' } })
  if (!response.headers.get('content-type')?.includes('application/x-ndjson')) return response.json()
  if (!response.body) throw new ApiError(0, '服务未返回操作结果，请重试。')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n')
      buffer = done ? '' : lines.pop()!
      for (const line of lines) {
        if (!line.trim()) continue
        const event = JSON.parse(line)
        if (event.type === 'progress') options.onProgress?.(event as OperationProgress)
        else if (event.type === 'result') return event.value as T
        else if (event.type === 'error') throw new ApiError(event.status || 500,
          typeof event.detail === 'string' ? event.detail : event.detail?.message || '操作未完成，请重试。', event.detail)
      }
      if (done) throw new ApiError(0, '连接已中断，未收到操作结果，请重试。')
    }
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}

export async function fetchState(): Promise<AppState> {
  const state = await request<Partial<AppState>>('GET', '/api/state')
  return {
    models: state.models || [], profiles: state.profiles || [], runtimes: state.runtimes || [],
    instances: state.instances || [], downloads: state.downloads || [], installations: state.installations || [],
    migrations: state.migrations || [],
    settings: state.settings || {}, backends: state.backends || [], hardware: state.hardware || {},
    generation_parameters: state.generation_parameters || [],
  }
}

export async function pickPath(kind: 'file' | 'directory') {
  return window.modloader?.pickPath(kind) ?? null
}

export async function copy(text: string) {
  if (window.modloader) return window.modloader.copy(text)
  return navigator.clipboard.writeText(text)
}

export async function openPath(path: string) {
  if (!window.modloader) throw new Error('请在桌面应用中打开本地路径。')
  return window.modloader.openPath(path)
}

export async function openExternal(url: string) {
  if (window.modloader) return window.modloader.openExternal(url)
  window.open(url, '_blank', 'noopener,noreferrer')
}

export async function exportHtml(source: string, action: 'open' | 'save') {
  if (!window.modloader) throw new Error('请在桌面应用中打开或保存 HTML。')
  return window.modloader.exportHtml(source, action)
}

export async function exportMarkdown(source: string, title: string) {
  if (!window.modloader) throw new Error('请在桌面应用中导出对话。')
  return window.modloader.exportMarkdown(source, title)
}

export async function saveImage(source: string, name: string) {
  if (!window.modloader) throw new Error('请在桌面应用中保存图片。')
  if (source.startsWith('blob:')) {
    const response = await fetch(source)
    if (!response.ok) throw new Error('附件图片已不可用，请重新打开。')
    const blob = await response.blob()
    source = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => typeof reader.result === 'string' ? resolve(reader.result) : reject(new Error('无法读取附件图片。'))
      reader.onerror = () => reject(reader.error || new Error('无法读取附件图片。'))
      reader.readAsDataURL(blob)
    })
  }
  return window.modloader.saveImage(source, name)
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : '操作未完成，请重试。'
}

export function isActive(state: string) { return ['preparing', 'starting', 'running', 'ready', 'stopping', 'loading'].includes(state) }
export function isReady(state: string) { return ['running', 'ready'].includes(state) }
export function jobState(job: { state?: string; status?: string }) { return job.state || job.status || 'queued' }
export function isJobActive(job: { state?: string; status?: string }) {
  return ['queued', 'pending', 'running', 'downloading', 'installing', 'resolving', 'extracting', 'pausing', 'cancelling'].includes(jobState(job))
}
