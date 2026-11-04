export interface Usage { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number; prompt_tokens_details?: { cached_tokens?: number }; completion_tokens_details?: { reasoning_tokens?: number }; ttft_ms?: number; generation_ms?: number; decode_tps?: number }
export interface Timings { predicted_per_second?: number; predicted_per_token_ms?: number; predicted_n?: number; predicted_ms?: number; prompt_n?: number; prompt_ms?: number; prompt_per_second?: number; prompt_per_token_ms?: number; cache_n?: number }

export function fullReasoning(message: { reasoningHistory?: string; reasoning?: string }) {
  return [message.reasoningHistory, message.reasoning].filter(Boolean).join('\n\n')
}

export function reportedNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

// Some local backends return the thought block in content instead of a separate field.
// Keep incomplete opening/closing tags out of the visible streaming text.
export function splitThinking(content: string, finished = false) {
  const trimmed = content.trimStart()
  const opening = '<think>'
  const closing = '</think>'
  if (!finished && trimmed && opening.startsWith(trimmed)) return { text: '', reasoning: '', thinking: true }
  if (!trimmed.startsWith(opening)) return { text: content, reasoning: '', thinking: false }
  const rest = trimmed.slice(opening.length)
  const end = rest.indexOf(closing)
  if (end >= 0) return { text: rest.slice(end + closing.length).trimStart(), reasoning: rest.slice(0, end), thinking: false }
  let reasoning = rest
  if (!finished) {
    for (let size = closing.length - 1; size > 0; size--) {
      if (reasoning.endsWith(closing.slice(0, size))) { reasoning = reasoning.slice(0, -size); break }
    }
  }
  return { text: '', reasoning, thinking: true }
}

export function responseSpeed(usage?: Usage, timings?: Timings, elapsed?: number) {
  const engineSpeed = reportedNumber(timings?.predicted_per_second)
  if (engineSpeed !== undefined) return { value: engineSpeed, label: '解码速度', detail: '后端报告的本次生成速度' }
  const count = reportedNumber(timings?.predicted_n)
  const milliseconds = reportedNumber(timings?.predicted_ms)
  if (count !== undefined && milliseconds && milliseconds > 0) return { value: count / (milliseconds / 1000), label: '解码速度', detail: '后端生成 Token 数 / 后端生成耗时' }
  const tokens = reportedNumber(usage?.completion_tokens)
  // SparkInfer's chat API calls this decode_tps, but divides by the complete
  // engine generation duration, including TTFT. Keep that basis explicit.
  const backendSpeed = reportedNumber(usage?.decode_tps)
  if (backendSpeed !== undefined) return { value: backendSpeed, label: '引擎平均', detail: '后端报告的完整生成平均速度，包含首 Token 等待' }
  if (tokens !== undefined && elapsed && elapsed > 0) return { value: tokens / elapsed, label: '端到端平均', detail: '输出 Token / 完整请求耗时，包含排队、预填充及传输时间' }
  return undefined
}
