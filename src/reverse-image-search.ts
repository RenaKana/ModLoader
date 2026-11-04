import type { Attachment } from './chat-history'
import { apiFetch } from './api'

export interface ReverseImageSearchResult {
  provider: string
  elapsed_ms: number
  matches: Array<{ similarity: number; thumbnail: string; title: string; author?: string; urls: string[]; index_name?: string }>
  quota?: { short_remaining?: number; long_remaining?: number }
}

export async function searchReverseImage(image: Attachment, signal: AbortSignal): Promise<ReverseImageSearchResult> {
  signal.throwIfAborted()
  return (await apiFetch('/api/web-reverse-image', { method: 'POST', body: JSON.stringify({ image: image.url, filename: image.name }), signal })).json()
}

export function reverseImageTool() {
  return {
    type: 'function', function: {
      name: 'reverse_image_search',
      description: '以图搜源：将指定的用户图片直接提交给 SauceNAO，查询原图、作者与发布页面。仅在用户需要查图片出处时使用，不代替看图识别。image_id 使用用户消息中图片引用提供的 ID；没有图片引用时不可调用。',
      parameters: { type: 'object', properties: { image_id: { type: 'string' } }, required: ['image_id'], additionalProperties: false },
    },
  }
}
