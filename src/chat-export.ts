import type { Conversation, Message } from './chat-history'
import { messageTime } from './chat-time'
import { fullReasoning } from './chat-response'

function label(text: string) {
  return text.replace(/[\r\n]+/g, ' ').replace(/[\\`*_{}\[\]<>#!|]/g, '\\$&')
}

function messageMarkdown(message: Message) {
  const parts = [`## ${message.role === 'user' ? '用户' : `助手${message.model ? ` · ${label(message.model)}` : ''}`}`]
  const time = messageTime(message)
  if (time) parts.push(`${time.label}：${time.full}`)
  for (const image of message.images) parts.push(`![${label(image.name)}](<${image.url}>)`)
  for (const document of message.documents || []) {
    const fence = '`'.repeat(Array.from(document.text.matchAll(/`+/g)).reduce((length, match) => Math.max(length, match[0].length + 1), 3))
    parts.push(`### 附件 · ${label(document.name)}\n\n${fence}text\n${document.text}\n${fence}`)
  }
  const reasoning = fullReasoning(message)
  if (message.activity?.length) {
    const statuses = { pending: '待执行', running: '执行中', completed: '已完成', failed: '失败', cancelled: '已取消' }
    const activity = message.activity.map(event => {
      if (event.kind !== 'tool') return event.text
      const content = `参数：${event.call.function.arguments}${event.result === undefined ? '' : `\n结果：${event.result}`}`
      const fence = '`'.repeat(Array.from(content.matchAll(/`+/g)).reduce((length, match) => Math.max(length, match[0].length + 1), 3))
      return `### 工具调用 · ${label(event.call.function.name || '未命名')} · ${statuses[event.status]}\n\n${fence}text\n${content}\n${fence}`
    }).join('\n\n')
    parts.push(`<details>\n<summary>${message.activity.some(event => event.kind === 'tool') ? '思考过程与工具调用' : '思考过程'}</summary>\n\n${activity}\n\n</details>`)
  } else if (reasoning) parts.push(`<details>\n<summary>思考过程</summary>\n\n${reasoning}\n\n</details>`)
  if (message.text && !(message.activity?.length && message.tools?.length)) parts.push(message.text)
  if (message.webSearch?.sources.length) {
    parts.push(`### 联网资料\n\n查询：${(message.webSearch.queries || [message.webSearch.query]).map(label).join('；')}\n\n` + message.webSearch.sources.map(source => {
      const pages = message.webSearch!.pages?.filter(page => page.source_id === source.id)
      const images = message.webSearch!.images?.filter(image => image.source_id === source.id) || []
      const summaries = message.webSearch!.summaries?.filter(summary => summary.url === source.url) || []
      const state = pages?.length ? `已读：${pages.map(page => `${page.start_line}–${page.end_line} 行`).join('、')}` : summaries.length ? '阅读摘要' : images.length ? '图片来源' : source.snippet ? '搜索摘要' : '搜索来源'
      return `${source.id}. [${label(source.title || source.url)}](<${source.url.replace(/>/g, '%3E').replace(/</g, '%3C')}>) · ${state}`
        + images.map(image => `\n   - [${label(image.title || '原图')}](<${image.url.replace(/>/g, '%3E').replace(/</g, '%3C')}>)`).join('')
        + summaries.map(summary => `\n\n   ${summary.text.replace(/\n/g, '\n   ')}\n`).join('')
    }).join('\n'))
  }
  for (const call of message.activity?.length ? [] : message.tools || []) {
    // A longer fence keeps backticks in raw tool arguments inside the code block.
    const fence = '`'.repeat(Array.from(call.function.arguments.matchAll(/`+/g)).reduce((length, match) => Math.max(length, match[0].length + 1), 3))
    parts.push(`### 工具调用 · ${label(call.function.name || '未命名')}\n\n${fence}json\n${call.function.arguments}\n${fence}`)
  }
  for (const result of message.activity?.length ? [] : message.toolResults || []) {
    const content = `参数：${result.call.function.arguments}\n结果：${result.content}`
    const fence = '`'.repeat(Array.from(content.matchAll(/`+/g)).reduce((length, match) => Math.max(length, match[0].length + 1), 3))
    parts.push(`### 工具结果 · ${label(result.call.function.name)}\n\n${fence}text\n${content}\n${fence}`)
  }
  const status = message.status === 'generating' ? '生成中（导出时的内容）'
    : message.status === 'interrupted' ? '已中断' : message.stopped || message.status === 'stopped' ? '已停止'
      : message.status === 'failed' ? '生成失败' : ''
  if (status) parts.push(`> ${status}`)
  if (message.error) parts.push(`> ${message.error.replace(/\r?\n/g, '\n> ')}`)
  if (message.finishReason === 'length') parts.push('> 已达到输出 Token 上限。')
  return parts.join('\n\n')
}

export function conversationMarkdown(conversation: Conversation) {
  return [`# ${label(conversation.title)}`, ...conversation.messages.map(messageMarkdown)].join('\n\n---\n\n') + '\n'
}
