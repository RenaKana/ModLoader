import type { Message } from './chat-history'

const dayFormat = new Intl.DateTimeFormat('sv-SE')
const clockFormat = new Intl.DateTimeFormat('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

export function messageTime(message: Message) {
  const timestamp = message.responseAt ?? message.createdAt
  const date = new Date(timestamp)
  if (!Number.isFinite(date.getTime())) return null
  const day = dayFormat.format(date)
  const clock = clockFormat.format(date)
  const label = message.role === 'user' ? '发送时间' : message.responseAt !== undefined ? '回复时间' : '消息记录时间'
  return { timestamp, iso: date.toISOString(), label, full: `${day} ${clock}`,
    short: date.toDateString() === new Date().toDateString() ? clock : `${day.slice(5)} ${clock}` }
}
