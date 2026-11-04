import { memo, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { Ban, Calculator, Check, ChevronRight, Clock3, FileText, Folder, Globe, Image, LoaderCircle, Paperclip, PencilLine, Search, Wrench, X } from 'lucide-react'
import type { Message } from './chat-history'
import { describeToolActivity, formatToolDetail, messageActivity, type ChatActivityEvent, type ChatToolStatus } from './chat-activity'
import { MarkdownContent, ReasoningContent } from './MessageContent'
import './chat-activity.css'

const toolIcons = { search_web: Search, search_images: Image, reverse_image_search: Image, read_page: Globe, list_directory: Folder, read_file: FileText, write_file: PencilLine, attach_file: Paperclip, current_time: Clock3, calculate: Calculator }
const statusLabels: Record<ChatToolStatus, string> = { pending: '待处理', running: '运行中', completed: '已完成', failed: '失败', cancelled: '已取消' }

function ToolStatus({ status }: { status: ChatToolStatus }) {
  return <span className="chat-activity-status" data-status={status} role="img" aria-label={statusLabels[status]} aria-hidden={status === 'pending' || status === 'cancelled'}>
    <LoaderCircle className="chat-activity-status-running" size={14} aria-hidden="true" />
    <span className="chat-activity-status-settled" aria-hidden="true">{status === 'failed' ? <X size={14} /> : status === 'cancelled' ? <Ban size={14} /> : status === 'pending' ? <Clock3 size={14} /> : <Check size={14} />}</span>
  </span>
}

const ActivityTool = memo(function ActivityTool({ event, detail }: { event: Extract<ChatActivityEvent, { kind: 'tool' }>; detail: 'compact' | 'full' }) {
  const [open, setOpen] = useState(detail === 'full')
  const [height, setHeight] = useState(0)
  const body = useRef<HTMLDivElement>(null)
  const contentId = useId()
  const description = describeToolActivity(event)
  const Icon = toolIcons[event.call.function.name as keyof typeof toolIcons] || Wrench
  useEffect(() => { setOpen(detail === 'full') }, [detail])
  useLayoutEffect(() => {
    const element = body.current
    if (!element) return
    const measure = () => setHeight(element.getBoundingClientRect().height)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  return <div className="chat-activity-tool" data-status={description.status} data-activity-id={event.id}>
    <button type="button" className="chat-activity-tool-toggle" aria-expanded={open} aria-controls={contentId} onClick={() => setOpen(value => !value)}>
      <Icon size={15} className="chat-activity-tool-icon" aria-hidden="true" />
      <span className="chat-activity-tool-action">{description.action}</span>
      {description.target && <span className="chat-activity-tool-target" title={description.target}>{description.target}</span>}
      {description.outcome && <span className="chat-activity-tool-outcome" title={description.outcome}>{description.outcome}</span>}
      {description.status === 'pending' && <span className="chat-activity-tool-outcome">待处理</span>}
      <ToolStatus status={description.status} />
      <ChevronRight size={13} className="chat-activity-tool-chevron" aria-hidden="true" />
    </button>
    {description.error && <div className="chat-activity-error" title={description.error}>{description.error}</div>}
    <div id={contentId} className="chat-activity-tool-collapse" data-open={open} style={{ height: open ? height : 0 }} aria-hidden={!open} inert={!open}>
      <div className="chat-activity-tool-body" ref={body}>
        <div className="chat-activity-detail"><span>参数</span><pre tabIndex={0} aria-label={`${description.action}参数`}>{formatToolDetail(event.call.function.arguments)}</pre></div>
        {event.result !== undefined && <div className="chat-activity-detail"><span>结果</span><pre tabIndex={0} aria-label={`${description.action}结果`}>{formatToolDetail(event.result)}</pre></div>}
      </div>
    </div>
  </div>
})

export const ChatActivity = memo(function ChatActivity({ message, generating, detail = 'compact' }: { message: Message; generating: boolean; detail?: 'compact' | 'full' }) {
  const activity = messageActivity(message)
  if (!activity.length) return null
  const thinking = Boolean(message.thinking)
  const running = activity.some(event => event.kind === 'tool' && event.status === 'running')
  const scrollKey = activity.map(event => event.kind === 'tool' ? `${event.id}:${event.status}:${event.result?.length ?? 0}` : event.text).join('\n')
  return <ReasoningContent text={scrollKey} thinking={thinking} active={generating} className="message-activity" summary={<>
    <span>{thinking ? '思考中' : generating ? '处理过程' : '思考过程'}</span>
    {generating && (thinking || running) && <LoaderCircle className="chat-activity-summary-spinner" size={13} aria-hidden="true" />}
  </>}>
    <div className="chat-activity-events">
      {activity.map(event => event.kind === 'tool'
        ? <ActivityTool key={event.id} event={event} detail={detail} />
        : <div className={`chat-activity-${event.kind}`} key={event.id} data-activity-id={event.id}><MarkdownContent text={event.text} streaming={generating} /></div>)}
    </div>
  </ReasoningContent>
})
