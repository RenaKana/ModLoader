import { reportedNumber } from './chat-response'
import './chat-context.css'

const countFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 0 })
const percentFormat = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 })

export default function ContextUsage({ used, capacity, detail }: {
  used?: number | null; capacity?: number | null; detail?: string
}) {
  const count = reportedNumber(used)
  const total = reportedNumber(capacity)
  const ratio = count !== undefined && total !== undefined && total > 0 ? count / total : undefined
  const values = `已用 ${count === undefined ? '—' : countFormat.format(count)} Token\n总容量 ${total ? countFormat.format(total) : '—'} Token`
  const tooltip = `上下文占用\n${values}${detail ? `\n${detail}` : ''}`
  return <span className={`chat-context-usage${ratio === undefined ? ' is-unavailable' : ''}`} tabIndex={0} role="img"
    aria-label={`上下文占用：${values.replace('\n', '；')}`} data-tooltip={tooltip}>
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle className="context-track" cx="12" cy="12" r="9" pathLength="100" />
      {ratio !== undefined && <circle className="context-fill" cx="12" cy="12" r="9" pathLength="100"
        strokeDasharray={`${Math.min(1, ratio) * 100} 100`} transform="rotate(-90 12 12)" />}
    </svg>
    <span aria-hidden="true">{ratio === undefined ? '—' : `${percentFormat.format(ratio * 100)}%`}</span>
  </span>
}
