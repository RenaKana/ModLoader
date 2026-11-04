import { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import './operation-status.css'

export type OperationProgress = { stage: string; message: string; completed?: number; total?: number }

export default function OperationStatus({ progress, startedAt, fallback }: {
  progress?: OperationProgress | null; startedAt: number; fallback: string
}) {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [startedAt])
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000))
  const counted = progress?.completed !== undefined && progress.total !== undefined && progress.total > 0
  const message = progress?.message || fallback
  return <span className="operation-status" role="status">
    <Loader2 size={13} className="spin" aria-hidden="true" />
    <span className="operation-status-message" title={message}>{message}</span>
    {counted && <><progress aria-label={message} value={progress!.completed} max={progress!.total} />
      <span className="operation-status-count">{progress!.completed} / {progress!.total}</span></>}
    {seconds > 0 && <span className="operation-status-time" aria-hidden="true">· {seconds < 60 ? `${seconds} 秒` : `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`}</span>}
  </span>
}
