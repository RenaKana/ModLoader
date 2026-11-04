import { useEffect, useState } from 'react'
import type { RuntimeInstance } from './types'
import './model-loading-status.css'

export default function ModelLoadingStatus({ instance, pendingSince }: { instance?: RuntimeInstance; pendingSince?: number }) {
  const timed = !!pendingSince || !!instance && ['preparing', 'starting', 'loading'].includes(instance.state)
  const waiting = timed || instance?.state === 'stopping'
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!timed) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [timed, instance?.id, pendingSince])
  if (!waiting) return null
  const since = pendingSince ?? Date.parse(instance?.started_at || '')
  const seconds = Math.max(0, Math.floor((now - since) / 1000))
  const elapsed = seconds < 60 ? `${seconds} 秒` : seconds < 3600 ? `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒` : `${Math.floor(seconds / 3600)} 时 ${Math.floor(seconds % 3600 / 60)} 分`
  const stage = pendingSince ? '检查启动条件' : instance?.state === 'stopping' ? '正在停止'
    : instance?.phase === 'validating' ? '检查运行文件' : instance?.phase === 'prepared' ? '运行文件已准备'
      : instance?.state === 'preparing' ? '准备运行文件' : instance?.phase === 'launching' ? '启动后端' : '加载模型'
  const message = pendingSince ? '' : instance?.message?.trim()
  return <div className="profile-preparing model-loading-status" role="status">
    <div className="model-loading-heading"><span>{stage}</span>{timed && Number.isFinite(since) && <span className="model-loading-elapsed" aria-live="off">已用 {elapsed}</span>}</div>
    {message && message !== stage && <p>{message}</p>}
    <div className="progress-track indeterminate" aria-hidden="true"><span /></div>
  </div>
}
