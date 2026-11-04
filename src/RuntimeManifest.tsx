import type { RuntimeManifest as Manifest } from './types'
import { Badge, formatBytes } from './components'

const fileStates: Record<string, string> = {
  pending: '待下载', queued: '待下载', downloading: '下载中', running: '下载中',
  completed: '已完成', complete: '已完成', downloaded: '已完成', cached: '已缓存',
  failed: '失败', error: '失败', skipped: '已跳过', paused: '已暂停',
}

export default function RuntimeManifest({ manifest, title = true }: { manifest: Manifest; title?: boolean }) {
  const files = manifest.files || []
  if (!files.length) return <div className="runtime-manifest">{title && <h4>{manifest.title}</h4>}<p className="runtime-subtle">此阶段无需下载文件。</p></div>
  const unknown = files.filter(file => file.size_bytes === null || file.size_bytes === undefined).length
  const known = files.reduce((total, file) => total + (file.size_bytes ?? 0), 0)
  const downloads = files.filter(file => !file.cached)
  const downloadUnknown = downloads.filter(file => file.size_bytes === null || file.size_bytes === undefined).length
  const downloadKnown = downloads.reduce((total, file) => total + (file.size_bytes ?? 0), 0)
  const sizeLabel = files.length && unknown === files.length ? '文件大小均未知' : unknown ? `文件已知大小 ${formatBytes(known)} · ${unknown} 个大小未知` : `文件总大小 ${formatBytes(known)}`
  const downloadLabel = !downloads.length ? '无需下载，全部已缓存' : downloadUnknown === downloads.length ? `需下载 ${downloads.length} 个文件，大小未知` : downloadUnknown ? `需下载：已知 ${formatBytes(downloadKnown)} · 另 ${downloadUnknown} 个大小未知` : `需下载 ${formatBytes(downloadKnown)}`
  return <div className="runtime-manifest">
    {title && <h4>{manifest.title}</h4>}
    <div className="runtime-file-summary"><span>{files.length} 个文件 · {files.filter(file => file.cached).length} 个已缓存</span><span>{sizeLabel}</span><span className="runtime-download-summary">{downloadLabel}</span></div>
    <table className="runtime-file-table"><thead><tr><th>文件 / 用途</th><th>大小</th><th>状态</th></tr></thead><tbody>{files.map(file => {
      const status = file.status || (file.cached ? 'cached' : 'pending')
      const done = ['completed', 'complete', 'downloaded', 'cached'].includes(status)
      return <tr key={file.id}><td><strong className="mono">{file.name}</strong><span>{file.purpose}</span></td><td>{formatBytes(file.size_bytes)}</td><td><Badge tone={done ? 'green' : ['error', 'failed'].includes(status) ? 'red' : status === 'downloading' ? 'blue' : 'neutral'}>{fileStates[status] || status}</Badge>{file.cached && status !== 'cached' && <span className="runtime-cache-label">缓存可复用</span>}</td></tr>
    })}</tbody></table>
  </div>
}
