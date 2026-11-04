import { useEffect, useState } from 'react'
import { Check, File, FolderOpen, LoaderCircle, ExternalLink } from 'lucide-react'
import { apiFetch, errorMessage } from './api'
import { Button, ErrorNotice, IconButton, Modal, formatBytes } from './components'
import { ImageGallery, GalleryImage } from './ImageViewer'
import { sameLocalDirectory, type LocalResource } from './chat-local-resources'
import './chat-local-resources.css'

export function ChatLocalResource({ resource, directory }: { resource: LocalResource; directory: string }) {
  const authorized = Boolean(directory) && sameLocalDirectory(resource.directory, directory)
  const [opened, setOpened] = useState(false)
  const [current, setCurrent] = useState(resource)
  const [image, setImage] = useState('')
  const [failure, setFailure] = useState('')
  const [busy, setBusy] = useState<'open' | 'reveal' | null>(null)
  const [success, setSuccess] = useState<'open' | 'reveal' | null>(null)
  const endpoint = `/api/chat-tools/resources/${encodeURIComponent(resource.id)}`
  useEffect(() => {
    setOpened(false); setCurrent(resource); setImage(''); setFailure(''); setSuccess(null)
    if (!authorized || resource.kind !== 'image') return
    const abort = new AbortController()
    let url = ''
    void apiFetch(`${endpoint}/content`, { signal: abort.signal }).then(response => response.blob()).then(blob => {
      if (abort.signal.aborted) return
      url = URL.createObjectURL(blob); setImage(url)
    }).catch(error => { if (!abort.signal.aborted) setFailure(errorMessage(error)) })
    return () => { abort.abort(); if (url) URL.revokeObjectURL(url) }
  }, [resource.id, directory, authorized, endpoint])
  useEffect(() => {
    if (!success) return
    const timer = setTimeout(() => setSuccess(null), 1800)
    return () => clearTimeout(timer)
  }, [success])

  async function detail() {
    setOpened(true); setFailure('')
    try { setCurrent(await (await apiFetch(endpoint)).json()) }
    catch (error) { setFailure(errorMessage(error)) }
  }
  async function action(action: 'open' | 'reveal') {
    setBusy(action); setFailure(''); setSuccess(null)
    try {
      if (!window.modloader) throw new Error('请在桌面应用中操作本地附件。')
      const result = await window.modloader.localResourceAction(resource.id, action)
      if (!result.handed_off) throw new Error('系统未接受附件操作。')
      setSuccess(action)
    } catch (error) { setFailure(errorMessage(error)) }
    finally { setBusy(null) }
  }
  const status = !authorized ? '附件所属目录已不再授权' : failure
  const feedbackIcon = (action: 'open' | 'reveal', Icon: typeof FolderOpen) => busy === action ? LoaderCircle : success === action ? Check : Icon
  const reveal = <IconButton icon={feedbackIcon('reveal', FolderOpen)} label={success === 'reveal' ? '已请求文件夹定位' : '在文件夹中定位'} disabled={!authorized || busy !== null}
    className={`chat-resource-reveal ${busy === 'reveal' ? 'spin-icon' : ''}`} data-success={success === 'reveal'} onClick={event => { event.stopPropagation(); void action('reveal') }} />
  return <div className="chat-local-resource" data-kind={resource.kind}>
    {resource.kind === 'image' && authorized && image && <ImageGallery key={`${resource.id}:${directory}`} className="chat-resource-image"><GalleryImage src={image} alt={resource.name} onError={() => { setImage(''); setFailure('图片无法解码或已损坏') }} /></ImageGallery>}
    <div className="chat-resource-card" onClick={() => void detail()}>
      <button type="button" className="chat-resource-detail" aria-label={`附件详情：${resource.name}`}>
        <File size={19} aria-hidden="true" /><span><strong>{resource.name}</strong><small>{formatBytes(resource.size)}</small></span>
      </button>{reveal}
    </div>
    {status && <div className="chat-resource-error" role="alert">{status}</div>}
    {opened && <Modal title="附件详情" onClose={() => setOpened(false)} className="chat-resource-modal" footer={<>
      <Button small icon={feedbackIcon('reveal', FolderOpen)} aria-label={success === 'reveal' ? '已请求定位' : '在文件夹中定位'} disabled={!authorized || busy !== null} data-success={success === 'reveal'} onClick={() => void action('reveal')}>在文件夹中定位</Button>
      <Button small variant="primary" icon={feedbackIcon('open', ExternalLink)} aria-label={success === 'open' ? '已交给默认应用' : '用默认应用打开'} busy={busy === 'open'} disabled={!authorized || !current.can_open || busy !== null} data-success={success === 'open'} onClick={() => void action('open')}>用默认应用打开</Button>
    </>}>
      <dl className="chat-resource-metadata"><dt>文件</dt><dd>{current.name}</dd><dt>位置</dt><dd>{current.directory} / {current.path}</dd><dt>类型</dt><dd>{current.mime}</dd><dt>大小</dt><dd>{formatBytes(current.size)}</dd></dl>
      {!current.can_open && <p className="chat-resource-restriction">此文件类型不允许直接打开，可在文件夹中定位。</p>}
      {status && <ErrorNotice message={status} />}
    </Modal>}
  </div>
}

export function ChatLocalResources({ resources, directory }: { resources?: readonly LocalResource[]; directory: string }) {
  if (!resources?.length) return null
  return <div className="chat-local-resources">{resources.map(resource => <ChatLocalResource key={resource.id} resource={resource} directory={directory} />)}</div>
}
