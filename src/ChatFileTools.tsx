import { useEffect, useRef, useState } from 'react'
import { apiFetch, errorMessage } from './api'
import { Button, ErrorNotice, Modal } from './components'
import type { ToolCall } from './chat-history'
import { toolArguments } from './chat-tools'

interface PreparedWrite { preparation_id: string; directory: string; path: string; operation: 'create' | 'overwrite' | 'edit'; original_content: string | null; content: string }
type Approval = { prepared: PreparedWrite; signal: AbortSignal; resolve: (approved: boolean) => void; committing?: boolean; failure?: string }

export function useChatFileTools(onLateWrite: (path: string) => void) {
  const [approval, setApproval] = useState<Approval | null>(null)
  const [error, setError] = useState('')
  const [writing, setWriting] = useState(false)
  const current = useRef<Approval | null>(null)
  const writes = useRef(new Set<Promise<unknown>>())
  useEffect(() => () => { current.current?.resolve(false) }, [])
  const callApi = async (operation: string, body: unknown, signal?: AbortSignal) => (await apiFetch(`/api/chat-tools/${operation}`, { method: 'POST', body: JSON.stringify(body), signal })).json()

  async function execute(call: ToolCall, directory: string, signal: AbortSignal, conversationId?: string): Promise<unknown> {
    if (!directory.trim()) throw new Error('请先选择本地工具目录。')
    const args = toolArguments(call)
    if (call.function.name === 'list_directory') return callApi('list', { ...args, directory }, signal)
    if (call.function.name === 'read_file') return callApi('read', { ...args, directory }, signal)
    if (call.function.name === 'attach_file') {
      if (!conversationId) throw new Error('请先保存当前对话再引用附件。')
      return callApi('attach', { ...args, directory, conversation_id: conversationId }, signal)
    }
    if (call.function.name !== 'write_file') throw new Error(`未注册的文件工具：${call.function.name}`)
    // The model only prepares a change. The commit ID never enters its context.
    const prepared = await callApi('prepare-write', { ...args, directory }, signal) as PreparedWrite
    let pending: Approval | null = null
    const abort = () => { if (!pending?.committing) pending?.resolve(false) }
    try {
      if (current.current) throw new Error('另一会话正在等待文件写入确认，请稍后重试。')
      const approved = await new Promise<boolean>(resolve => {
        pending = { prepared, signal, resolve }
        current.current = pending; setError(''); setApproval(pending)
        signal.addEventListener('abort', abort, { once: true })
        if (signal.aborted) resolve(false)
      })
      if (signal.aborted) throw signal.reason ?? new DOMException('已停止', 'AbortError')
      return approved ? { path: prepared.path, written: true } : { path: prepared.path, written: false, error: (pending as Approval | null)?.failure || '用户拒绝了这次写入。不要重复请求同一变更。' }
    } finally {
      signal.removeEventListener('abort', abort)
      if (current.current === pending) { current.current = null; setApproval(null) }
      // Also releases preparations after rejection/cancellation; already consumed IDs are harmless.
      await apiFetch(`/api/chat-tools/prepared/${prepared.preparation_id}`, { method: 'DELETE' }).catch(() => {})
    }
  }
  async function approve() {
    const pending = current.current
    if (!pending || writing || pending.signal.aborted) return
    pending.committing = true
    setWriting(true); setError('')
    try {
      // Once confirmed, wait for the commit result rather than cancelling an uncertain write.
      const writing = callApi('write', { preparation_id: pending.prepared.preparation_id, approved: true }, AbortSignal.timeout(30_000))
      writes.current.add(writing)
      try { await writing } finally { writes.current.delete(writing) }
      if (pending.signal.aborted) onLateWrite(pending.prepared.path)
      pending.resolve(true)
    } catch (issue) { pending.failure = errorMessage(issue); setError(pending.failure) }
    finally { pending.committing = false; setWriting(false); if (pending.signal.aborted) pending.resolve(false) }
  }
  const dialog = approval && <Modal dismissible={!writing} title={approval.prepared.operation === 'create' ? '允许创建文件？' : '允许修改文件？'} onClose={() => { if (!writing) approval.resolve(false) }}
    footer={<><Button disabled={writing} onClick={() => approval.resolve(false)}>{error ? '关闭' : '拒绝'}</Button><Button variant="primary" disabled={Boolean(error)} busy={writing} onClick={() => void approve()}>允许写入</Button></>}>
    <p className="chat-file-path">{approval.prepared.directory} / {approval.prepared.path}</p>
    <div className="chat-file-changes">
      {approval.prepared.original_content !== null && <section><h4>修改前</h4><pre>{approval.prepared.original_content}</pre></section>}
      <section><h4>{approval.prepared.original_content === null ? '文件内容' : '修改后'}</h4><pre>{approval.prepared.content}</pre></section>
    </div>
    {error && <ErrorNotice message={error} />}
  </Modal>
  return { execute, dialog, waitForWrites: () => Promise.allSettled([...writes.current]) }
}
