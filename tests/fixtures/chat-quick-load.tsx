import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import ChatReasoningControl from '../../src/ChatReasoningControl'
import Models from '../../src/pages/Models'
import { fetchState, isReady } from '../../src/api'
import { useChatModelLoading } from '../../src/chat-model-loading'
import type { ReasoningEffort } from '../../src/chat-reasoning'
import type { AppState } from '../../src/types'
import '../../src/styles.css'
import '../../src/chat-layout.css'

function Fixture() {
  const [state, setState] = useState<AppState>({ models: [], profiles: [], runtimes: [], instances: [], downloads: [], installations: [], settings: {}, hardware: {}, backends: [] })
  const [page, setPage] = useState('chat')
  const [conversationId, setConversationId] = useState('first')
  const [instanceId, setInstanceId] = useState('')
  const [effort, setEffort] = useState<ReasoningEffort>()
  const [notice, setNotice] = useState('')
  const refresh = async () => setState(await fetchState())
  const { quickLoadOptions, loadProfile, cancelAutoSelect, unloadInstance, unloadingIds, unloadError } = useChatModelLoading(state, refresh, setInstanceId, conversationId, instanceId)
  useEffect(() => { void refresh() }, [])
  const ready = state.instances.filter(instance => isReady(instance.state))
  return <main className="chat-section" style={{ padding: 24, '--chat-ink': '#252528', '--chat-secondary': '#667085' } as React.CSSProperties}>
    <nav style={{ display: 'flex', gap: 8 }}><button onClick={() => setPage('models')}>模型库</button><button onClick={() => setPage('chat')}>测试对话</button><button onClick={() => void refresh()}>刷新夹具</button><button onClick={() => { setConversationId(id => id === 'first' ? 'second' : 'first'); setInstanceId('') }}>切换夹具会话</button></nav>
    {page === 'models' ? <Models state={state} refresh={refresh} notify={setNotice} active stopInstance={async () => { throw new Error('夹具不允许停止实例') }} openLogs={() => {}} navigate={() => setPage('chat')} />
      : <section style={{ marginTop: 370 }}><ChatReasoningControl disabled={false} resetKey={conversationId} modelId={instanceId} modelName={ready.find(instance => instance.id === instanceId)?.alias || '选择模型'}
        modelOptions={ready.map(instance => ({ value: instance.id, label: instance.alias }))} quickLoadOptions={quickLoadOptions} onLoadProfile={loadProfile} onUnloadModel={unloadInstance} unloadingIds={unloadingIds} unloadError={unloadError}
        onModelChange={id => { cancelAutoSelect(); setInstanceId(id) }} capability={ready.some(instance => instance.id === instanceId) ? { kind: 'qwen', levels: [{ value: 'none', label: '关闭' }, { value: 'low', label: '低' }, { value: 'medium', label: '中' }, { value: 'xhigh', label: '高' }] } : undefined}
        value={effort} onChange={setEffort} /><output aria-label="选中实例">{instanceId}</output><output aria-label="选中推理强度">{effort || '默认'}</output></section>}
    {notice && <div role="alert">{notice}</div>}
  </main>
}
createRoot(document.getElementById('root')!).render(<Fixture />)
