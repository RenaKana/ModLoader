import { useState } from 'react'
import { Loader2, Zap } from 'lucide-react'
import { errorMessage, isJobActive, request } from './api'
import { Button, ErrorNotice } from './components'
import { useAction } from './hooks'
import type { PageProps, RuntimeInstallation } from './types'

export default function RuntimeStartupPrefetch({ runtime, state, notify, refresh }: Pick<PageProps, 'state' | 'notify' | 'refresh'> & { runtime: RuntimeInstallation }) {
  const [error, setError] = useState('')
  const action = useAction(notify, refresh)
  const capability = runtime.startup_expert_prefetch
  const preparing = state.installations.some(job => job.operation === 'prepare_startup_prefetch' && job.runtime_id === runtime.id && isJobActive(job))
  if (runtime.backend !== 'strata') return null

  async function prepare() {
    setError('')
    await action.run(async () => {
      try { return await request('POST', `/api/runtimes/${runtime.id}/prepare-startup-prefetch`) }
      catch (cause) { setError(errorMessage(cause)); throw cause }
    }, '启动预取准备已开始')
  }

  return <div className="runtime-startup-prefetch" aria-label="启动预取能力">
    <span role="status">{preparing ? <><Loader2 size={13} className="spin" />正在准备启动预取</> : capability?.available ? capability.source === 'legacy' ? '开发版启动预取可用' : '启动预取已就绪' : capability?.message || '启动预取未检查'}</span>
    {capability?.preparable && !capability.available && <Button small variant="ghost" icon={Zap} busy={action.busy} disabled={preparing} onClick={() => void prepare()}>准备启动预取</Button>}
    {error && <ErrorNotice message={error} />}
  </div>
}
