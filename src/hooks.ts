import { useState } from 'react'
import { errorMessage } from './api'

export type Notify = (message: string, tone?: 'success' | 'error' | 'info') => void
export function useAction(notify: Notify, refresh?: () => Promise<void>) {
  const [busy, setBusy] = useState(false)
  async function run<T>(task: () => Promise<T>, message?: string): Promise<{ ok: true; data: T } | { ok: false }> {
    setBusy(true)
    try {
      const data = await task()
      if (refresh) await refresh()
      if (message) notify(message, 'success')
      return { ok: true, data }
    } catch (error) {
      notify(errorMessage(error), 'error')
      return { ok: false }
    } finally { setBusy(false) }
  }
  return { busy, run }
}
