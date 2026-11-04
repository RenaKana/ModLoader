import { instance } from '@viz-js/viz'

// Each preview owns this worker, so cancellation can stop synchronous WASM layout.
const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<string>) => void) | null
  postMessage: (message: { svg: string } | { failed: true }) => void
}

workerScope.onmessage = async ({ data }) => {
  try {
    const viz = await instance()
    workerScope.postMessage({ svg: viz.renderString(data, { format: 'svg', engine: 'dot' }) })
  } catch {
    workerScope.postMessage({ failed: true })
  }
}
