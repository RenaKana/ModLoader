const { contextBridge, ipcRenderer } = require('electron');
const BEFORE_CLOSE_REQUEST = 'modloader:before-close-request';
const BEFORE_CLOSE_RESULT = 'modloader:before-close-result';
const BEFORE_CLOSE_READY = 'modloader:before-close-ready';
const CLOSE_CANCELLED = 'modloader:close-cancelled';
const beforeCloseHandlers = new Set();
const closeCancelledHandlers = new Set();
const activeRequests = new Set();

function publishBeforeCloseReadiness() {
  ipcRenderer.send(BEFORE_CLOSE_READY, beforeCloseHandlers.size > 0);
}

ipcRenderer.on(BEFORE_CLOSE_REQUEST, async (_event, requestId) => {
  if (typeof requestId !== 'string' || activeRequests.has(requestId)) return;
  activeRequests.add(requestId);
  const handlers = [...beforeCloseHandlers];
  try {
    const results = await Promise.allSettled(handlers.map(handler => Promise.resolve().then(handler)));
    const success = handlers.length > 0 && results.every(result => result.status === 'fulfilled');
    ipcRenderer.send(BEFORE_CLOSE_RESULT, requestId, success);
  } catch {
    try {
      ipcRenderer.send(BEFORE_CLOSE_RESULT, requestId, false);
    } catch {}
  } finally {
    activeRequests.delete(requestId);
  }
});

ipcRenderer.on(CLOSE_CANCELLED, () => {
  for (const handler of [...closeCancelledHandlers]) {
    try {
      handler();
    } catch {}
  }
});

contextBridge.exposeInMainWorld('modloader', {
  connection: () => ipcRenderer.invoke('connection'),
  pickPath: kind => ipcRenderer.invoke('pick-path', kind),
  openExternal: url => ipcRenderer.invoke('open-external', url),
  exportHtml: (source, action) => ipcRenderer.invoke('export-html', source, action),
  exportMarkdown: (source, title) => ipcRenderer.invoke('export-markdown', source, title),
  saveImage: (source, name) => ipcRenderer.invoke('save-image', source, name),
  openPath: path => ipcRenderer.invoke('open-path', path),
  localResourceAction: (id, action) => ipcRenderer.invoke('local-resource-action', id, action),
  copy: text => ipcRenderer.invoke('copy', text),
  openMiniMonitor: () => ipcRenderer.invoke('mini-monitor:open'),
  publishMiniMonitor: snapshot => ipcRenderer.send('mini-monitor:publish', snapshot),
  getMonitorVisibility: () => ipcRenderer.invoke('mini-monitor:get-visibility'),
  onMonitorVisibility: handler => {
    if (typeof handler !== 'function') throw new TypeError('onMonitorVisibility requires a callback');
    const listener = (_event, visible) => handler(visible);
    ipcRenderer.on('mini-monitor:visibility', listener);
    return () => ipcRenderer.removeListener('mini-monitor:visibility', listener);
  },
  onMiniMonitorAction: handler => {
    if (typeof handler !== 'function') throw new TypeError('onMiniMonitorAction requires a callback');
    const listener = (_event, action) => handler(action);
    ipcRenderer.on('mini-monitor:action', listener);
    return () => ipcRenderer.removeListener('mini-monitor:action', listener);
  },
  onBeforeClose: handler => {
    if (typeof handler !== 'function') throw new TypeError('onBeforeClose requires a callback');
    beforeCloseHandlers.add(handler);
    if (beforeCloseHandlers.size === 1) publishBeforeCloseReadiness();
    return () => {
      if (!beforeCloseHandlers.delete(handler)) return;
      if (beforeCloseHandlers.size === 0) publishBeforeCloseReadiness();
    };
  },
  onWindowHidden: handler => {
    if (typeof handler !== 'function') throw new TypeError('onWindowHidden requires a callback');
    const listener = () => handler();
    ipcRenderer.on('modloader:window-hidden', listener);
    return () => ipcRenderer.removeListener('modloader:window-hidden', listener);
  },
  onCloseCancelled: handler => {
    if (typeof handler !== 'function') throw new TypeError('onCloseCancelled requires a callback');
    closeCancelledHandlers.add(handler);
    return () => closeCancelledHandlers.delete(handler);
  },
});
