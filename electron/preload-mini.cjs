const { contextBridge, ipcRenderer } = require('electron');

function subscribe(channel, handler) {
  if (typeof handler !== 'function') throw new TypeError('A callback is required');
  const listener = (_event, value) => handler(value);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

contextBridge.exposeInMainWorld('miniMonitor', {
  ready: () => ipcRenderer.invoke('mini-monitor:ready'),
  onSnapshot: handler => subscribe('mini-monitor:snapshot', handler),
  onAlwaysOnTop: handler => subscribe('mini-monitor:always-on-top', handler),
  onOpacity: handler => subscribe('mini-monitor:opacity', handler),
  onEditing: handler => subscribe('mini-monitor:editing', handler),
  selectInstance: id => ipcRenderer.send('mini-monitor:select-instance', id),
  selectGpu: id => ipcRenderer.send('mini-monitor:select-gpu', id),
  setAlwaysOnTop: value => ipcRenderer.invoke('mini-monitor:set-always-on-top', value),
  setOpacity: value => ipcRenderer.invoke('mini-monitor:set-opacity', value),
  setEditing: value => ipcRenderer.invoke('mini-monitor:set-editing', value),
  fitContent: height => ipcRenderer.invoke('mini-monitor:fit-content', height),
  reportContentHeight: height => ipcRenderer.send('mini-monitor:content-height', height),
  close: () => ipcRenderer.send('mini-monitor:close'),
});
