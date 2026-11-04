const { BrowserWindow, ipcMain, screen } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

const CHANNEL = {
  open: 'mini-monitor:open',
  publish: 'mini-monitor:publish',
  action: 'mini-monitor:action',
  ready: 'mini-monitor:ready',
  snapshot: 'mini-monitor:snapshot',
  alwaysOnTop: 'mini-monitor:always-on-top',
  opacity: 'mini-monitor:opacity',
  editing: 'mini-monitor:editing',
  selectInstance: 'mini-monitor:select-instance',
  selectGpu: 'mini-monitor:select-gpu',
  setAlwaysOnTop: 'mini-monitor:set-always-on-top',
  setOpacity: 'mini-monitor:set-opacity',
  setEditing: 'mini-monitor:set-editing',
  fitContent: 'mini-monitor:fit-content',
  contentHeight: 'mini-monitor:content-height',
  close: 'mini-monitor:close',
  getVisibility: 'mini-monitor:get-visibility',
  visibility: 'mini-monitor:visibility',
};
const MIN_WIDTH = 220;
const MIN_HEIGHT = 100;
const EDITING_HEIGHT = 420;
const STARTUP_TIMEOUT_MS = 15000;
// Windows' floating level reorders behind the taskbar and can drop topmost state.
const ALWAYS_ON_TOP_LEVEL = process.platform === 'win32' ? 'normal' : 'floating';

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && Object.getPrototypeOf(value) === Object.prototype;
}

function readState(statePath) {
  try {
    const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));
    return isRecord(state) ? state : {};
  } catch {
    return {};
  }
}

function validBounds(bounds) {
  return isRecord(bounds) && ['x', 'y', 'width', 'height'].every(key => Number.isSafeInteger(bounds[key]))
    && bounds.width > 0 && bounds.height > 0;
}

function fitBounds(bounds) {
  const area = screen.getDisplayMatching(bounds).workArea;
  const width = Math.min(area.width, Math.max(MIN_WIDTH, Math.round(bounds.width)));
  const height = Math.min(area.height, Math.max(MIN_HEIGHT, Math.round(bounds.height)));
  return {
    x: Math.max(area.x, Math.min(Math.round(bounds.x), area.x + area.width - width)),
    y: Math.max(area.y, Math.min(Math.round(bounds.y), area.y + area.height - height)),
    width,
    height,
  };
}

function createMiniMonitor({ mainWindow, dataRoot, projectRoot, icon, devUrl, isTrustedMainFrame }) {
  const statePath = path.join(dataRoot, 'desktop', 'mini-monitor.json');
  const state = readState(statePath);
  let miniWindow = null;
  let opening = null;
  let startup = null;
  let snapshot = null;
  let normalBounds = validBounds(state.bounds) ? state.bounds : null;
  let fitContentEnabled = typeof state.fitContent === 'boolean' ? state.fitContent : !normalBounds;
  let contentHeight = null;
  let alwaysOnTop = typeof state.alwaysOnTop === 'boolean' ? state.alwaysOnTop : true;
  let opacity = Number.isFinite(state.opacity) && state.opacity >= 0 && state.opacity <= 1 ? state.opacity : 1;
  let editing = false;
  let saveTimer = null;

  const entryPath = path.join(projectRoot, 'dist', 'index.html');
  const entryUrl = new URL(devUrl || pathToFileURL(entryPath).href);
  entryUrl.searchParams.set('window', 'mini-monitor');

  const windowVisible = win => !!win && !win.isDestroyed() && win.isVisible() && !win.isMinimized();
  const monitorVisible = () => windowVisible(mainWindow) || windowVisible(miniWindow);
  let visibility = monitorVisible();
  function notifyVisibility() {
    const next = monitorVisible();
    if (next === visibility) return;
    visibility = next;
    if (!mainWindow.isDestroyed() && !mainWindow.webContents.isDestroyed()) mainWindow.webContents.send(CHANNEL.visibility, next);
  }
  function watchVisibility(win) {
    for (const event of ['show', 'hide', 'minimize', 'restore', 'closed']) win.on(event, notifyVisibility);
  }
  watchVisibility(mainWindow);

  function isTrustedMiniFrame(event) {
    return !!miniWindow && !miniWindow.isDestroyed() && !miniWindow.webContents.isDestroyed()
      && event.sender === miniWindow.webContents && event.senderFrame === miniWindow.webContents.mainFrame;
  }

  function requireMiniFrame(event) {
    if (!isTrustedMiniFrame(event)) throw new Error('只允许从监控小窗执行此操作。');
  }

  function notify(channel, value) {
    if (!miniWindow || miniWindow.isDestroyed() || miniWindow.webContents.isDestroyed()) return;
    miniWindow.webContents.send(channel, value);
  }

  function saveState() {
    clearTimeout(saveTimer);
    saveTimer = null;
    if (!normalBounds) return;
    fs.mkdirSync(path.dirname(statePath), { recursive: true });
    fs.writeFileSync(statePath, JSON.stringify({ bounds: normalBounds, alwaysOnTop, opacity, fitContent: fitContentEnabled }));
  }

  function saveStateFromWindowEvent() {
    try {
      saveState();
    } catch (error) {
      console.error('监控小窗设置保存失败：', error);
    }
  }

  function rememberBounds() {
    if (!miniWindow || miniWindow.isDestroyed()) return;
    const current = miniWindow.getNormalBounds();
    // The expanded editor is temporary; moving it may still update the position.
    normalBounds = editing && normalBounds
      ? { ...current, width: normalBounds.width, height: normalBounds.height }
      : current;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveStateFromWindowEvent, 200);
  }

  function applyContentHeight() {
    if (!fitContentEnabled || editing || contentHeight === null || !miniWindow || miniWindow.isDestroyed()) return;
    const current = miniWindow.getNormalBounds();
    const frameHeight = current.height - miniWindow.getContentBounds().height;
    const height = Math.ceil(contentHeight * miniWindow.webContents.getZoomFactor()) + frameHeight;
    const next = fitBounds({ ...current, height });
    normalBounds = next;
    if (current.height !== next.height || current.x !== next.x || current.y !== next.y) miniWindow.setBounds(next);
  }

  function setEditing(value) {
    const changed = editing !== value;
    if (changed) {
      const current = miniWindow.getNormalBounds();
      if (value) {
        normalBounds = current;
        editing = true;
        miniWindow.setBounds(fitBounds({ ...current, height: Math.max(EDITING_HEIGHT, current.height) }));
      } else {
        editing = false;
        normalBounds = fitBounds({ ...current, width: normalBounds.width, height: normalBounds.height });
        miniWindow.setBounds(normalBounds);
      }
    }
    miniWindow.setOpacity(editing ? 1 : opacity);
    notify(CHANNEL.editing, editing);
    if (changed && !editing) saveState();
  }

  function invalidateSnapshot() {
    snapshot = null;
    notify(CHANNEL.snapshot, null);
  }

  function isEntryUrl(url) {
    try {
      const target = new URL(url);
      target.hash = '';
      const expected = new URL(entryUrl);
      expected.hash = '';
      return target.href === expected.href;
    } catch {
      return false;
    }
  }

  async function createWindow() {
    if (!normalBounds) {
      const area = screen.getDisplayMatching(mainWindow.getBounds()).workArea;
      normalBounds = { x: area.x + area.width - 464, y: area.y + 24, width: 440, height: 140 };
    }
    normalBounds = fitBounds(normalBounds);
    const area = screen.getDisplayMatching(normalBounds).workArea;
    const win = new BrowserWindow({
      ...normalBounds,
      minWidth: Math.min(MIN_WIDTH, area.width),
      minHeight: Math.min(MIN_HEIGHT, area.height),
      title: 'ModLoader · 监控小窗',
      icon,
      show: false,
      frame: false,
      resizable: true,
      maximizable: false,
      fullscreenable: false,
      backgroundColor: '#f7f9fb',
      webPreferences: {
        preload: path.join(__dirname, 'preload-mini.cjs'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        backgroundThrottling: false,
        partition: 'persist:mini-monitor',
      },
    });
    // Windows may adjust the initial frameless height; restore the saved outer size exactly.
    win.setBounds(normalBounds);
    miniWindow = win;
    watchVisibility(win);
    editing = false;
    win.setMenu(null);
    win.setMenuBarVisibility(false);
    if (win.isAlwaysOnTop() !== alwaysOnTop) win.setAlwaysOnTop(alwaysOnTop, ALWAYS_ON_TOP_LEVEL);
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    const preventUntrustedNavigation = (event, url) => {
      if (!isEntryUrl(event.url || url)) event.preventDefault();
    };
    win.webContents.on('will-navigate', preventUntrustedNavigation);
    win.webContents.on('will-redirect', preventUntrustedNavigation);
    win.webContents.on('will-frame-navigate', event => {
      if (!event.isMainFrame || !isEntryUrl(event.url)) event.preventDefault();
    });
    win.on('move', rememberBounds);
    win.on('resize', rememberBounds);
    // Electron emits will-resize for user drags, not for setBounds (fit/editor restoration).
    win.on('will-resize', (_event, bounds) => {
      if (!editing && bounds.height !== win.getBounds().height) fitContentEnabled = false;
    });
    win.on('always-on-top-changed', (_event, value) => {
      alwaysOnTop = value;
      notify(CHANNEL.alwaysOnTop, value);
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveStateFromWindowEvent, 200);
    });
    win.on('close', () => {
      rememberBounds();
      saveStateFromWindowEvent();
    });
    win.once('closed', () => {
      if (startup?.window === win) startup.reject(new Error('监控小窗已关闭。'));
      if (miniWindow === win) miniWindow = null;
      editing = false;
    });

    const ready = new Promise((resolve, reject) => {
      startup = { window: win, resolve, reject };
    });
    const timer = setTimeout(() => {
      if (startup?.window === win) startup.reject(new Error('监控小窗启动超时，请重试。'));
    }, STARTUP_TIMEOUT_MS);
    win.webContents.on('preload-error', (_event, _preloadPath, error) => {
      if (startup?.window === win) startup.reject(new Error(`监控小窗启动失败：${error.message}`));
    });
    win.webContents.on('render-process-gone', (_event, details) => {
      if (startup?.window === win) startup.reject(new Error(`监控小窗进程已停止（${details.reason}）。`));
      if (!win.isDestroyed()) win.close();
    });
    try {
      const loaded = Promise.resolve().then(() => {
        setEditing(opacity === 0);
        return devUrl
          ? win.loadURL(entryUrl.href)
          : win.loadFile(entryPath, { query: { window: 'mini-monitor' } });
      });
      await Promise.all([loaded, ready]);
      if (win.isDestroyed()) throw new Error('监控小窗已关闭。');
      win.show();
      win.focus();
    } catch (error) {
      if (!win.isDestroyed()) win.destroy();
      throw error;
    } finally {
      clearTimeout(timer);
      if (startup?.window === win) startup = null;
    }
  }

  async function open() {
    if (opening) return opening;
    if (miniWindow && !miniWindow.isDestroyed()) {
      if (opacity === 0) setEditing(true);
      if (miniWindow.isMinimized()) miniWindow.restore();
      miniWindow.show();
      miniWindow.focus();
      return;
    }
    opening = createWindow();
    try {
      await opening;
    } finally {
      opening = null;
    }
  }

  function select(event, type, id) {
    if (!isTrustedMiniFrame(event) || typeof id !== 'string' || !snapshot) return;
    const options = type === 'instance' ? snapshot.instances : snapshot.gpus;
    if (!Array.isArray(options) || !options.some(option => isRecord(option) && option.id === id)) return;
    if (mainWindow.isDestroyed() || mainWindow.webContents.isDestroyed()) return;
    mainWindow.webContents.send(CHANNEL.action, { type, id });
  }

  ipcMain.handle(CHANNEL.open, event => {
    if (!isTrustedMainFrame(event)) throw new Error('只允许从主界面打开监控小窗。');
    return open();
  });
  ipcMain.handle(CHANNEL.getVisibility, event => {
    if (!isTrustedMainFrame(event)) throw new Error('只允许从主界面读取监控窗口状态。');
    return monitorVisible();
  });
  ipcMain.on(CHANNEL.publish, (event, value) => {
    if (!isTrustedMainFrame(event) || !isRecord(value)) return;
    snapshot = value;
    notify(CHANNEL.snapshot, snapshot);
  });
  ipcMain.handle(CHANNEL.ready, event => {
    requireMiniFrame(event);
    if (startup?.window === miniWindow) startup.resolve();
    return { snapshot, alwaysOnTop, opacity, editing };
  });
  ipcMain.on(CHANNEL.selectInstance, (event, id) => select(event, 'instance', id));
  ipcMain.on(CHANNEL.selectGpu, (event, id) => select(event, 'gpu', id));
  ipcMain.handle(CHANNEL.setAlwaysOnTop, (event, value) => {
    requireMiniFrame(event);
    if (typeof value !== 'boolean') throw new Error('置顶状态无效。');
    miniWindow.setAlwaysOnTop(value, ALWAYS_ON_TOP_LEVEL);
    alwaysOnTop = miniWindow.isAlwaysOnTop();
    saveState();
    return alwaysOnTop;
  });
  ipcMain.handle(CHANNEL.setEditing, (event, value) => {
    requireMiniFrame(event);
    if (typeof value !== 'boolean') throw new Error('编辑状态无效。');
    setEditing(value);
  });
  ipcMain.on(CHANNEL.contentHeight, (event, height) => {
    if (!isTrustedMiniFrame(event) || !Number.isFinite(height) || height <= 0) return;
    contentHeight = height;
    applyContentHeight();
  });
  ipcMain.handle(CHANNEL.fitContent, (event, height) => {
    requireMiniFrame(event);
    if (!Number.isFinite(height) || height <= 0) throw new Error('内容高度无效。');
    contentHeight = height;
    fitContentEnabled = true;
    applyContentHeight();
    saveState();
  });
  ipcMain.handle(CHANNEL.setOpacity, (event, value) => {
    requireMiniFrame(event);
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error('窗口不透明度必须在 0 到 1 之间。');
    opacity = value;
    miniWindow.setOpacity(editing ? 1 : opacity);
    notify(CHANNEL.opacity, opacity);
    saveState();
    return opacity;
  });
  ipcMain.on(CHANNEL.close, event => {
    if (isTrustedMiniFrame(event)) miniWindow.close();
  });

  mainWindow.webContents.on('did-start-navigation', details => {
    if (details.isMainFrame && !details.isSameDocument) invalidateSnapshot();
  });
  mainWindow.webContents.on('render-process-gone', invalidateSnapshot);
  mainWindow.webContents.once('destroyed', invalidateSnapshot);

  return { open };
}

module.exports = { createMiniMonitor };
