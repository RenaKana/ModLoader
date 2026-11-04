import { expect, test } from '@playwright/test'
import { EventEmitter } from 'node:events'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as vm from 'node:vm'

const root = path.resolve(__dirname, '..')

test('主窗口和监控小窗使用主图，托盘独立加载 Windows ICO', async () => {
  const windows: FakeWindow[] = [], trayImages: unknown[] = [], errors: string[] = []
  const handlers = new Map<string, Function>()
  const area = { x: 0, y: 0, width: 1920, height: 1080 }
  class FakeWindow extends EventEmitter {
    visible = false
    bounds: Record<string, number>
    webContents = Object.assign(new EventEmitter(), {
      mainFrame: {}, isDestroyed: () => false, send() {}, setWindowOpenHandler() {}, getZoomFactor: () => 1,
    })
    constructor(public options: any) { super(); this.bounds = options; windows.push(this) }
    getBounds() { return this.bounds }; getNormalBounds() { return this.bounds }
    setBounds(bounds: Record<string, number>) { this.bounds = bounds }
    isDestroyed() { return false }; isVisible() { return this.visible }; isMinimized() { return false }
    isMaximized() { return false }; isFullScreen() { return false }; isAlwaysOnTop() { return true }
    show() { this.visible = true; this.emit('show') }; focus() {}; setOpacity() {}; setMenu() {}; setMenuBarVisibility() {}
    async loadFile() {}
  }
  const fakeFs = { existsSync: () => true, readFileSync: () => { throw Error('isolated data has no saved state') }, mkdirSync() {}, writeFileSync() {} }
  const fakeElectron = {
    app: { isPackaged: false, setPath() {}, requestSingleInstanceLock: () => true, on() {}, whenReady: () => Promise.resolve() },
    BrowserWindow: FakeWindow,
    Tray: class extends EventEmitter {
      constructor(source: unknown) { super(); trayImages.push(source) }
      setToolTip() {}; setContextMenu() {}
    },
    Menu: { buildFromTemplate: (items: unknown) => items, setApplicationMenu() {} },
    nativeImage: { createFromPath: (source: string) => ({ source }) },
    ipcMain: { handle: (name: string, callback: Function) => handlers.set(name, callback), on() {} },
    screen: { getDisplayMatching: () => ({ workArea: area }), getPrimaryDisplay: () => ({ workArea: area }) },
    dialog: { showErrorBox: (title: string, message: string) => errors.push(`${title}: ${message}`) },
  }
  const loadModule = (filename: string) => {
    const module = { exports: {} as any }
    const dependencies: Record<string, unknown> = {
      electron: fakeElectron, 'node:fs': fakeFs,
      'node:child_process': { spawn: () => Object.assign(new EventEmitter(), { stderr: new EventEmitter() }) },
      'node:readline': { createInterface: () => {
        const reader = new EventEmitter()
        queueMicrotask(() => reader.emit('line', JSON.stringify({ event: 'ready', port: 12341 })))
        return reader
      } },
    }
    vm.runInNewContext(fs.readFileSync(filename, 'utf8'), {
      module, __dirname: path.dirname(filename), console, URL, setTimeout, clearTimeout,
      process: { env: { MODLOADER_DATA_DIR: path.join(root, '.artifacts/codex/branding-icons/vm-data') }, resourcesPath: root, platform: process.platform },
      require: (name: string) => name === './mini-monitor.cjs' ? loadModule(path.join(root, 'electron/mini-monitor.cjs')) : dependencies[name] ?? require(name),
    })
    return module.exports
  }
  loadModule(path.join(root, 'electron/main.cjs'))
  await expect.poll(() => ({ windows: windows.length, errors })).toEqual({ windows: 1, errors: [] })
  const main = windows[0]
  const opening = handlers.get('mini-monitor:open')!({ sender: main.webContents, senderFrame: main.webContents.mainFrame })
  expect(windows).toHaveLength(2)
  const mini = windows[1]
  await handlers.get('mini-monitor:ready')!({ sender: mini.webContents, senderFrame: mini.webContents.mainFrame })
  await opening

  expect(windows.map(window => window.options.icon.source)).toEqual([
    path.join(root, 'electron/icon.png'), path.join(root, 'electron/icon.png'),
  ])
  expect(windows.every(window => window.visible)).toBe(true)
  expect(trayImages).toEqual([path.join(root, 'electron/tray.ico')])
})
