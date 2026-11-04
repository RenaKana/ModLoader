import { expect, test } from '@playwright/test'
import { EventEmitter } from 'node:events'
import * as fs from 'node:fs'
import * as path from 'node:path'
import * as vm from 'node:vm'

test('native union只用可见非最小化；初始隐藏、最小化、托盘、mini关闭重开及无focus依赖', async () => {
  const handles = new Map<string, Function>(), messages: Array<{ channel: string; value: unknown }> = [], windows: MockWindow[] = []
  class MockWindow extends EventEmitter {
    visible = false; minimized = false; destroyed = false; top = true
    webContents = Object.assign(new EventEmitter(), { mainFrame: {}, isDestroyed: () => false, send: (channel: string, value: unknown) => messages.push({ channel, value }), setWindowOpenHandler: () => {} })
    constructor(..._args: unknown[]) { super(); windows.push(this) }
    isDestroyed() { return this.destroyed }; isVisible() { return this.visible }; isMinimized() { return this.minimized }
    show() { this.visible = true; this.emit('show') }; hide() { this.visible = false; this.emit('hide') }
    minimize() { this.minimized = true; this.emit('minimize') }; restore() { this.minimized = false; this.emit('restore') }
    focus() { this.emit('focus') }; isAlwaysOnTop() { return this.top }; setAlwaysOnTop(value: boolean) { this.top = value }
    getBounds() { return { x: 0, y: 0, width: 440, height: 140 } }; getNormalBounds() { return this.getBounds() }
    setBounds() {}; setOpacity() {}; setMenu() {}; setMenuBarVisibility() {}; async loadFile() {}
    close() { this.emit('close'); this.destroyed = true; this.emit('closed') }; destroy() { this.close() }
  }
  const main = new MockWindow(), exports = { exports: {} as any }, ipcMain = { handle: (name: string, callback: Function) => handles.set(name, callback), on: () => {} }
  const file = path.resolve(__dirname, '../electron/mini-monitor.cjs')
  vm.runInNewContext(fs.readFileSync(file, 'utf8'), { module: exports, __dirname: path.dirname(file), console, process, URL,
    setTimeout: () => 1, clearTimeout: () => {}, require: (name: string) => name === 'electron' ? { BrowserWindow: MockWindow, ipcMain, screen: { getDisplayMatching: () => ({ workArea: { x: 0, y: 0, width: 1920, height: 1080 } }) } }
      : name === 'node:fs' ? { readFileSync: () => { throw Error('no saved state') }, mkdirSync: () => {}, writeFileSync: () => {} } : require(name) })
  const controller = exports.exports.createMiniMonitor({ mainWindow: main, dataRoot: 'fixture-only', projectRoot: path.resolve(__dirname, '..'), isTrustedMainFrame: (event: unknown) => event === main })
  const get = () => handles.get('mini-monitor:get-visibility')!(main)
  expect(handles.has('mini-monitor:get-visibility')).toBe(true); expect(get()).toBe(false)
  expect(() => handles.get('mini-monitor:get-visibility')!({})).toThrow()
  main.show(); expect(get()).toBe(true); main.minimize(); expect(get()).toBe(false)
  const open = async () => { const result = controller.open(); const mini = windows.at(-1)!; await Promise.resolve();
    handles.get('mini-monitor:ready')!({ sender: mini.webContents, senderFrame: mini.webContents.mainFrame }); await result; return mini }
  let mini = await open(); expect(get()).toBe(true)
  const count = messages.length; mini.emit('blur'); expect(messages.length).toBe(count); expect(get()).toBe(true)
  mini.minimize(); expect(get()).toBe(false); mini.restore(); expect(get()).toBe(true)
  mini.hide(); expect(get()).toBe(false); main.restore(); expect(get()).toBe(true)
  main.hide(); expect(get()).toBe(false); mini.show(); expect(get()).toBe(true)
  mini.close(); expect(get()).toBe(false); mini = await open(); expect(get()).toBe(true)
  main.show(); const transitions = messages.filter(message => message.channel === 'mini-monitor:visibility').map(message => message.value)
  expect(transitions).toEqual([true, false, true, false, true, false, true, false, true, false, true])
})
