import { test, expect } from '@playwright/test'
import * as fs from 'node:fs'
import * as path from 'node:path'

const root = path.resolve(__dirname, '..')

test('应用与托盘图标提供 Windows 所需的独立尺寸和透明 PNG 帧', () => {
  const png = fs.readFileSync(path.join(root, 'electron/icon.png'))
  expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([256, 256])
  for (const [file, sizes] of [
    ['icon.ico', [16, 20, 24, 32, 40, 48, 64, 128, 256]],
    ['tray.ico', [16, 20, 24, 32, 40, 48]],
  ] as const) {
    const ico = fs.readFileSync(path.join(root, 'electron', file))
    expect(ico.readUInt16LE(0)).toBe(0)
    expect(ico.readUInt16LE(2)).toBe(1)
    expect(ico.readUInt16LE(4)).toBe(sizes.length)
    sizes.forEach((size, index) => {
      const entry = 6 + index * 16
      expect(ico[entry] || 256).toBe(size)
      expect(ico[entry + 1] || 256).toBe(size)
      const offset = ico.readUInt32LE(entry + 12)
      const frame = ico.subarray(offset, offset + ico.readUInt32LE(entry + 8))
      expect(frame.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      expect([frame.readUInt32BE(16), frame.readUInt32BE(20)]).toEqual([size, size])
      expect(frame[25], `${file} ${size}px alpha channel`).toBe(6)
    })
  }
})

test('Windows 打包嵌入正式 ICO 并保持未签名构建', () => {
  const { build } = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))
  expect(build.win.icon).toBe('electron/icon.ico')
  expect(build.win.signAndEditExecutable).not.toBe(false)
  expect(build.win.signExecutable).toBe(false)
})
