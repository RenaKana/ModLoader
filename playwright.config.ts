import { defineConfig } from '@playwright/test'

// Fixture suites never request administrator access. Real hardware acceptance
// explicitly sets MODLOADER_SENSOR_AUTO_START=1 before launching Electron.
process.env.MODLOADER_SENSOR_AUTO_START ??= '0'

export default defineConfig({ testDir: './tests' })
