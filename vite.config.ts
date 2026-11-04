import { defineConfig } from 'vite'

export default defineConfig({
  base: './',
  esbuild: { jsx: 'automatic' },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.MODLOADER_MANAGEMENT_URL || 'http://127.0.0.1:12341',
        changeOrigin: true,
        headers: process.env.MODLOADER_DEV_TOKEN
          ? { Authorization: `Bearer ${process.env.MODLOADER_DEV_TOKEN}` }
          : {},
      },
    },
  },
})
