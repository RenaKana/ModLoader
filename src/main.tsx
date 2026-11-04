import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import MiniMonitor from './MiniMonitor'
import './styles.css'

const mini = new URLSearchParams(window.location.search).get('window') === 'mini-monitor'
if (mini) document.body.classList.add('mini-monitor-body')
createRoot(document.getElementById('root')!).render(
  <React.StrictMode>{mini ? <MiniMonitor /> : <App />}</React.StrictMode>,
)
