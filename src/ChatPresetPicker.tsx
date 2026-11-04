import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Save } from 'lucide-react'
import { Button, ErrorNotice, Field, IconButton, Modal, Select } from './components'
import { errorMessage } from './api'
import type { PageProps } from './types'
import type { useChatHistory } from './useChatHistory'
import { DEFAULT_WEB_SEARCH_SETTINGS } from './web-search'
import './chat-presets.css'

export default function ChatPresetPicker({ history, disabled, notify }: {
  history: ReturnType<typeof useChatHistory>
  disabled?: boolean
  notify: PageProps['notify']
}) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState('')
  const [useForNew, setUseForNew] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const nameInput = useRef<HTMLInputElement>(null)
  const locked = disabled || !history.presetsReady
  const settings = history.document.settings
  const selected = history.presets.items.find(item => item.id === settings.presetId)
  const modified = Boolean(selected && (selected.system !== settings.system || selected.toolsDraft !== settings.toolsDraft || selected.reasoningEffort !== settings.reasoningEffort || (selected.toolConcurrency ?? DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency) !== (settings.toolConcurrency ?? DEFAULT_WEB_SEARCH_SETTINGS.toolConcurrency)))
  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => nameInput.current?.focus())
    return () => cancelAnimationFrame(frame)
  }, [open])
  function showSave() {
    setName(selected?.name || ''); setUseForNew(Boolean(selected && selected.id === history.presets.default_id)); setError(''); setOpen(true)
  }
  async function save(update: boolean) {
    setBusy(true); setError('')
    try {
      await history.savePreset(name, update ? selected?.id : undefined, useForNew)
      notify(update ? '已更新预设' : '预设已保存', 'success')
      setOpen(false)
    } catch (issue) { setError(errorMessage(issue)) }
    finally { setBusy(false) }
  }
  return <>
    <div className="chat-preset-picker">
      <span className="field-label">对话预设</span>
      <div className="chat-preset-control">
        <Select aria-label="对话预设" value={selected?.id || ''} disabled={locked || busy} onChange={event => {
          try { history.applyPreset(event.target.value) }
          catch (issue) { notify(errorMessage(issue), 'error') }
        }}>
          <option value="">当前会话</option>
          {history.presets.items.map(preset => <option key={preset.id} value={preset.id}>{preset.name}{history.presets.default_id === preset.id ? ' · 默认' : ''}{selected?.id === preset.id && modified ? ' · 已修改' : ''}</option>)}
        </Select>
        <IconButton icon={Save} label="保存对话预设" disabled={locked || busy} onClick={showSave} />
      </div>
    </div>
    {open && createPortal(<Modal title="保存对话预设" className="chat-preset-dialog" dismissible={!busy} onClose={() => setOpen(false)} footer={<>
      <Button disabled={busy} onClick={() => setOpen(false)}>取消</Button>
      {selected && <Button disabled={locked || busy || !name.trim()} onClick={() => void save(false)}>另存为</Button>}
      <Button variant="primary" busy={busy} disabled={locked || !name.trim()} onClick={() => void save(Boolean(selected))}>{selected ? '更新预设' : '保存预设'}</Button>
    </>}>
      {error && <ErrorNotice message={error} />}
      <Field label="预设名称"><input ref={nameInput} value={name} disabled={busy} onChange={event => setName(event.target.value)} onKeyDown={event => {
        if (event.key === 'Enter' && !event.nativeEvent.isComposing && name.trim() && !busy && !locked) { event.preventDefault(); void save(Boolean(selected)) }
      }} /></Field>
      <label className="check-option"><input type="checkbox" checked={useForNew} disabled={busy} onChange={event => setUseForNew(event.target.checked)} /><strong>用于新会话</strong></label>
    </Modal>, document.body)}
  </>
}
