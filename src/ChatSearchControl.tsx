import { useEffect, useId, useRef, useState, type CSSProperties } from 'react'
import { Check, Globe2 } from 'lucide-react'
import type { WebSearchMode } from './web-search'

export default function ChatSearchControl({ enabled, mode, disabled, onChange }: {
  enabled: boolean; mode: WebSearchMode; disabled: boolean; onChange: (mode: WebSearchMode | 'off') => void
}) {
  const id = useId(), popup = useRef<HTMLDivElement>(null), anchor = `--search-${id.replace(/[^a-zA-Z0-9-]/g, '')}`
  const [open, setOpen] = useState(false)
  const selected = enabled ? mode : 'off'
  const options = [['off', '关闭'], ['agent', '自主搜索'], ['retrieve', '检索后回答']] as const
  useEffect(() => { if (disabled) popup.current?.hidePopover() }, [disabled])
  return <>
    <button type="button" className="web-search-toggle composer-tool-icon" aria-label="联网搜索" aria-pressed={enabled} aria-haspopup="dialog" aria-expanded={open}
      data-mode={selected} data-tooltip={`联网搜索 · ${options.find(([value]) => value === selected)?.[1]}`} disabled={disabled} popoverTarget={id} style={{ anchorName: anchor } as CSSProperties}><Globe2 size={17} /></button>
    <div ref={popup} id={id} popover="auto" role="dialog" aria-label="联网搜索选项" className="chat-search-popover" style={{ positionAnchor: anchor } as CSSProperties}
      onToggle={event => setOpen((event as unknown as { newState: string }).newState === 'open')}>
      {options.map(([value, label]) => <button key={value} type="button" role="radio" aria-checked={selected === value} disabled={disabled} onClick={() => { onChange(value); popup.current?.hidePopover() }}><span>{label}</span>{selected === value && <Check size={14} />}</button>)}
    </div>
  </>
}
