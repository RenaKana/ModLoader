import { FileText, X } from 'lucide-react'
import { IconButton, formatBytes } from './components'
import type { DocumentAttachment } from './chat-history'

export default function ChatDocuments({ documents, onRemove, disabled }: { documents?: DocumentAttachment[]; onRemove?: (id: string) => void; disabled?: boolean }) {
  if (!documents?.length) return null
  return <div className="chat-documents">{documents.map(document => <div className="chat-document" key={document.id}>
    <details><summary><FileText size={18} aria-hidden="true" /><strong>{document.name}</strong><span>{formatBytes(document.size)}</span></summary><pre>{document.text}</pre></details>
    {onRemove && <IconButton icon={X} label={`移除 ${document.name}`} disabled={disabled} onClick={() => onRemove(document.id)} />}
  </div>)}</div>
}
