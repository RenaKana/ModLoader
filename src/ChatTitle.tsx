import { memo } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { Nodes, PhrasingContent, Root } from 'mdast'
import { remarkMessageSyntax, remarkTexDelimiters } from './markdown-syntax'
import 'katex/dist/katex.min.css'
import './chat-title.css'

const text = (value: string): PhrasingContent => ({ type: 'text', value })

// Parse with the message syntax, then retain only title-sized inline content.
// Links/images contribute their labels, never interactive elements or resources.
function inlineTitle(nodes: Nodes[]): PhrasingContent[] {
  return nodes.flatMap((node): PhrasingContent[] => {
    if (node.type === 'text') return [text(node.value)]
    if (node.type === 'inlineCode' || node.type === 'code') return [{ type: 'inlineCode', value: node.value }]
    if (node.type === 'inlineMath' || node.type === 'math') return [{
      type: 'inlineMath', value: node.value,
      data: { hName: 'code', hProperties: { className: ['language-math', 'math-inline'] }, hChildren: [{ type: 'text', value: node.value }] },
    }]
    if (node.type === 'strong' || node.type === 'emphasis' || node.type === 'delete') return [{ ...node, children: inlineTitle(node.children) }]
    if (node.type === 'image' || node.type === 'imageReference') return [text(node.alt || '')]
    if (node.type === 'definition' || node.type === 'footnoteDefinition' || node.type === 'html') return []
    if (node.type === 'break' || node.type === 'thematicBreak') return [text(' ')]
    if ('children' in node) {
      const children = inlineTitle(node.children)
      return node.type === 'link' || node.type === 'linkReference' ? children : [...children, text(' ')]
    }
    return []
  })
}

function remarkTitleInline() {
  return (tree: Root) => { tree.children = [{ type: 'paragraph', children: inlineTitle(tree.children) }] }
}

const plugins = [remarkGfm, remarkMath, remarkTexDelimiters, remarkMessageSyntax, remarkTitleInline] as import('unified').PluggableList

export const ChatTitle = memo(function ChatTitle({ text }: { text: string }) {
  return <span className="chat-title"><Markdown remarkPlugins={plugins} rehypePlugins={[[rehypeKatex, { trust: false, strict: false }]]} skipHtml urlTransform={() => ''} components={{ p: ({ children }) => <>{children}</> }}>{text}</Markdown></span>
})
