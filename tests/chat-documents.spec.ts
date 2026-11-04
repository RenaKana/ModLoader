import { test, expect } from '@playwright/test'
import { conversationMarkdown } from '../src/chat-export'
import { emptyConversation } from '../src/chat-history'

test('Markdown 导出保留文档名称和全文，并用更长代码围栏包住正文反引号', () => {
  const document = {
    id: 'doc-1',
    name: '接口说明.md',
    size: 64,
    text: '文档开头。\n\n```ts\nconst marker = "`";\n```\n\n保留在围栏之后的完整结尾。',
  }
  const conversation = emptyConversation()
  conversation.messages = [{
    id: 'user-1', role: 'user', text: '', images: [], documents: [document],
    createdAt: 100, status: 'completed',
  }]

  const markdown = conversationMarkdown(conversation)
  const fence = '````'
  expect(markdown).toContain('### 附件 · 接口说明.md')
  expect(markdown).toContain(`${fence}text\n${document.text}\n${fence}`)
})
