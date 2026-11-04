import { expect, type Page } from '@playwright/test'

export const chatCapsule = (page: Page) => page.getByRole('button', { name: /^对话模型与推理强度：/ })
export const chatModel = (page: Page) => page.getByRole('button', { name: '选择对话模型', exact: true, includeHidden: true })
export const chatModelList = (page: Page) => page.getByRole('listbox', { name: '对话模型', exact: true })

export async function disableWebSearch(page: Page) {
  const toggle = page.getByRole('button', { name: '联网搜索', exact: true })
  if (await toggle.getAttribute('aria-pressed') === 'true') { await toggle.click(); await page.getByRole('dialog', { name: '联网搜索选项' }).getByRole('radio', { name: '关闭', exact: true }).click() }
  await expect(toggle).toHaveAttribute('aria-pressed', 'false')
}

export async function openChatControls(page: Page) {
  const panel = page.getByRole('dialog', { name: '模型与推理强度', exact: true })
  const element = page.locator('.chat-reasoning-popover')
  if (!await element.evaluate(node => node.matches(':popover-open'))) await chatCapsule(page).click()
  await expect.poll(() => element.evaluate(node => node.matches(':popover-open'))).toBe(true)
  await expect(panel).toBeVisible()
  await expect(panel).toHaveCSS('opacity', '1')
  return panel
}

export async function selectChatModel(page: Page, id: string, keepOpen = false) {
  const panel = await openChatControls(page)
  await chatModel(page).click()
  await chatModelList(page).locator(`[data-model-id=${JSON.stringify(id)}]`).click()
  await expect(chatModel(page)).toHaveJSProperty('value', id)
  await expect(panel.locator('.chat-reasoning-view')).toHaveAttribute('aria-hidden', 'false')
  await expect(panel).toBeVisible()
  if (!keepOpen) { await page.keyboard.press('Escape'); await expect(panel).toBeHidden() }
}

export async function setChatReasoning(page: Page, label: string) {
  const panel = await openChatControls(page)
  const slider = panel.getByRole('slider', { name: '推理强度档位', exact: true })
  if (await slider.getAttribute('min') === '0') {
    await slider.focus(); await page.keyboard.press('Home')
    for (let step = 0; step < ['默认', '关闭', '低', '中', '高'].indexOf(label); step++) await page.keyboard.press('ArrowRight')
  } else if (label === '默认' || label === '关闭') await panel.getByRole('button', { name: label, exact: true }).click()
  else await panel.getByRole('spinbutton', { name: '自定义推理强度', exact: true }).fill(String(({ 低: 50, 中: 62, 高: 75, 最高: 100 } as Record<string, number>)[label]))
}
