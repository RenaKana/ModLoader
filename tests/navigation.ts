import type { Page } from '@playwright/test'

// Unrelated UI checks keep navigation pinned; hover/pin behavior is covered in ui-header.
export async function pinNavigation(page: Page) {
  await page.locator('.app-header').waitFor({ state: 'attached' })
  const pin = page.getByRole('button', { name: '钉住顶部栏', exact: true })
  if (await pin.isVisible()) await pin.click()
}
