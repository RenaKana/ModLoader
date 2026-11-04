export function readLibraryOrder(key: string): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || '[]')
    return Array.isArray(value) ? [...new Set(value.filter((id): id is string => typeof id === 'string'))] : []
  } catch { return [] }
}

export function orderedItems<T extends { id: string }>(items: T[], order: string[]): T[] {
  const positions = new Map(order.map((id, index) => [id, index]))
  return [...items].sort((a, b) => (positions.get(a.id) ?? order.length) - (positions.get(b.id) ?? order.length))
}

// Reorder only visible slots: filtered-out models and other models' profiles
// keep their positions. New registrations keep their original relative order.
export function mergeVisibleOrder<T extends { id: string }>(items: T[], visible: T[]): string[] {
  const ids = new Set(visible.map(item => item.id))
  let index = 0
  return items.map(item => ids.has(item.id) ? visible[index++].id : item.id)
}
