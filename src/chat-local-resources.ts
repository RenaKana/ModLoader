export interface LocalResource {
  id: string
  name: string
  path: string
  directory: string
  size: number
  mime: string
  kind: 'image' | 'file'
  can_open: boolean
}

export function localResourceFromResult(result: unknown): LocalResource | undefined {
  if (!result || typeof result !== 'object' || !('resource' in result)) return
  const value = result.resource as Partial<LocalResource> | null
  if (!value || typeof value !== 'object' || typeof value.id !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(value.id)) return
  if (!['name', 'path', 'directory', 'mime'].every(key => typeof value[key as keyof LocalResource] === 'string')) return
  if (typeof value.size !== 'number' || !Number.isSafeInteger(value.size) || value.size < 0 || !['file', 'image'].includes(value.kind || '') || typeof value.can_open !== 'boolean') return
  return value as LocalResource
}

export function localResourceReference(url: string, resources: readonly LocalResource[]): LocalResource | undefined {
  if (!url.startsWith('modloader-resource:')) return
  const id = url.slice('modloader-resource:'.length)
  return resources.find(resource => resource.id === id)
}

export function sameLocalDirectory(left: string, right: string) {
  return left.replaceAll('\\', '/').replace(/\/+$/, '').toLowerCase() === right.replaceAll('\\', '/').replace(/\/+$/, '').toLowerCase()
}
