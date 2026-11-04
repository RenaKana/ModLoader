import type { RuntimeInstallation } from './types'
import { runtimeDistribution, runtimePlatform } from './platforms'

export function runtimeVersion(runtime: RuntimeInstallation) {
  if (['unavailable', 'error'].includes(String(runtime.metadata.version_status))) return '版本无法读取'
  const version = runtime.version?.trim()
  if (!version || version === 'unknown') return '版本未知'
  const build = /\bbuild\s+(\d+)/i.exec(version)
  const commit = /\bcommit\s+([a-f0-9]+)/i.exec(version)
  return build ? `b${build[1]}${commit ? ` · ${commit[1].slice(0, 7)}` : ''}` : version
}

export function runtimeDisplayName(runtime: RuntimeInstallation) {
  if (runtime.backend !== 'strata') return runtime.name
  const requested = runtime.metadata.requested_version
  const tag = (runtime.metadata.provenance as { tag?: unknown } | undefined)?.tag
  const release = ((typeof requested === 'string' && requested.trim()) || (typeof tag === 'string' ? tag.trim() : '')).replace(/^v(?=\d)/, '')
  if (!release) return runtime.name
  const name = runtime.name.trim()
  const generated = !name || ['Strata', `Strata · ${runtime.version}`, `Strata · ${release}`].includes(name)
  return `${generated ? '' : `${name} · `}Strata ${release} · 引擎 ${runtimeVersion(runtime)}`
}

export function runtimeLabel(runtime: RuntimeInstallation) {
  let name = runtimeDisplayName(runtime).trim()
  // Imported environments already carry the version in their generated name.
  // Only remove an exact suffix; keep user-chosen names and variant identifiers.
  if (runtime.version) {
    const suffix = ` · ${runtime.version}`
    while (name.endsWith(suffix)) name = name.slice(0, -suffix.length).trimEnd()
  }
  if (runtime.backend === 'custom') return name || '自定义引擎'
  const version = runtimeVersion(runtime)
  return !name ? version : name.includes(version) ? name : `${name} · ${version}`
}

export function runtimeOptionLabel(runtime: RuntimeInstallation, peers: RuntimeInstallation[]) {
  const optionLabel = (item: RuntimeInstallation) => `${runtimeLabel(item)}${runtimePlatform(item) === 'wsl2' ? ` · WSL2 / ${runtimeDistribution(item)}` : ''}`
  const label = optionLabel(runtime)
  const same = peers.filter(item => optionLabel(item) === label)
  if (same.length < 2) return label
  const location = (item: RuntimeInstallation) => item.path.split(/[\\/]/).filter(Boolean).slice(-2).join('/')
  const suffix = location(runtime)
  const collision = same.some(item => item.id !== runtime.id && location(item) === suffix)
  return `${label} · ${suffix || runtime.id.slice(0, 8)}${collision ? ` · ${runtime.id.slice(0, 8)}` : ''}`
}
