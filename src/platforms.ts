import type { Backend, ModelAsset, ModelLocation, RuntimeInstallation, RuntimePlatform } from './types'

export const platformLabels: Record<RuntimePlatform, string> = { windows: 'Windows 本机', wsl2: 'WSL2 / Linux' }

export function runtimePlatform(runtime: RuntimeInstallation, backends: Backend[] = []): RuntimePlatform {
  return runtime.platform || (backends.find(backend => backend.id === runtime.backend)?.import_kind === 'wsl' || runtime.backend === 'vllm' ? 'wsl2' : 'windows')
}

export function backendPlatforms(backend: Backend): RuntimePlatform[] {
  return backend.supported_platforms || (backend.id === 'custom' ? ['windows', 'wsl2'] : backend.import_kind === 'wsl' || backend.id === 'vllm' ? ['wsl2'] : ['windows'])
}

export function runtimeDistribution(runtime: RuntimeInstallation) {
  return typeof runtime.metadata.distribution === 'string' ? runtime.metadata.distribution : 'ModLoader'
}

export function runtimeLocation(runtime: RuntimeInstallation, backends: Backend[] = []): ModelLocation {
  return runtimePlatform(runtime, backends) === 'wsl2' ? { platform: 'wsl2', distribution: runtimeDistribution(runtime) } : { platform: 'windows' }
}

export function locationLabel(location?: ModelLocation) {
  return location?.platform === 'wsl2' ? `WSL2 · ${location.distribution || '发行版未指定'}` : platformLabels.windows
}

export function desktopPath(path: string, location?: ModelLocation) {
  return location?.platform === 'wsl2' && location.distribution && path.startsWith('/')
    ? `\\\\wsl.localhost\\${location.distribution}${path.replaceAll('/', '\\')}` : path
}

export function modelSupportsRuntime(model: ModelAsset, runtime: RuntimeInstallation, backends: Backend[]) {
  return model.location?.platform !== 'wsl2' || runtimePlatform(runtime, backends) === 'wsl2' && model.location.distribution === runtimeDistribution(runtime)
}

export function modelSizeKnown(model: ModelAsset) {
  return model.metadata.size_bytes_known !== false && model.size_bytes != null
}

export function sameRuntimePath(first: RuntimeInstallation, second: RuntimeInstallation, backends: Backend[]) {
  if (runtimePlatform(first, backends) !== runtimePlatform(second, backends)) return false
  if (runtimePlatform(first, backends) === 'wsl2') return runtimeDistribution(first) === runtimeDistribution(second) && first.path === second.path
  const normalized = (path: string) => path.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase()
  return normalized(first.path) === normalized(second.path)
}
