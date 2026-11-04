import type { LoadProfile, PreparationStatus } from './types'

export type StartupDraft = Omit<LoadProfile, 'id' | 'revision'> & { id?: string }
export interface ParameterDiagnostic {
  start: number
  end: number
  flag: string
  message: string
  value_start?: number
  value_end?: number
}
export interface UncheckedParameter { flag: string; message: string }
export interface StartupParametersResult {
  parameters: Record<string, unknown>
  generation: Record<string, unknown>
  extra_args: string[]
  parameter_editor_version: 1
  text: string
  errors: ParameterDiagnostic[]
  unchecked: UncheckedParameter[]
  argv?: string[]
  cwd?: string
  config?: unknown
  compatibility?: { status: string; reasons: string[] }
  preparation?: PreparationStatus
  preview_error?: string
}

export function editedDiagnostics(errors: ParameterDiagnostic[], before: string, after: string) {
  if (before === after) return errors
  let start = 0, oldEnd = before.length, newEnd = after.length
  while (start < oldEnd && start < newEnd && before[start] === after[start]) start++
  while (oldEnd > start && newEnd > start && before[oldEnd - 1] === after[newEnd - 1]) { oldEnd--; newEnd-- }
  const delta = newEnd - oldEnd
  return errors.filter(error => !(start <= error.end && oldEnd >= error.start)).map(error => error.start >= oldEnd ? {
    ...error, start: error.start + delta, end: error.end + delta,
    ...(error.value_start == null ? {} : { value_start: error.value_start + delta }),
    ...(error.value_end == null ? {} : { value_end: error.value_end + delta }),
  } : error)
}

export function uncheckedTooltip(items: UncheckedParameter[]) {
  const groups = new Map<string, Set<string>>()
  for (const item of items) {
    if (!groups.has(item.flag)) groups.set(item.flag, new Set())
    groups.get(item.flag)!.add(item.message)
  }
  return [...groups].map(([flag, messages]) => `${flag}：${[...messages].join('；')}`).join('\n')
}

export function startupDraftKey(draft: StartupDraft) {
  return JSON.stringify([draft.model_id, draft.runtime_id, draft.backend, draft.alias, draft.parameters, draft.generation, draft.extra_args])
}
