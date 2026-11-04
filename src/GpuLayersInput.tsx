import { Field, NumberInput } from './components'
import { numberError } from './NumericParameter'
import type { ParameterField } from './types'

// Preserve the stored representation until the user edits. Blank is still unset;
// an all/legacy sentinel is only projected to the model's numeric endpoint.
export function gpuDisplayValue(value: unknown, maximum: number | null | undefined, legacy = false) {
  if (legacy && (value === -1 || value === 'all' || typeof value === 'number' && maximum != null && value > maximum)) return maximum ?? ''
  return typeof value === 'number' && Number.isFinite(value) ? value : ''
}

export function gpuInputError(field: ParameterField, value: unknown) {
  if (value === undefined) return ''
  return numberError(field, value)
}

export default function GpuLayersInput({ field, value, edited, onChange }: {
  field: ParameterField; value: unknown; edited: boolean; onChange: (value: number | undefined) => void
}) {
  const maximum = field.maximum
  const minimum = field.minimum == null ? undefined : Math.min(field.minimum, ...(field.special_values || []).filter((item): item is number => typeof item === 'number'))
  const error = edited ? gpuInputError(field, value) : ''
  return <Field label={<><span>{field.label}</span>{maximum != null && <span className="field-range">0–{maximum}</span>}</>}>
    <NumberInput aria-label={field.label} aria-invalid={!!error} min={minimum} max={maximum ?? undefined} step={1}
      value={gpuDisplayValue(value, maximum, !edited)}
      placeholder={value === 'auto' ? '自动（已保存）' : maximum == null ? '默认 全部' : `默认 ${maximum}`}
      onValueChange={next => onChange(next === '' ? undefined : Number(next))} />
    {error && <span className="error-text" role="alert">{error}</span>}
  </Field>
}
