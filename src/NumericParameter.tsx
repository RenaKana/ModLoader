import { Field, NumberInput } from './components'
import type { ParameterField } from './types'
import { parameterDefaultText } from './parameter-defaults'

export function numberError(field: ParameterField, value: unknown): string {
  if (value === undefined || value === null || typeof value === 'string' && !value.trim()) return ''
  if ((typeof value === 'number' || typeof value === 'string') && field.special_values?.includes(value)) return ''
  if (typeof value !== 'number' || !Number.isFinite(value)) return `${field.label}必须是有限数字。`
  if (field.integer && !Number.isSafeInteger(value)) return `${field.label}必须是可精确表示的整数。`
  if (field.minimum != null && (value < field.minimum || field.exclusive_minimum && value === field.minimum)) return `${field.label}必须${field.exclusive_minimum ? '大于' : '不小于'} ${field.minimum}。`
  if (field.maximum != null && value > field.maximum) return `${field.label}不能大于 ${field.maximum}。`
  return ''
}

export default function NumericParameter({ field, value, placeholder, disabled, onChange }: {
  field: ParameterField; value: unknown; placeholder?: string; disabled?: boolean; onChange: (value: number | undefined) => void
}) {
  const error = numberError(field, value)
  const minimum = field.minimum == null ? undefined : Math.min(field.minimum, ...(field.special_values || []).filter((item): item is number => typeof item === 'number'))
  return <Field label={field.label}>
    <NumberInput aria-label={field.label} aria-invalid={!!error} disabled={disabled} min={minimum} max={field.maximum ?? undefined}
      step={field.step ?? (field.integer ? 1 : 'any')} value={typeof value === 'number' && Number.isFinite(value) ? value : ''}
      placeholder={placeholder ?? parameterDefaultText(field)} onValueChange={next => onChange(next === '' ? undefined : Number(next))} />
    {error && <span className="error-text" role="alert">{error}</span>}
  </Field>
}
