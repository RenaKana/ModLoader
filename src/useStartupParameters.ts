import { useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import { editedDiagnostics, startupDraftKey, type ParameterDiagnostic, type StartupDraft, type StartupParametersResult } from './startup-parameters'

export function useStartupParameters(draft: StartupDraft, active: boolean, onApply: (value: StartupParametersResult) => void) {
  const [opened, setOpened] = useState(active)
  const [text, setText] = useState('')
  const [result, setResult] = useState<StartupParametersResult>()
  const [errors, setErrors] = useState<ParameterDiagnostic[]>([])
  const [pending, setPending] = useState(false)
  const [failure, setFailure] = useState('')
  const latest = useRef({ draft, onApply })
  latest.current = { draft, onApply }
  const input = useRef(''), canonical = useRef(''), dirty = useRef(false), edited = useRef(false)
  const previousText = useRef(''), mergePaste = useRef(false)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const sequence = useRef(0), abort = useRef<AbortController | undefined>(undefined)
  const renderedKey = useRef('')
  const parsing = useRef<Promise<StartupDraft | null> | null>(null)
  const key = startupDraftKey(draft)

  function cancel() {
    clearTimeout(timer.current)
    abort.current?.abort()
    sequence.current++
    parsing.current = null
  }

  function accept(value: StartupParametersResult, normalize: boolean) {
    setResult(value); setErrors(value.errors); setFailure('')
    if (!value.errors.length) {
      canonical.current = value.text
      if (normalize) { input.current = value.text; setText(value.text) }
    }
  }

  async function render() {
    cancel()
    const revision = sequence.current, base = latest.current.draft
    if (!base.runtime_id) { setPending(false); return }
    const controller = new AbortController()
    abort.current = controller
    setPending(true); setFailure(''); setErrors([])
    try {
      const value = await request<StartupParametersResult>('POST', '/api/profiles/launch-parameters', { ...base, preview: true }, { signal: controller.signal })
      if (revision !== sequence.current) return
      input.current = value.text; canonical.current = value.text
      setText(value.text); accept(value, false)
    } catch (error) {
      if (revision === sequence.current) setFailure(errorMessage(error))
    } finally { if (revision === sequence.current) setPending(false) }
  }

  function parse(normalize = false): Promise<StartupDraft | null> {
    clearTimeout(timer.current)
    if (parsing.current) return parsing.current
    if (!dirty.current) {
      if (errors.length || failure && edited.current) return Promise.resolve(null)
      if (normalize && canonical.current) { input.current = canonical.current; setText(canonical.current) }
      return Promise.resolve(latest.current.draft)
    }
    cancel()
    const revision = sequence.current, base = latest.current.draft, source = input.current
    const controller = new AbortController()
    abort.current = controller
    setPending(true); setFailure('')
    const promise = (async () => {
      try {
        const value = await request<StartupParametersResult>('POST', '/api/profiles/launch-parameters', {
          ...base, text: source, ...(!mergePaste.current ? { previous_text: previousText.current } : {}), preview: normalize,
        }, { signal: controller.signal })
        if (revision !== sequence.current) return null
        accept(value, normalize)
        dirty.current = false
        if (value.errors.length) return null
        const next = { ...base, parameters: value.parameters, generation: value.generation, extra_args: value.extra_args, parameter_editor_version: 1 as const }
        renderedKey.current = startupDraftKey(next)
        latest.current.draft = next
        latest.current.onApply(value)
        return next
      } catch (error) {
        if (revision === sequence.current) setFailure(errorMessage(error))
        return null
      } finally {
        if (revision === sequence.current) { setPending(false); parsing.current = null }
      }
    })()
    parsing.current = promise
    return promise
  }

  function edit(next: string, immediate = false, composing = false) {
    cancel()
    if (!dirty.current) { previousText.current = input.current; mergePaste.current = false }
    if (immediate) mergePaste.current = true
    const before = input.current
    setErrors(previous => editedDiagnostics(previous, before, next))
    input.current = next; dirty.current = true; edited.current = true
    setText(next); setFailure(''); setPending(true)
    if (!composing) timer.current = setTimeout(() => void parse(immediate), immediate ? 0 : 650)
  }

  useEffect(() => { if (active) setOpened(true) }, [active])
  useEffect(() => {
    if (!opened || renderedKey.current === key) return
    renderedKey.current = key
    dirty.current = false
    void render()
  }, [key, opened])
  useEffect(() => () => { cancel(); renderedKey.current = '' }, [])

  return { text, result, errors, pending, failure, edit,
    flush: async () => {
      const value = await parse(true)
      if (value && canonical.current) { input.current = canonical.current; setText(canonical.current) }
      return value
    },
    currentText: () => input.current,
    prepareSave: () => opened ? parse(false) : Promise.resolve(latest.current.draft),
    retry: () => dirty.current ? void parse() : void render(),
  }
}
