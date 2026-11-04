import type { Root } from 'mdast'
import type { Extension as FromMarkdownExtension } from 'mdast-util-from-markdown'
import type { Processor } from 'unified'
import type { Construct, Extension, State } from 'micromark-util-types'
import { content as markdownContent } from 'micromark-core-commonmark'
import { math } from 'micromark-extension-math'
import { visit } from 'unist-util-visit'
import 'remark-math'

declare module 'micromark-util-types' {
  interface TokenTypeMap { texMathFlowMarker: 'texMathFlowMarker' }
}

declare module 'mdast-util-math' {
  interface MathData { texFenceClosed?: boolean }
}

const isLineEnding = (code: number | null) => code === -3 || code === -4 || code === -5
const isSpace = (code: number | null) => code === 32 || code === -1 || code === -2

// Tokenize TeX delimiters before Markdown consumes their backslashes. Code spans,
// code fences and URLs retain their normal Markdown parsing rules. Reuse the math
// extension's token structure/resolver so multiline text follows micromark rules.
const standardMath = math().text![36] as Construct
const texMath: Construct = {
  name: 'mathText',
  resolve: standardMath.resolve,
  tokenize(effects, ok, nok) {
    let closing: number
    const start: State = code => {
      effects.enter('mathText')
      effects.enter('mathTextSequence')
      effects.consume(code)
      return opening
    }
    const opening: State = code => {
      if (code !== 40 && code !== 91) return nok(code)
      closing = code === 40 ? 41 : 93
      effects.consume(code)
      effects.exit('mathTextSequence')
      return between
    }
    const closingSequence: Construct = { tokenize(check, yes, no) {
      return code => {
        check.enter('mathTextSequence'); check.consume(code)
        return next => {
          if (next !== closing) return no(next)
          check.consume(next); check.exit('mathTextSequence')
          return yes
        }
      }
    } }
    const between: State = code => {
      if (code === null) return nok(code)
      if (code === 92) return effects.check(closingSequence, close, slash)(code)
      if (code === 32 || isLineEnding(code)) {
        const token = code === 32 ? 'space' : 'lineEnding'
        effects.enter(token); effects.consume(code); effects.exit(token)
        return between
      }
      effects.enter('mathTextData')
      return content(code)
    }
    const content: State = code => {
      if (code === null || code === 92 || code === 32 || isLineEnding(code)) {
        effects.exit('mathTextData')
        return between(code)
      }
      effects.consume(code)
      return content
    }
    const slash: State = code => {
      effects.enter('mathTextData'); effects.consume(code)
      return escaped
    }
    const escaped: State = code => {
      if (code === null) return nok(code)
      if (isLineEnding(code)) { effects.exit('mathTextData'); return between(code) }
      effects.consume(code)
      return content
    }
    const close: State = code => {
      effects.enter('mathTextSequence'); effects.consume(code)
      return next => {
        effects.consume(next); effects.exit('mathTextSequence'); effects.exit('mathText')
        return ok
      }
    }
    return start
  },
}

// Standalone display fences must be recognized before block Markdown (notably
// a standalone '=' becoming a Setext heading) splits their contents into text.
const texMathFlow: Construct = {
  name: 'mathFlow',
  concrete: true,
  tokenize(effects, ok, nok) {
    const self = this
    const nonLazyContinuation: Construct = { partial: true, tokenize(check, yes, no) {
      return code => {
        if (code === null) return no(code)
        check.enter('lineEnding'); check.consume(code); check.exit('lineEnding')
        return next => self.parser.lazy[self.now().line] ? no(next) : yes(next)
      }
    } }
    const closingFence: Construct = { partial: true, tokenize(check, yes, no) {
      let prefix = false
      const before: State = code => {
        if (isSpace(code)) {
          if (!prefix) check.enter('linePrefix')
          prefix = true; check.consume(code)
          return before
        }
        if (prefix) check.exit('linePrefix')
        if (code !== 92) return no(code)
        check.enter('mathFlowFence'); check.enter('texMathFlowMarker'); check.consume(code)
        return bracket
      }
      const bracket: State = code => {
        if (code !== 93) return no(code)
        check.consume(code); check.exit('texMathFlowMarker')
        return after
      }
      const after: State = code => {
        if (isSpace(code)) { check.consume(code); return after }
        if (code !== null && !isLineEnding(code)) return no(code)
        check.exit('mathFlowFence')
        return yes(code)
      }
      return before
    } }
    const embeddedClosing: Construct = { partial: true, tokenize(check, yes, no) {
      return code => {
        check.consume(code)
        return next => {
          if (next !== 93) return no(next)
          check.consume(next)
          return yes
        }
      }
    } }
    const start: State = code => {
      effects.enter('mathFlow'); effects.enter('mathFlowFence'); effects.enter('texMathFlowMarker')
      effects.consume(code)
      return opening
    }
    const opening: State = code => {
      if (code !== 91) return nok(code)
      effects.consume(code); effects.exit('texMathFlowMarker')
      return openingEnd
    }
    const openingEnd: State = code => {
      if (isSpace(code)) { effects.consume(code); return openingEnd }
      if (code !== null && !isLineEnding(code)) return nok(code)
      effects.exit('mathFlowFence')
      if (self.interrupt) return ok(code)
      return continuation(code)
    }
    const continuation: State = code => code === null
      ? after(code)
      : effects.attempt(nonLazyContinuation, lineStart, after)(code)
    const lineStart: State = code => effects.attempt(closingFence, after, contentStart)(code)
    const contentStart: State = code => {
      if (code === null || isLineEnding(code)) return continuation(code)
      effects.enter('mathFlowValue')
      return content(code)
    }
    const content: State = code => {
      if (code === null || isLineEnding(code)) {
        effects.exit('mathFlowValue')
        return continuation(code)
      }
      if (code === 92) return effects.check(embeddedClosing, closeInContent, slash)(code)
      effects.consume(code)
      return content
    }
    const closeInContent: State = code => {
      effects.exit('mathFlowValue'); effects.enter('mathFlowFence'); effects.enter('texMathFlowMarker')
      effects.consume(code)
      return closingBracket
    }
    const closingBracket: State = code => {
      effects.consume(code); effects.exit('texMathFlowMarker')
      return closingEnd
    }
    const closingEnd: State = code => {
      if (isSpace(code)) { effects.consume(code); return closingEnd }
      effects.exit('mathFlowFence'); effects.exit('mathFlow')
      // Existing mixed-line closings may have prose after them. Let CommonMark
      // parse that remainder and its continuation instead of consuming it as TeX.
      return code === null || isLineEnding(code) ? ok(code) : effects.attempt(markdownContent, ok)(code)
    }
    const slash: State = code => {
      effects.consume(code)
      return escaped
    }
    const escaped: State = code => {
      if (code === null || isLineEnding(code)) {
        effects.exit('mathFlowValue')
        return continuation(code)
      }
      effects.consume(code)
      return content
    }
    const after: State = code => {
      effects.exit('mathFlow')
      return ok(code)
    }
    return start
  },
}

export function remarkTexDelimiters(this: Processor<Root>) {
  const data = this.data() as { micromarkExtensions?: Extension[]; fromMarkdownExtensions?: FromMarkdownExtension[] }
  ;(data.micromarkExtensions ||= []).push({ flow: { 92: texMathFlow }, text: { 92: texMath } })
  ;(data.fromMarkdownExtensions ||= []).push({ exit: {
    texMathFlowMarker(token) {
      const node = this.stack.find(node => node.type === 'math')
      if (node) node.data!.texFenceClosed = this.sliceSerialize(token) === '\\]'
    },
  } })
}

function closedFence(raw: string) {
  const lines = raw.split(/\r?\n/)
  const opening = /^\s*(`{3,}|~{3,}|\${2,})/.exec(lines[0])?.[1]
  if (!opening) return true // Indented code is not a streaming fence.
  if (lines.length < 2) return false
  const last = lines[lines.length - 1].trim()
  return last.length >= opening.length && [...last].every(character => character === opening[0])
}

// TeX arrays use a text-sized strut even when a cell requests display fractions.
// A 2x strut gives 2.4em rows versus ~2.01em for numeric dfrac cells, leaving
// clearance without changing text line-height. Keep authored spacing untouched.
function spaceFractionArrays(value: string) {
  if (value.includes('\\arraystretch')) return value
  const environments: { name: string; start: number }[] = []
  const insertions: { at: number; text: string }[] = []
  for (const match of value.matchAll(/\\(begin|end)\{([^}]+)\}/g)) {
    const [, command, name] = match
    if (command === 'begin') { environments.push({ name, start: match.index }); continue }
    const opening = environments.pop()
    if (!opening || opening.name !== name || name !== 'array') continue
    const body = value.slice(opening.start, match.index)
    if (!/\\dfrac(?![a-zA-Z])/.test(body) || /\\\\\s*\[/.test(body)) continue
    insertions.push({ at: opening.start, text: '{\\def\\arraystretch{2}' }, { at: match.index + match[0].length, text: '}' })
  }
  for (const insertion of insertions.sort((a, b) => b.at - a.at)) {
    value = value.slice(0, insertion.at) + insertion.text + value.slice(insertion.at)
  }
  return value
}

export function remarkMessageSyntax({ streaming = false }: { streaming?: boolean } = {}) {
  return (tree: Root, file: { value: unknown }) => {
    const source = String(file.value)
    visit(tree, (node, index, parent) => {
      const raw = source.slice(node.position?.start.offset, node.position?.end.offset)
      if (node.type === 'inlineMath' && (raw.startsWith('\\[') || raw.startsWith('$$'))) {
        node.data!.hProperties = { className: ['language-math', 'math-display'] }
      }
      if (node.type === 'inlineMath' && /^\$[^$]/.test(raw)) {
        // Dollar math follows the usual no-edge-whitespace convention. A closing
        // dollar followed by a digit belongs to the next price, not this formula.
        if (/^\$\s|\s\$$/.test(raw) || /\d/.test(source.charAt(node.position?.end.offset ?? source.length))) {
          if (parent && index !== undefined) parent.children[index] = { type: 'text', value: raw }
        }
      }
      if (node.type === 'math' && (node.data?.texFenceClosed === false || (streaming && !closedFence(raw)))) {
        if (parent && index !== undefined) parent.children[index] = { type: 'paragraph', children: [{ type: 'text', value: raw }] }
        return
      }
      if (node.type === 'html' && parent && parent.type !== 'paragraph' && index !== undefined) {
        // Raw block HTML is previewed in the same isolated frame as fenced HTML;
        // it never becomes live HTML in the chat document.
        parent.children[index] = {
          type: 'code', lang: /^\s*<svg[\s>]/i.test(node.value) ? 'svg' : 'html', value: node.value,
          data: { hProperties: { 'data-pending': streaming } },
        }
      }
      if (node.type === 'code') {
        const pending = streaming && !closedFence(raw)
        node.data = { ...node.data, hProperties: { ...node.data?.hProperties, 'data-pending': pending } }
        if (['math', 'latex', 'tex'].includes(node.lang || '')) {
          node.data.hProperties!.className = pending ? ['language-text'] : ['language-math', 'math-display']
        }
      }
      if (node.type === 'inlineMath' || node.type === 'math' || (node.type === 'code' && ['math', 'latex', 'tex'].includes(node.lang || ''))) {
        if (node.type !== 'inlineMath' && streaming && !closedFence(raw)) return
        const spaced = spaceFractionArrays(node.value)
        if (spaced !== node.value) {
          node.value = spaced
          // remark-math already populated hChildren before this transform.
          for (const child of node.data?.hChildren || []) {
            if (child.type === 'text') child.value = spaced
            if (child.type === 'element' && child.tagName === 'code') child.children = [{ type: 'text', value: spaced }]
          }
        }
      }
    })
  }
}
