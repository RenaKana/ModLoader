import { test, expect } from '@playwright/test'
import { editedDiagnostics, uncheckedTooltip } from '../src/startup-parameters'

test('编辑只清除相交错误，后续标记随文本移动', () => {
  const before = '--ctx-size abc\n--cache-type-k f1'
  const errors = [
    { start: 0, end: 14, flag: '--ctx-size', message: '必须是整数', value_start: 11, value_end: 14 },
    { start: 15, end: 32, flag: '--cache-type-k', message: '未知类型', value_start: 30, value_end: 32 },
  ]
  expect(editedDiagnostics(errors, before, '--ctx-size 8192\n--cache-type-k f1')).toEqual([
    { ...errors[1], start: 16, end: 33, value_start: 31, value_end: 33 },
  ])
  expect(editedDiagnostics(errors, before, before)).toEqual(errors)
  expect(editedDiagnostics(errors, before, '')).toEqual([])
})

test('未校验提示每个参数一行并去重', () => {
  expect(uncheckedTooltip([
    { flag: '--batch-size', message: '取值范围未校验' },
    { flag: '--dws', message: '缺少校验规则' },
    { flag: '--dws', message: '缺少校验规则' },
  ])).toBe('--batch-size：取值范围未校验\n--dws：缺少校验规则')
})
