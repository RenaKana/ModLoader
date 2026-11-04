import { ExternalLink } from 'lucide-react'
import { openExternal } from './api'
import { Button, DiagnosticNotice, Field, Select, formatBytes } from './components'
import type { ModelAsset, ModelLocation, MtpStatus, StrataDraftingStatus } from './types'
import { PlatformPathInput } from './WslPlatform'
import './profile-acceleration.css'
import NumericParameter from './NumericParameter'
import OperationStatus, { type OperationProgress } from './OperationStatus'

export default function ProfileAcceleration({ model, backend, location, parameters, status, drafting, checking, progress, startedAt, error, retry, onChange }: {
  model: ModelAsset; backend: string; location?: ModelLocation; parameters: Record<string, unknown>; status?: MtpStatus
  drafting?: StrataDraftingStatus
  checking: boolean; error?: string; retry: () => void; onChange: (values: Record<string, unknown>) => void
  progress?: OperationProgress; startedAt: number
}) {
  if (backend === 'strata' && drafting?.supported) {
    const mode = String(parameters.strata_drafting ?? drafting.mode)
    const effective = mode === 'auto' || mode === 'legacy' ? drafting.effective_mode : mode
    const mtpPath = typeof parameters.mtp_path === 'string' ? parameters.mtp_path : status?.path || ''
    const draftPath = typeof parameters.dflash2_path === 'string' ? parameters.dflash2_path : effective === 'dflash2' ? drafting.path || '' : ''
    const activeLabel = effective === 'off' ? '关闭' : effective === 'dflash2' ? 'DFlash2' : 'MTP'
    return <section className="profile-acceleration capability-section" aria-label="草稿设置">
      <div className="engine-acceleration-mode">
        <Field label="草稿模式"><Select value={mode} onChange={event => onChange({ strata_drafting: event.target.value })}>
          <option value="legacy">继承（{status?.enabled ? 'MTP' : '关闭'}）</option>
          <option value="auto">Auto</option><option value="mtp">MTP</option><option value="dflash2">DFlash2</option><option value="off">关闭</option>
        </Select></Field>
        <span role="status">{checking ? <OperationStatus progress={progress} startedAt={startedAt} fallback="检查草稿资源" /> : mode === 'auto' ? activeLabel : ''}</span>
      </div>
      {error ? <DiagnosticNotice summary="草稿检查失败" detail={error} retry={retry} /> : drafting.status === 'error'
        ? <p className="error-text">{drafting.message}</p> : drafting.status === 'needs_preparation' && effective !== 'off'
          ? <p className="field-help">{drafting.download_bytes ? `首次加载下载 ${formatBytes(drafting.download_bytes)} 并准备草稿` : '首次加载准备草稿'}</p> : null}
      {effective !== 'off' && <div className="capability-fields">
        {effective === 'mtp' && status?.token_field && <NumericParameter field={status.token_field} value={parameters.mtp_tokens}
          placeholder={status.tokens != null ? `默认 ${status.tokens}` : '默认'} onChange={value => onChange({ mtp_tokens: value })} />}
        {effective === 'mtp' ? <Field label="MTP 运行目录"><PlatformPathInput label="MTP 运行目录" location={location} kind="directory"
          value={mtpPath} onChange={value => onChange({ mtp_path: value })} placeholder="自动准备" /></Field>
          : <Field label="DFlash2 配套权重"><PlatformPathInput label="DFlash2 配套权重" location={location} kind="file"
            value={draftPath} onChange={value => onChange({ dflash2_path: value })} placeholder="自动下载并准备 .pack" /></Field>}
      </div>}
    </section>
  }
  const enabled = parameters.mtp_enabled === undefined ? status?.enabled ?? false : parameters.mtp_enabled === true
  const path = typeof parameters.mtp_path === 'string' ? parameters.mtp_path : status?.path || ''
  const pathLabel = backend === 'strata' ? 'MTP 运行目录' : 'MTP 配套权重'
  const labels = { builtin: '内置可用', external: '已有配套文件', missing: '缺少配套文件', unsupported: '当前环境不支持', unverified: '待确认' }
  const source = typeof model.source.repo_id === 'string' ? model.source.repo_id : ''
  const sourceUrl = source ? `${model.source.kind === 'modelscope' || model.source.source === 'modelscope' ? 'https://modelscope.cn/models/' : 'https://huggingface.co/'}${source.split('/').map(encodeURIComponent).join('/')}` : null
  const downloadUrl = sourceUrl || (backend === 'strata' ? 'https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md' : 'https://huggingface.co/models?search=' + encodeURIComponent(model.name))
  const needsAttention = status && ['missing', 'unsupported', 'unverified'].includes(status.status)
  return <section className="profile-acceleration capability-section" aria-label="MTP 设置">
    <div className="acceleration-heading"><label><input type="checkbox" checked={enabled} disabled={!enabled && (!status || status.backend_supported === false)} onChange={event => {
      const next: Record<string, unknown> = { mtp_enabled: event.target.checked }
      if (event.target.checked && !path && status?.candidates.length === 1) next.mtp_path = status.candidates[0].path
      onChange(next)
    }} />MTP 推测加速</label><span>{checking ? <OperationStatus progress={progress} startedAt={startedAt} fallback="检查模型能力" /> : error || status?.diagnostic ? '检查失败' : needsAttention ? labels[status.status] : ''}</span></div>
    {error ? <DiagnosticNotice summary="MTP 检查失败" detail={error} retry={retry} /> : needsAttention && (status.diagnostic
      ? <DiagnosticNotice summary={status.message} detail={status.diagnostic} />
      : <p className={status.status === 'missing' && enabled ? 'error-text' : 'field-help'}>{status.message}</p>)}
    {enabled && status?.backend_supported !== false && <div className="capability-fields">
      {status?.token_field && <NumericParameter field={status.token_field} value={parameters.mtp_tokens}
        placeholder={status.tokens != null ? `默认 ${status.tokens}` : '默认'} onChange={value => onChange({ mtp_tokens: value })} />}
      {!!status?.candidates.length && <Field label="复用此模型的 MTP 文件"><Select value={status.candidates.some(candidate => candidate.path === path) ? path : ''} onChange={event => { if (event.target.value) onChange({ mtp_path: event.target.value }) }}><option value="">选择已有配置的文件</option>{status.candidates.map(candidate => <option key={candidate.path} value={candidate.path}>{candidate.label}</option>)}</Select></Field>}
      <Field label={pathLabel}><PlatformPathInput label={pathLabel} location={location} kind={status?.path_kind || 'file'} value={path} onChange={value => onChange({ mtp_path: value })} placeholder={backend === 'strata' ? '本地 rt 目录' : status?.needs_path ? '选择配套权重' : '内置'} /></Field>
    </div>}
    {status?.status === 'missing' && <div className="acceleration-source"><Button small variant="ghost" icon={ExternalLink} onClick={() => void openExternal(downloadUrl)}>查找配套文件</Button></div>}
  </section>
}
