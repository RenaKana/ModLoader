import { ChevronDown, Globe2 } from 'lucide-react'
import { openExternal } from './api'
import type { WebSearchResult } from './web-search'
import { MarkdownContent } from './MessageContent'

export default function WebSearchSources({ result }: { result: WebSearchResult }) {
  const readIds = new Set(result.pages?.map(page => page.source_id))
  const imageIds = new Set(result.images?.map(image => image.source_id))
  const summaryUrls = new Set(result.summaries?.flatMap(summary => summary.url ? [summary.url] : []))
  const searched = result.queries ? result.queries.length > 0 : Boolean(result.query)
  return <details className="web-search-sources">
    <summary><Globe2 size={14} /><span>{searched ? '搜索结果' : '网页来源'} {result.sources.length}{readIds.size > 0 && ` · 已读 ${readIds.size}`}{summaryUrls.size > 0 && ` · 阅读摘要 ${summaryUrls.size}`}{!!result.images?.length && ` · 图片 ${result.images.length}`}</span>{searched && <span className="search-elapsed">搜索累计 {(result.elapsed_ms / 1000).toFixed(1)} s</span>}<ChevronDown size={14} /></summary>
    <div className="web-search-result-list">
      <div className="web-search-query">{(result.queries || [result.query]).filter(Boolean).join(' · ')}</div>
      <ol>{result.sources.map(source => <li key={source.id} value={source.id}>
        <a href={source.url} onClick={event => { event.preventDefault(); void openExternal(source.url) }}>{source.title || source.url}</a>
        <span className="web-search-domain">{source.url.replace(/^https?:\/\//, '').split('/')[0]}<span className={readIds.has(source.id) ? 'source-read' : ''}>{readIds.has(source.id) ? '已读' : summaryUrls.has(source.url) ? '阅读摘要' : imageIds.has(source.id) ? '图片' : source.snippet ? '摘要' : '来源'}</span></span>
        {result.images?.filter(image => image.source_id === source.id).map((image, index) => <div key={image.url}><a href={image.url} onClick={event => { event.preventDefault(); void openExternal(image.url) }}>原图 {index + 1}{image.title && ` · ${image.title}`}</a></div>)}
        {readIds.has(source.id) ? result.pages?.filter(page => page.source_id === source.id).map((page, index) => <div className="web-page-excerpt" key={index}><span>第 {page.start_line}–{page.end_line} 行 / 共 {page.total_lines} 行</span><p>{page.content}</p></div>) : source.snippet && <p>{source.snippet}</p>}
        {result.summaries?.filter(summary => summary.url === source.url).map((summary, index) => <div className="web-page-summary" key={index}><MarkdownContent text={summary.text} /></div>)}
      </li>)}</ol>
    </div>
  </details>
}
