"""SauceNAO's image-upload API; no URL downloads or local-file access.

Wire shape checked against PicImageSearch 3.12.11's SauceNAO adapter/model.
The upstream API documentation is https://saucenao.com/user.php?page=search-api.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import math
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

from .web_search import SEARCH_TIMEOUT_S, _plain_text, validate_http_url


def _image_bytes(image, filename):
    if not isinstance(filename, str) or not filename.strip() or any(ord(char) < 32 for char in filename):
        raise ValueError('请提供有效的图片文件名')
    if not isinstance(image, str):
        raise ValueError('反向搜图需要 data:image/...;base64 格式的图片内容')
    match = re.fullmatch(r'data:(image/[a-zA-Z0-9.+-]+);base64,([a-zA-Z0-9+/=]+)', image)
    if match is None:
        raise ValueError('反向搜图只接受请求中的 Base64 图片，不接受图片网址或本机路径')
    try:
        content = base64.b64decode(match[2], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('图片 Base64 数据无效') from error
    if not content:
        raise ValueError('图片内容不能为空')
    return match[1].lower(), content


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _url(value):
    if not isinstance(value, str):
        return ''
    value = value.strip()
    try:
        validate_http_url(value)
        if '@' in urlsplit(value).netloc:
            return ''
    except ValueError:
        return ''
    return value


def _first_text(data, fields):
    for field in fields:
        value = data.get(field)
        if isinstance(value, list):
            value = ', '.join(part for part in value if isinstance(part, str))
        text = _plain_text(value)
        if text:
            return text
    return ''


def _matches(payload):
    rows = payload.get('results')
    if not isinstance(rows, list):
        raise HTTPException(502, 'SauceNAO 响应缺少有效的 results 数组')
    matches = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('header'), dict) or not isinstance(row.get('data'), dict):
            continue
        header, data = row['header'], row['data']
        similarity = _number(header.get('similarity'))
        if similarity is None or not 0 <= similarity <= 100:
            continue
        urls = []
        for value in [*(data.get('ext_urls') if isinstance(data.get('ext_urls'), list) else []), data.get('source')]:
            url = _url(value)
            if url and url not in urls:
                urls.append(url)
        index_name = _plain_text(header.get('index_name'))
        title = _first_text(data, ('title', 'material', 'jp_name', 'eng_name', 'source', 'created_at')) or index_name
        thumbnail = _url(header.get('thumbnail'))
        if not (title or thumbnail or urls):
            continue
        value = {'similarity': similarity, 'thumbnail': thumbnail, 'title': title, 'urls': urls}
        author = _first_text(data, ('author', 'member_name', 'creator', 'twitter_user_handle',
                                    'pawoo_user_display_name', 'author_name', 'user_name', 'artist', 'company'))
        if author:
            value['author'] = author
        if index_name:
            value['index_name'] = index_name
        matches.append(value)
    if rows and not matches:
        raise HTTPException(502, 'SauceNAO 未返回有效的匹配项')
    return matches


def _result(payload, started):
    if not isinstance(payload, dict) or not isinstance(payload.get('header'), dict):
        raise HTTPException(502, 'SauceNAO 响应缺少有效的 header')
    header = payload['header']
    status = _number(header.get('status'))
    if status is None or not status.is_integer():
        raise HTTPException(502, 'SauceNAO 响应缺少有效的状态码')
    quota = {}
    for field in ('short_remaining', 'long_remaining'):
        remaining = _number(header.get(field))
        if remaining is not None and remaining >= 0:
            quota[field] = remaining
    if status != 0:
        # Remote error text is used only for classification, never echoed with
        # API keys, request contents or upstream diagnostics.
        message = str(header.get('message', '')).lower()
        if any(word in message for word in ('api key', 'api_key', 'auth', 'permission', 'login', 'log in')):
            raise HTTPException(502, 'SauceNAO 认证或访问被拒绝，请检查 API Key')
        if quota.get('long_remaining') == 0 or ('limit' in message and any(word in message for word in ('daily', 'day', 'long'))):
            raise HTTPException(429, 'SauceNAO 当日搜索额度已用尽')
        if quota.get('short_remaining') == 0 or any(word in message for word in ('limit', 'too many', 'rate')):
            raise HTTPException(429, 'SauceNAO 搜索频率受限，请稍后重试')
        if 'access' in message:
            raise HTTPException(502, 'SauceNAO 认证或访问被拒绝，请检查 API Key')
        raise HTTPException(502, f'SauceNAO 查询失败（状态 {int(status)}），请检查图片或服务状态')
    return {'provider': 'saucenao', 'elapsed_ms': round((time.monotonic() - started) * 1000),
            'matches': _matches(payload), **({'quota': quota} if quota else {})}


async def search_saucenao(image, filename, endpoint, key, proxy='', *, progress=lambda event: None):
    mime, content = _image_bytes(image, filename)
    fields = {'output_type': '2'}
    if key:
        fields['api_key'] = key
    started = time.monotonic()
    progress({'stage': 'web_reverse', 'message': '正在反向搜索图片'})
    try:
        async with asyncio.timeout(SEARCH_TIMEOUT_S):
            async with httpx.AsyncClient(proxy=proxy or None, trust_env=False, verify=True,
                                         follow_redirects=False, timeout=SEARCH_TIMEOUT_S) as client:
                response = await client.post(endpoint, data=fields, files={'file': (filename, content, mime)},
                                             headers={'Accept': 'application/json'})
                is_html = 'text/html' in response.headers.get('content-type', '').lower() or response.content.lstrip().lower().startswith((b'<!doctype html', b'<html'))
                if is_html and (response.is_success or response.status_code == 403):
                    raise HTTPException(502, 'SauceNAO 返回了登录或访问验证页面，无法读取反向搜图结果')
                if not response.is_success:
                    if response.status_code == 429:
                        raise HTTPException(429, 'SauceNAO 搜索额度或频率受限，请稍后重试')
                    reason = '，请检查 API Key 与访问权限' if response.status_code in (401, 403) else ''
                    raise HTTPException(502, f'SauceNAO 请求失败（HTTP {response.status_code}）{reason}')
                try:
                    payload = response.json()
                except ValueError as error:
                    raise HTTPException(502, 'SauceNAO 未返回有效 JSON') from error
    except (httpx.TimeoutException, TimeoutError) as error:
        raise HTTPException(504, 'SauceNAO 请求超时，请检查服务或代理') from error
    except httpx.HTTPError as error:
        raise HTTPException(502, 'SauceNAO 连接失败，请检查服务地址、TLS 证书或代理') from error
    return _result(payload, started)
