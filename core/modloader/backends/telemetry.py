"""Read-only backend telemetry. Never reads conversations, request logs or generates tokens.

Source contracts checked against Strata 0.1.43/0.1.44.1, llama.cpp b11609,
vLLM 0.31.0, Transformers 5.19.0, SGLang 0.5.21, NInfer 64d10fbf9f8e
and SparkInfer 0.6.34. Optional fields remain absent/null when
an older runtime does not expose them; counters are never reported as rates.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from ..models import LaunchSpec, RuntimeInstance


class Metrics:
    def __init__(self):
        self.groups: dict[str, dict] = {}
        self.unavailable: list[str] = []
        self.status: str | None = None

    def add(self, group, title, key, label, value, unit="", kind="gauge", description=""):
        if isinstance(value, bool):
            value = "是" if value else "否"
            kind = "text"
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            value = None
        elif value is not None and not isinstance(value, (str, int, float)):
            return
        metric = {"id": key, "label": label, "value": value, "kind": kind}
        if unit:
            metric["unit"] = unit
        if description:
            metric["description"] = description
        self.groups.setdefault(group, {"id": group, "title": title, "metrics": []})["metrics"].append(metric)

    def fields(self, group, title, source, definitions, endpoint, qualifier=""):
        if not isinstance(source, dict):
            return
        for key, label, unit, kind in definitions:
            if key in source:
                self.add(group, title, key, label, source[key], unit, kind,
                         f"来源 {endpoint} · {key}" + (f"；{qualifier}" if qualifier else ""))

    def result(self):
        result = {"groups": list(self.groups.values()), "unavailable": list(dict.fromkeys(self.unavailable))}
        if self.status is not None:
            result["backend_status"] = self.status
        return result


async def _fetch(client: httpx.AsyncClient, path: str, *, text=False):
    try:
        response = await client.get(path, headers={"Accept": "text/plain; version=0.0.4" if text else "application/json"})
        if not response.is_success:
            return None, f"{path} 不可用（HTTP {response.status_code}）"
        if text:
            return response.text, None
        return response.json() if response.content else {}, None
    except httpx.TimeoutException:
        return None, f"{path} 读取超时"
    except httpx.HTTPError:
        return None, f"{path} 连接不可用"
    except ValueError:
        return None, f"{path} 未返回可解析 JSON"


STRATA_ENGINE = [
    ("expert_slots", "GPU 专家驻留槽位", "个", "gauge"),
    ("expert_cache_mib", "GPU 专家权重缓存（所有 GPU）", "MiB", "gauge"),
    ("expert_slots_primary", "主 GPU 专家槽位", "个", "gauge"),
    ("expert_cache_primary_mib", "主 GPU 专家权重缓存", "MiB", "gauge"),
    ("expert_slots_full", "专家缓存满配槽位", "个", "gauge"),
    ("expert_cache_full_mib", "专家缓存满配容量", "MiB", "gauge"),
    ("arena_mib", "主机 RAM 专家权重区", "MiB", "gauge"),
    ("vram_elastic", "弹性专家显存", "", "gauge"),
]
STRATA_CONTEXT = [
    ("max_context", "最大上下文", "token", "gauge"),
    ("kv", "KV 缓存类型", "", "text"),
    ("kv_resident", "GPU KV 驻留容量", "token", "gauge"),
    ("batch_slots", "引擎批处理槽位", "个", "gauge"),
    ("batch_groups", "流水线批处理组", "组", "gauge"),
    ("slot_cache", "槽位缓存配置", "", "gauge"),
    ("pool_workers", "CPU 专家工作线程", "个", "gauge"),
    ("spec", "推测解码长度", "token", "gauge"),
    ("mtp_max", "MTP 最大草稿长度", "token", "gauge"),
    ("spec_min_p", "推测解码最低概率", "", "gauge"),
    ("pcie_frac", "PCIe 路由配置比例", "比例", "gauge"),
]
STRATA_LIVE = [
    ("state", "当前推理状态", "", "text"),
    ("phase", "当前阶段", "", "text"),
    ("queued", "排队请求", "个", "gauge"),
    ("running", "运行中请求", "个", "gauge"),
    ("waiting", "等待控制通道", "个", "gauge"),
    ("admitting", "等待接纳", "个", "gauge"),
    ("outside_slots", "槽位外运行请求", "个", "gauge"),
    ("parallel", "有效并发槽位", "个", "gauge"),
    ("prompt_tokens", "当前输入长度", "token", "gauge"),
    ("prompt_read", "当前预填充位置（含复用）", "token", "gauge"),
    ("prompt_total", "当前预填充目标", "token", "gauge"),
    ("generated", "当前已生成", "token", "gauge"),
    ("max_tokens", "本次最大生成长度", "token", "gauge"),
    ("elapsed_s", "当前请求耗时", "s", "gauge"),
    ("tok_s", "后端速度报告（可能回退均值）", "token/s", "gauge"),
    ("tok_s_window_s", "后端标称速度窗口", "s", "gauge"),
    ("tok_s_mean", "当前请求平均解码速度", "token/s", "gauge"),
    ("prefill_tok_s_mean", "当前请求平均预填充速度", "token/s", "gauge"),
]
STRATA_TOTALS = [
    ("requests", "累计完成请求", "个", "counter"),
    ("prompt_tokens", "累计输入 token（含复用）", "token", "counter"),
    ("reused", "累计复用输入 token", "token", "counter"),
    ("output_tokens", "累计输出 token", "token", "counter"),
    ("prompt_ms", "累计预填充时间", "ms", "counter"),
    ("decode_ms", "累计解码时间", "ms", "counter"),
    ("drafts_offered", "累计提供草稿", "token", "counter"),
    ("drafts_accepted", "累计接受草稿", "token", "counter"),
]
STRATA_CACHE = [
    ("enabled", "会话缓存已启用", "", "text"),
    ("budget_mib", "会话缓存预算", "MiB", "gauge"),
    ("slots", "会话缓存槽位", "个", "gauge"),
    ("requests", "保留统计的请求数", "个", "gauge"),
    ("requests_reused", "保留统计中发生复用的请求", "个", "gauge"),
    ("reused_tokens", "累计复用 token", "token", "counter"),
    ("prompt_tokens", "累计输入 token", "token", "counter"),
    ("last_reused", "最近请求复用 token", "token", "gauge"),
    ("last_prompt", "最近请求输入 token", "token", "gauge"),
]
STRATA_LAST = [
    ("duration_s", "最近请求总耗时", "s", "gauge"),
    ("prompt_ms", "最近预填充时间", "ms", "gauge"),
    ("decode_ms", "最近解码时间", "ms", "gauge"),
    ("decode_tok_s", "最近请求平均解码速度", "token/s", "gauge"),
    ("prompt_read", "最近实际预填充 token", "token", "gauge"),
    ("prompt_tokens", "最近输入 token", "token", "gauge"),
    ("output_tokens", "最近输出 token", "token", "gauge"),
    ("reused", "最近复用 token", "token", "gauge"),
    ("hit_rate", "最近请求专家显存缓存命中率", "比例", "gauge"),
    ("pcie_share", "最近请求 PCIe/远端专家分担比例", "比例", "gauge"),
    ("ram_blobs", "最近请求 RAM 专家 blob 访问", "次", "gauge"),
    ("file_blobs", "最近请求文件专家 blob 访问", "次", "gauge"),
    ("file_mb", "最近请求读取专家文件", "MB", "gauge"),
    ("drafts_offered", "最近提供草稿", "token", "gauge"),
    ("drafts_accepted", "最近接受草稿", "token", "gauge"),
]


def _strata(data: dict, health: dict, result: Metrics):
    result.fields("strata_health", "Strata 加载状态", health,
                  [("loaded", "引擎已加载", "", "text"), ("images", "视觉编码器已配置", "", "text")], "/health")
    engine = data.get("engine") if isinstance(data.get("engine"), dict) else {}
    # Elastic updates put newer main-cache measurements under info.vram; do not
    # label that main-GPU value as a new all-GPU total.
    result.fields("strata_experts", "专家权重与分层缓存", engine, STRATA_ENGINE, "/metrics engine",
                  "引擎启动/最近重载报告；RAM 专家区不是进程 RSS 或系统 RAM 总量")
    result.fields("strata_elastic", "最近弹性显存调整", engine.get("vram"),
                  [("expert_cache_mib", "最近主 GPU 专家缓存容量", "MiB", "gauge"),
                   ("expert_cache_full_mib", "主 GPU 专家缓存满配容量", "MiB", "gauge"),
                   ("expert_slots", "最近主 GPU 专家槽位", "个", "gauge"),
                   ("reserve_mib", "引擎显存预留", "MiB", "gauge"),
                   ("prompt_chunk", "预填充块大小", "token", "gauge")], "/metrics engine.vram", "最近一次弹性调整的报告值")
    result.fields("strata_context", "KV、上下文与推理配置", engine, STRATA_CONTEXT, "/metrics engine")
    result.fields("strata_live", "当前请求与吞吐", data.get("live"), STRATA_LIVE, "/metrics live", "空值表示当前空闲或引擎未报告")
    # These are distinct upstream fields. Strata 0.1.43 uses a 2s window,
    # falling back to the request mean when its samples span less than .25s.
    for metric in result.groups.get('strata_live', {}).get('metrics', []):
        if metric['id'] == 'tok_s':
            metric['description'] += '；后端 token 采样窗口，不是页面刷新间隔；窗口样本跨度不足 0.25s 时回退为当前请求均值；并行时为各请求窗口速度之和'
        elif metric['id'] == 'tok_s_mean':
            metric['description'] += '；从首 token 起的请求均值；并行时为各请求均值之和'
    result.fields("strata_totals", "累计请求与 token", data.get("totals"), STRATA_TOTALS, "/metrics totals", "自后端启动以来累计，非每秒速率")
    result.fields("strata_cache", "会话缓存与 token 复用", data.get("conversation_cache"), STRATA_CACHE, "/metrics conversation_cache")
    last_requests = data.get("requests")
    if isinstance(last_requests, list) and last_requests and isinstance(last_requests[0], dict):
        result.fields("strata_last", "最近完成请求的引擎统计", last_requests[0], STRATA_LAST, "/metrics requests[0]", "最近一次请求的统计，不是实时值")
    else:
        result.unavailable.append("Strata 尚无最近完成请求统计；缓存命中、PCIe 分担和文件读取不能当作 0")
    live = data.get("live") if isinstance(data.get("live"), dict) else {}
    slots = live.get("slots")
    if isinstance(slots, list):
        for slot in slots:
            if not isinstance(slot, dict) or not isinstance(slot.get("slot"), int):
                continue
            sid = slot["slot"]
            result.fields(f"strata_slot_{sid}", f"Strata 批处理槽位 {sid}", slot,
                          [("state", "槽位状态", "", "text"),
                           ("held_tokens", "空闲槽位保留 token", "token", "gauge"),
                           ("prompt_tokens", "当前输入 token", "token", "gauge"),
                           ("generated", "当前已生成 token", "token", "gauge"),
                           ("elapsed_s", "当前请求耗时", "s", "gauge"),
                           ("tok_s", "槽位当前请求平均解码速度", "token/s", "gauge")], "/metrics live.slots")
    if live.get("state") in {"reading", "generating", "unloaded", "idle"}:
        result.status = live["state"]
    elif health.get("loaded") is False:
        result.status = "unloaded"
    elif health.get("loaded") is True:
        result.status = "loaded"
    for key, label in [("expert_cache_mib", "GPU 专家权重缓存容量"), ("arena_mib", "RAM 专家权重区"), ("kv_resident", "GPU KV 驻留容量")]:
        if engine.get(key) is None:
            result.unavailable.append(f"Strata 当前引擎未报告{label}")
    result.unavailable.append("会话缓存 park/restore/eviction 依赖原生日志；为避免保存会话内容已禁用，不展示其默认零值")
    result.unavailable.append("Strata 未通过这些端点提供独立 KV 字节占用或全部权重的实时 RAM 驻留字节")


# name -> group, title, Chinese label, unit, semantics. Whitelist excludes
# request IDs, model identifiers, custom labels, Python/process/system metrics.
LLAMA_PROM = {
    "requests_processing": ("llama_requests", "请求与槽位", "正在处理请求", "个", "gauge"),
    "requests_deferred": ("llama_requests", "请求与槽位", "等待请求", "个", "gauge"),
    "prompt_tokens_seconds": ("llama_throughput", "采集窗口吞吐", "预填充速度", "token/s", "gauge"),
    "predicted_tokens_seconds": ("llama_throughput", "后端采集统计", "已完成请求解码速度（采集窗口）", "token/s", "gauge"),
    "n_busy_slots_per_decode": ("llama_requests", "请求与槽位", "每次 decode 平均繁忙槽位", "个", "gauge"),
    "prompt_tokens_total": ("llama_totals", "累计 token 与计算时间", "累计预填充 token（不含缓存）", "token", "counter"),
    "prompt_tokens_cached_total": ("llama_totals", "累计 token 与计算时间", "累计缓存复用 token", "token", "counter"),
    "prompt_seconds_total": ("llama_totals", "累计 token 与计算时间", "累计预填充耗时", "s", "counter"),
    "tokens_predicted_total": ("llama_totals", "累计 token 与计算时间", "累计生成 token", "token", "counter"),
    "tokens_predicted_seconds_total": ("llama_totals", "累计 token 与计算时间", "累计生成耗时", "s", "counter"),
    "n_decode_total": ("llama_totals", "累计 token 与计算时间", "累计 decode 调用", "次", "counter"),
    "n_tokens_max": ("llama_context", "上下文观测", "历史最长序列", "token", "gauge"),
    "n_tokens_max_total": ("llama_context", "上下文观测", "历史最长序列", "token", "gauge"),
    "kv_cache_usage_ratio": ("llama_context", "上下文观测", "KV 使用比例", "比例", "gauge"),
    "kv_cache_tokens": ("llama_context", "上下文观测", "KV 缓存 token", "token", "gauge"),
    "spec_decode_num_draft_tokens_total": ("llama_spec", "累计推测解码", "累计草稿 token", "token", "counter"),
    "spec_decode_num_accepted_tokens_total": ("llama_spec", "累计推测解码", "累计接受草稿 token", "token", "counter"),
    "spec_decode_num_drafts_total": ("llama_spec", "累计推测解码", "累计草稿验证步骤", "次", "counter"),
    "spec_decode_num_accepted_tokens_per_pos_total": ("llama_spec", "累计推测解码", "各位置累计接受草稿 token", "token", "counter"),
}
VLLM_PROM = {
    "num_requests_running": ("vllm_requests", "请求调度", "运行请求", "个", "gauge"),
    "num_requests_waiting": ("vllm_requests", "请求调度", "等待请求", "个", "gauge"),
    "num_requests_waiting_by_reason": ("vllm_requests", "请求调度", "按原因等待请求", "个", "gauge"),
    "engine_sleep_state": ("vllm_requests", "请求调度", "引擎睡眠状态指示", "", "gauge"),
    "kv_cache_usage_perc": ("vllm_cache", "KV 与前缀缓存", "KV 缓存使用比例", "比例", "gauge"),
    "gpu_cache_usage_perc": ("vllm_cache", "KV 与前缀缓存", "GPU KV 缓存使用比例", "比例", "gauge"),
    "cpu_cache_usage_perc": ("vllm_cache", "KV 与前缀缓存", "CPU KV 缓存使用比例", "比例", "gauge"),
    "prefix_cache_queries_total": ("vllm_cache", "KV 与前缀缓存", "累计前缀缓存查询 token", "token", "counter"),
    "prefix_cache_hits_total": ("vllm_cache", "KV 与前缀缓存", "累计前缀缓存命中 token", "token", "counter"),
    "external_prefix_cache_queries_total": ("vllm_cache", "KV 与前缀缓存", "累计外部前缀缓存查询 token", "token", "counter"),
    "external_prefix_cache_hits_total": ("vllm_cache", "KV 与前缀缓存", "累计外部前缀缓存命中 token", "token", "counter"),
    "mm_cache_queries_total": ("vllm_cache", "KV 与前缀缓存", "累计多模态缓存查询", "个", "counter"),
    "mm_cache_hits_total": ("vllm_cache", "KV 与前缀缓存", "累计多模态缓存命中", "个", "counter"),
    "num_preemptions_total": ("vllm_totals", "累计请求与 token", "累计请求抢占", "次", "counter"),
    "prompt_tokens_total": ("vllm_totals", "累计请求与 token", "累计输入 token", "token", "counter"),
    "prompt_tokens_by_source_total": ("vllm_totals", "累计请求与 token", "按来源累计输入 token", "token", "counter"),
    "prompt_tokens_cached_total": ("vllm_totals", "累计请求与 token", "累计复用输入 token", "token", "counter"),
    "generation_tokens_total": ("vllm_totals", "累计请求与 token", "累计生成 token", "token", "counter"),
    "request_success_total": ("vllm_totals", "累计请求与 token", "累计成功请求", "个", "counter"),
    "corrupted_requests_total": ("vllm_totals", "累计请求与 token", "累计损坏请求", "个", "counter"),
    "avg_prompt_throughput_toks_per_s": ("vllm_throughput", "后端吞吐报告", "平均预填充速度", "token/s", "gauge"),
    "avg_generation_throughput_toks_per_s": ("vllm_throughput", "后端吞吐报告", "平均解码速度", "token/s", "gauge"),
}
VLLM_HISTOGRAMS = {
    "time_to_first_token_seconds": "首 token 等待",
    "inter_token_latency_seconds": "token 间隔",
    "request_time_per_output_token_seconds": "请求每输出 token 耗时",
    "e2e_request_latency_seconds": "请求端到端延迟",
    "request_queue_time_seconds": "请求排队耗时",
    "request_inference_time_seconds": "请求推理耗时",
    "request_prefill_time_seconds": "请求预填充耗时",
    "request_decode_time_seconds": "请求解码耗时",
}
SGLANG_PROM = {
    'num_running_reqs': ('sglang_requests', '请求调度', '运行请求', '个', 'gauge'),
    'num_queue_reqs': ('sglang_requests', '请求调度', '排队请求', '个', 'gauge'),
    'gen_throughput': ('sglang_throughput', '后端吞吐报告', '模型生成吞吐', 'token/s', 'gauge'),
    'token_usage': ('sglang_cache', 'KV 使用', 'Token 池使用比例', '比例', 'gauge'),
    'full_token_usage': ('sglang_cache', 'KV 使用', '完整 Token 池使用比例', '比例', 'gauge'),
    'prompt_tokens_total': ('sglang_totals', '累计 Token', '累计输入 Token', 'token', 'counter'),
    'generation_tokens_total': ('sglang_totals', '累计 Token', '累计生成 Token', 'token', 'counter'),
    'spec_accept_length': ('sglang_spec', 'MTP 采集窗口', '每次验证接受长度（含 bonus）', 'token', 'gauge'),
    'spec_accept_rate': ('sglang_spec', 'MTP 采集窗口', '草稿接受率', '比例', 'gauge'),
}
NINFER_PROM = {
    'requests_running': ('ninfer_requests', '请求调度', '运行请求', '个', 'gauge'),
    'requests_waiting': ('ninfer_requests', '请求调度', '等待接纳请求', '个', 'gauge'),
    'requests_paused': ('ninfer_requests', '请求调度', '暂停等待恢复请求', '个', 'gauge'),
    'max_context_tokens': ('ninfer_context', '上下文配置', '每请求上下文容量', 'token', 'gauge'),
    'prompt_tokens_total': ('ninfer_tokens', '累计 Token', '输入 Token', 'token', 'counter'),
    'prompt_tokens_cached_total': ('ninfer_tokens', '累计 Token', '缓存命中 Token', 'token', 'counter'),
    'prefill_tokens_total': ('ninfer_tokens', '累计 Token', '实际预填充 Token', 'token', 'counter'),
    'replayed_tokens_total': ('ninfer_tokens', '累计 Token', '回放 Token', 'token', 'counter'),
    'generation_tokens_total': ('ninfer_tokens', '累计 Token', '生成 Token', 'token', 'counter'),
    'decode_tokens_total': ('ninfer_tokens', '累计 Token', '解码 Token', 'token', 'counter'),
    'spec_decode_rounds_total': ('ninfer_spec', '累计 MTP', '推测验证轮数', '轮', 'counter'),
    'spec_decode_draft_tokens_total': ('ninfer_spec', '累计 MTP', '提出的草稿 Token', 'token', 'counter'),
    'spec_decode_accepted_tokens_total': ('ninfer_spec', '累计 MTP', '接受的草稿 Token', 'token', 'counter'),
    'spec_decode_fallback_steps_total': ('ninfer_spec', '累计 MTP', '回退步骤', '步', 'counter'),
}
SPARK_PROM = {
    'uptime_seconds': ('sparkinfer_service', '服务状态', '进程运行时间', 's', 'gauge'),
    'requests_total': ('sparkinfer_requests', '累计请求', '请求', '个', 'counter'),
    'requests_streaming_total': ('sparkinfer_requests', '累计请求', '流式请求', '个', 'counter'),
    'requests_by_outcome_total': ('sparkinfer_requests', '累计请求', '按结果请求', '个', 'counter'),
    'tokens_total': ('sparkinfer_tokens', '累计 Token', 'Token', 'token', 'counter'),
    'speculative_runs_total': ('sparkinfer_spec', '累计草稿加速', '推测解码请求', '个', 'counter'),
    'speculative_tokens_total': ('sparkinfer_spec', '累计草稿加速', '推测解码产生的 Token', 'token', 'counter'),
    'speculative_handoffs_total': ('sparkinfer_spec', '累计草稿加速', '调度交接', '次', 'counter'),
    'speculative_tier_stops_total': ('sparkinfer_spec', '累计草稿加速', '分层停止', '次', 'counter'),
    'active_requests': ('sparkinfer_occupancy', '实时占用', '活跃请求', '个', 'gauge'),
    'waiting_requests': ('sparkinfer_occupancy', '实时占用', '等待请求', '个', 'gauge'),
    'free_kv_blocks': ('sparkinfer_occupancy', '实时占用', '空闲 KV 块', '块', 'gauge'),
    'prefix_cache_lookups_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计前缀查找', '次', 'counter'),
    'prefix_cache_hits_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计前缀命中', '次', 'counter'),
    'prefix_cache_tokens_reused_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计复用输入 Token', 'token', 'counter'),
    'prefix_cache_evictions_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计淘汰前缀', '个', 'counter'),
    'prefix_cache_entries': ('sparkinfer_prefix_cache', '前缀缓存', '缓存前缀', '个', 'gauge'),
    'prefix_cache_kv_blocks': ('sparkinfer_prefix_cache', '前缀缓存', '缓存持有 KV 块', '块', 'gauge'),
    'prefix_cache_host_bytes': ('sparkinfer_prefix_cache', '前缀缓存', '循环状态快照主机内存', 'bytes', 'gauge'),
    'prefix_cache_host_hits_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计主机 KV 层命中', '次', 'counter'),
    'prefix_cache_demotions_total': ('sparkinfer_prefix_cache', '前缀缓存', '累计移入主机 KV 层前缀', '个', 'counter'),
    'prefix_cache_host_entries': ('sparkinfer_prefix_cache', '前缀缓存', '仅主机 KV 层缓存前缀', '个', 'gauge'),
    'prefix_cache_host_kv_bytes': ('sparkinfer_prefix_cache', '前缀缓存', '主机 KV 层内存', 'bytes', 'gauge'),
}
LABELS = {"engine", "engine_id", "engine_index", "dp_rank", "tp_rank", "pp_rank", "reason", "finished_reason", "sleep_state", "position", "source", "kind", "outcome"}


def _prometheus(text: str, backend: str, result: Metrics):
    table = {'llama_cpp': LLAMA_PROM, 'vllm': VLLM_PROM, 'sglang': SGLANG_PROM, 'ninfer': NINFER_PROM, 'sparkinfer': SPARK_PROM}[backend]
    prefix = {'llama_cpp': 'llamacpp:', 'vllm': 'vllm:', 'sglang': 'sglang:', 'ninfer': 'ninfer_', 'sparkinfer': 'sparkinfer_'}[backend]
    histogram: dict[tuple, dict] = {}
    seen = 0
    try:
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if not sample.name.startswith(prefix):
                    continue
                name = sample.name.removeprefix(prefix)
                # SGLang's tokenizer counters split by is_streaming; scheduler
                # series also carry model/engine/rank/priority and configured
                # extra labels. Preserve every native dimension in both the
                # stable identity and readable label, without aggregating them.
                labels = tuple(sorted((k, v) for k, v in sample.labels.items() if backend == 'sglang' or k in LABELS))
                suffix = " · " + ", ".join(f"{k}={v}" for k, v in labels) if labels else ""
                ident = hashlib.sha256(repr(labels).encode()).hexdigest()[:12] if labels else ""
                if name in table:
                    group, title, label, unit, kind = table[name]
                    description = f"来源 /metrics · {sample.name}；"
                    description += "后端启动以来累计，非每秒速率" if kind == "counter" else "后端当前报告值"
                    if backend == 'sglang' and sample.labels.get('is_streaming') in {'true', 'false'}:
                        description += '；流式请求序列' if sample.labels['is_streaming'] == 'true' else '；非流式请求序列'
                    if backend == 'sglang' and name in {'spec_accept_length', 'spec_accept_rate'}:
                          description += '；decode 采集窗口均值（默认每 40 次 decode 更新），不是启动以来累计'
                          description += '；每次 forward 接受的草稿加 bonus token' if name == 'spec_accept_length' else '；接受草稿 / 提出草稿'
                    elif backend == 'sglang' and name == 'gen_throughput':
                        description += '；模型范围采样吞吐，不能归属到当前对话请求'
                    elif backend in {'ninfer', 'sparkinfer'} and kind == 'counter':
                        description += '；包含进程预热后的累计，重启归零，不换算成当前请求速度'
                    if backend == "llama_cpp" and name in {"prompt_tokens_seconds", "predicted_tokens_seconds"}:
                        description += "；llama.cpp 在相邻 metrics 采集间的窗口均值，其他抓取器也会影响窗口"
                        if name == "predicted_tokens_seconds":
                            description += "；完成请求时更新，不作为生成中实时速度；实时速度使用 /slots 的 token 计数"
                    result.add(group, title, name + ("_" + ident if ident else ""), label + suffix, sample.value, unit, kind, description)
                    seen += 1
                elif backend == "vllm" and name.endswith(("_sum", "_count")):
                    base, component = name.rsplit("_", 1)
                    if base in VLLM_HISTOGRAMS:
                        histogram.setdefault((base, labels), {})[component] = sample.value
    except (ValueError, TypeError):
        result.unavailable.append("/metrics Prometheus 数据格式无法解析")
        return
    for (name, labels), components in histogram.items():
        ident = hashlib.sha256(repr(labels).encode()).hexdigest()[:12] if labels else ""
        suffix = " · " + ", ".join(f"{k}={v}" for k, v in labels) if labels else ""
        label = VLLM_HISTOGRAMS[name]
        description = f"来源 /metrics · vllm:{name}；自后端启动以来累计分布，非当前请求延迟或分位数"
        key = name + ("_" + ident if ident else "")
        for component in ("sum", "count"):
            if component in components:
                result.add("vllm_latency", "累计延迟分布", key + "_" + component,
                           label + ("累计耗时" if component == "sum" else "累计样本数") + suffix,
                           components[component], "s" if component == "sum" else "个", "counter", description)
                seen += 1
        count, total = components.get("count"), components.get("sum")
        if count is not None and total is not None:
            mean = total / count if count > 0 and math.isfinite(count) and math.isfinite(total) else None
            result.add("vllm_latency", "累计延迟分布", key + "_mean", label + "累计均值" + suffix, mean, "s", "gauge", description)
    if not seen:
        result.unavailable.append("/metrics 未提供可用的引擎指标")
    if backend == "vllm":
        result.unavailable.append("vLLM 累计 token 计数未换算成实时速度；延迟仅展示累计 sum/count/均值，不推测分位数")
    elif backend == 'ninfer':
        result.unavailable.append('NInfer 累计 Token/轮/步计数不换算为当前请求速度；实时速度来自当前 SSE 请求的累计 Token timings')
    elif backend == 'sparkinfer':
        result.unavailable.append('SparkInfer 输出 Token 计数仅在请求结束结算，当前版本无普通解码实时计数；未提供通用的 draft 提出/接受分母，不把推测产生 Token 当成接受率')


def _llama_slots(data: list, result: Metrics):
    for slot in data:
        if not isinstance(slot, dict) or not isinstance(slot.get("id"), int):
            continue
        sid = slot["id"]
        result.fields(f"llama_slot_{sid}", f"槽位 {sid}", slot,
                      [("is_processing", "正在处理请求", "", "text"),
                       ("n_ctx", "槽位上下文容量", "token", "gauge"),
                       ("n_prompt_tokens", "槽位已持有 token", "token", "gauge"),
                       ("n_prompt_tokens_processed", "本次实际预填充 token", "token", "gauge"),
                       ("n_prompt_tokens_cache", "本次缓存复用 token", "token", "gauge"),
                       ("speculative", "推测解码可用", "", "text")], "/slots")
        next_tokens = slot.get("next_token")
        if isinstance(next_tokens, list) and next_tokens and isinstance(next_tokens[0], dict):
            result.fields(f"llama_slot_{sid}", f"槽位 {sid}", next_tokens[0],
                          [("n_decoded", "本次已解码 token", "token", "gauge"),
                           ("n_remain", "本次剩余生成预算", "token", "gauge")], "/slots next_token")


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _live_generation(backend, received, result):
    """Model-scoped rates only; backend snapshots contain no gateway request ID."""
    overview = {}
    extra = {}
    prefill_source = None
    def field(key):
        values = [metric['value'] for group in result.groups.values() for metric in group['metrics']
                  if metric['id'] == key or metric['id'].startswith(key+'_') and len(metric['id']) == len(key)+13]
        return _number(values[0]) if len(values) == 1 else None
    if backend == 'strata':
        data = received.get('/metrics')
        live = data.get('live') if isinstance(data, dict) else None
        if not isinstance(live, dict):
            return None
        running = _number(live.get('running'))
        if running is None and not live.get('parallel') and live.get('state') in ('reading', 'generating', 'idle', 'unloaded'):
            running = 1 if live['state'] in ('reading', 'generating') else 0
        count = _number(live.get('generated')) if live.get('state') == 'generating' else None
        elapsed = _number(live.get('elapsed_s'))
        slot_id = None
        source = '/metrics live.generated'
        queued = _number(live.get('queued'))
        engine = data.get('engine') if isinstance(data.get('engine'), dict) else {}
        prompt = _number(live.get('prompt_tokens'))
        generated = _number(live.get('generated'))
        prompt_read, total = _number(live.get('prompt_read')), _number(live.get('prompt_total'))
        phase = 'prefill' if live.get('state') == 'reading' and prompt_read is not None else 'waiting' if live.get('state') == 'reading' else live.get('state')
        overview = {'phase': phase, 'prompt_tokens': prompt, 'generated_tokens': generated,
                    'average_tokens_per_second': _number(live.get('tok_s_mean')) if live.get('state') == 'generating' else None,
                    # prompt_read includes reused tokens. Without SSE's cache
                    # counter this position cannot identify computed work.
                    'prefill_total_tokens': total,
                    'average_prefill_tokens_per_second': _number(live.get('prefill_tok_s_mean')) if live.get('state') in {'reading', 'generating'} else None,
                    'context_tokens': prompt + generated if prompt is not None and generated is not None else None,
                    'context_capacity': _number(engine.get('max_context')), 'context_basis': '输入 + 已生成'}
    elif backend == 'llama_cpp':
        # Multiple labeled engines or active slots are ambiguous. The metric
        # gauge only accounts for completed requests on verified llama builds.
        running, queued = field('requests_processing'), field('requests_deferred')
        slots = received.get('/slots')
        active = [slot for slot in slots if isinstance(slot,dict) and type(slot.get('id')) is int and slot.get('is_processing') is True] if isinstance(slots,list) else []
        count, elapsed, slot_id = None, None, None
        if len(active) == 1:
            slot = active[0]
            next_tokens = slot.get('next_token')
            if isinstance(next_tokens,list) and next_tokens and isinstance(next_tokens[0],dict):
                count = _number(next_tokens[0].get('n_decoded'))
            elapsed, slot_id = _number(slot.get('elapsed_s')), slot['id']
            processed, cached = _number(slot.get('n_prompt_tokens_processed')), _number(slot.get('n_prompt_tokens_cache'))
            overview = {'phase': 'generating' if count else 'prefill' if processed is not None else 'waiting', 'generated_tokens': count,
                        'prefill_processed_tokens': processed, 'prefill_cached_tokens': cached,
                        'context_tokens': _number(slot.get('n_prompt_tokens')),
                        'context_capacity': _number(slot.get('n_ctx')), 'context_basis': '槽位已持有 Token'}
            prefill_source = '/slots n_prompt_tokens_processed' if processed is not None else None
        source = '/slots next_token.n_decoded'
    elif backend in {'ninfer', 'vllm', 'sglang', 'sparkinfer'}:
        running_key, queued_key = {'ninfer': ('requests_running', 'requests_waiting'),
                                   'vllm': ('num_requests_running', 'num_requests_waiting'),
                                   'sglang': ('num_running_reqs', 'num_queue_reqs'),
                                   'sparkinfer': ('active_requests', 'waiting_requests')}[backend]
        running, queued = field(running_key), field(queued_key)
        # These counters are model-global and include other requests. Only the
        # gateway's exact SSE usage/timings can supply current-request tokens.
        count, elapsed, slot_id = None, None, None
        source = '/metrics 请求调度；等待当前 SSE 请求的累计 Token 计数'
        if backend == 'ninfer':
            overview['context_capacity'] = field('max_context_tokens')
            extra['backend_paused'] = field('requests_paused')
        elif backend == 'sparkinfer':
            info = received.get('/v1/info')
            overview['context_capacity'] = _number(info.get('max_context')) if isinstance(info, dict) else None
        elif backend in {'vllm', 'sglang'}:
            overview['context_capacity'] = field('configured_context_tokens')
            overview['context_capacity_basis'] = '启动配置上限'
    else:
        return None
    return {'tokens_per_second': None, 'window_s': None, 'scope': 'model', 'source': source,
            'backend_running': running, 'backend_queued': queued,
            **extra,
            'available_for_chat': False, 'reason': '等待当前请求的实时 token 采样',
            '_generated_tokens': count, '_elapsed_s': elapsed, '_slot_id': slot_id,
            '_prefill_source': prefill_source, '_overview': overview}


async def collect_telemetry(instance: RuntimeInstance, spec: LaunchSpec) -> dict[str, Any]:
    result = Metrics()
    backend = instance.backend
    if backend not in {"strata", "llama_cpp", "vllm", "transformers", "custom", "sglang", "ninfer", "sparkinfer"}:
        return {"groups": [], "unavailable": ["此后端尚未提供监控适配"]}
    paths = [(spec.health_path, True)] if backend == "custom" else [("/health", False)]
    if backend == "strata":
        paths.append(("/metrics", False))
    elif backend in {"llama_cpp", "vllm", "sglang", "ninfer", "sparkinfer"}:
        paths.append(("/metrics", True))
        if backend == 'sparkinfer':
            paths.append(('/v1/info', False))
        if backend == "llama_cpp":
            if spec.env.get("LLAMA_SERVER_SLOTS_DEBUG") == "0":
                paths.append(("/slots", False))
            else:
                result.unavailable.append("未确认 slots 调试关闭，为避免读取正文未采集 /slots；重新加载后可用")
    # This client can only use HTTP loopback, with redirects disabled. Avoid
    # rebuilding the unused TLS trust store on every poll; it can take hundreds
    # of milliseconds and miss an entire short generation's sampling window.
    client = await asyncio.to_thread(httpx.AsyncClient, base_url=f"http://127.0.0.1:{instance.port}", timeout=3, trust_env=False, verify=False, follow_redirects=False)
    async with client:
        responses = await asyncio.gather(*(_fetch(client, path, text=is_text) for path, is_text in paths))
    received = {}
    for (path, _), (data, error) in zip(paths, responses):
        if error:
            result.unavailable.append(error)
        else:
            received[path] = data
    if backend == "custom":
        result.status = "ready" if spec.health_path in received else "unreachable"
        result.add("custom_service", "自定义服务", "health", "服务健康", result.status, kind="text",
                   description="来源 " + spec.health_path + " 的 HTTP 状态")
        result.unavailable.append("自定义引擎没有通用的 KV、专家、吞吐或实时 token 指标映射；仅展示健康检查、进程资源与后端实际返回的请求 usage/timings")
        return {**result.result(), "live_generation": None}
    health = received.get("/health")
    if isinstance(health, dict):
        result.status = "ready" if all(health.get(k) == v for k, v in spec.health_expected.items()) else "not_ready"
        if backend == "strata" and health.get("loaded") is False:
            result.status = "unloaded"
    else:
        result.status = "unreachable"
    if backend == "strata":
        data = received.get("/metrics")
        if isinstance(data, dict):
            _strata(data, health if isinstance(health, dict) else {}, result)
        elif data is not None:
            result.unavailable.append("Strata /metrics 未返回 JSON 对象")
    elif backend in {"llama_cpp", "vllm", "sglang", "ninfer", "sparkinfer"}:
        text = received.get("/metrics")
        if isinstance(text, str):
            _prometheus(text, backend, result)
        slots = received.get("/slots")
        if isinstance(slots, list):
            _llama_slots(slots, result)
        if backend == 'sparkinfer':
            result.fields('sparkinfer_context', '上下文配置', received.get('/v1/info'),
                          [('max_context', '每请求上下文容量', 'token', 'gauge')], '/v1/info')
        if backend in {'vllm', 'sglang'}:
            configured = _number(spec.capabilities.get('configured_context_tokens'))
            if configured is not None and configured > 0:
                result.add(backend+'_context', '上下文配置', 'configured_context_tokens', '启动配置上下文上限', configured, 'token',
                           description='来源本实例启动 argv '+str(spec.capabilities.get('configured_context_flag', ''))+'；每请求配置上限，不是实时 KV 池容量')
    else:
        result.add("transformers_service", "Transformers 服务", "model", "固定加载模型", spec.model_name, kind="text",
                   description="来源本实例启动配置；force_model 在监听前同步加载，健康状态单独检查 /health")
        result.add("transformers_service", "Transformers 服务", "modalities", "配置声明的输入类型", ", ".join(spec.capabilities.get("input", ["text"])), kind="text",
                   description="来源：模型元数据")
        result.add("transformers_service", "Transformers 服务", "health", "服务健康", result.status, kind="text", description="来源 /health")
        result.unavailable.append("Transformers 5.19 serve 未提供请求、KV、专家或吞吐指标端点")
    return {**result.result(), 'live_generation': _live_generation(backend, received, result)}
