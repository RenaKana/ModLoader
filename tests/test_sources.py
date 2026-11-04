from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import httpx
import pytest

from modloader.models import AppPaths, DownloadRequest, SourceFile, SourceRepository
from modloader.sources import SourceError, SourceService
from modloader.sources._common import NETWORK_ATTEMPTS
from modloader.sources.planning import build_plan, classify_repository


@pytest.mark.parametrize("encoding", ["utf-8", "gbk"])
def test_worker_protocol_preserves_unicode_with_windows_stdout(monkeypatch, encoding):
    from modloader.sources import worker

    raw = io.BytesIO()
    output = io.TextIOWrapper(raw, encoding=encoding)
    monkeypatch.setattr(worker, "_OUTPUT", output)
    data = {"items": [{"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "千问 / 🦙"}]}
    worker.emit("result", data)
    worker.emit("error", message="仓库不存在")

    # The parent reads raw pipe bytes as JSON, without a locale-specific decoder.
    assert [json.loads(line) for line in raw.getvalue().splitlines()] == [
        {"type": "result", "data": data},
        {"type": "error", "message": "仓库不存在"},
    ]


def repository(names: list[str], source: str = "hf") -> SourceRepository:
    return SourceRepository(source=source, repo_id="test/model", revision="commit", name="model", files=[SourceFile(path=name, size=8) for name in names])


async def no_json(_):
    return {}


def deepseek_shards(quantization, directory=""):
    return [f"{directory}DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quantization}-{index:05d}-of-00004.gguf" for index in range(1, 5)]


@pytest.mark.asyncio
async def test_gguf_plan_completes_only_selected_quantization_without_vision_or_hf_metadata():
    repo = repository([
        "model-Q4_K_M-00001-of-00002.gguf", "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q8_0.gguf", "mmproj-f16.gguf", "config.json", "tokenizer.json", "README.md",
    ])
    plan = await build_plan(DownloadRequest(source="hf", files=["model-Q4_K_M-00001-of-00002.gguf"]), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {
        "model-Q4_K_M-00001-of-00002.gguf", "model-Q4_K_M-00002-of-00002.gguf",
    }
    assert plan["total_bytes"] == 16
    assert [file["path"] for file in plan["automatic_files"]] == ["model-Q4_K_M-00002-of-00002.gguf"]
    assert plan["automatic_files"][0]["reason"] == "同一模型的必要分片"
    assert [file["path"] for file in plan["optional_files"]] == ["mmproj-f16.gguf"]
    assert len(plan["model_groups"]) == 1
    assert plan["vision_groups"] == []


@pytest.mark.asyncio
async def test_enabled_vision_adds_one_complete_projector_family_and_maps_subfolder_models():
    repo = repository([
        "q4/model.gguf", "q8/model.gguf", "mmproj-f16-00001-of-00002.gguf", "mmproj-f16-00002-of-00002.gguf",
    ])
    plan = await build_plan(DownloadRequest(source="hf", files=["q4/model.gguf", "q8/model.gguf"], include_vision=True), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {file.path for file in repo.files}
    assert len(plan["model_groups"]) == 2
    assert len(plan["vision_groups"]) == 1
    projector = plan["vision_groups"][0]
    assert all(plan["vision_assignments"][model["id"]] == [projector["id"]] for model in plan["model_groups"])
    assert all(file["role"] == "vision" and file["reason"] for file in plan["automatic_files"])


@pytest.mark.asyncio
@pytest.mark.parametrize("has_projector", [False, True])
async def test_gguf_vision_candidates_exclude_hf_towers(has_projector):
    names = ["model.gguf", "vision_tower/config.json", "vision_tower/model.safetensors"]
    if has_projector:
        names.append("mmproj-f16.gguf")
    plan = await build_plan(DownloadRequest(source="hf", files=["model.gguf"], include_vision=True), repository(names), no_json)
    assert {file["path"] for file in plan["files"]} == {"model.gguf", *(["mmproj-f16.gguf"] if has_projector else [])}
    assert all(group["format"] == "gguf" for group in plan["vision_candidates"])
    assert len(plan["vision_groups"]) == int(has_projector)
    assert plan["vision_assignments"][plan["model_groups"][0]["id"]] == [group["id"] for group in plan["vision_groups"]]
    assert not plan["requires_vision_selection"]


@pytest.mark.asyncio
async def test_gguf_vision_candidates_do_not_associate_an_explicit_raw_hf_tower():
    repo = repository(["model.gguf", "vision_tower/config.json", "vision_tower/model.safetensors"])
    plan = await build_plan(DownloadRequest(source="hf", files=["model.gguf", "vision_tower/model.safetensors"], include_vision=True), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {file.path for file in repo.files}
    assert plan["vision_candidates"] == []
    assert plan["vision_assignments"] == {plan["model_groups"][0]["id"]: []}


@pytest.mark.asyncio
async def test_shared_classification_groups_hf_directories_and_keeps_mtp_and_documents_in_all_files():
    repo = repository([
        "first/config.json", "first/tokenizer.json", "first/model-00001-of-00002.safetensors", "first/model-00002-of-00002.safetensors",
        "second/config.json", "second/model.safetensors", "second/mtp-head.safetensors", "mmproj-f16.gguf", "README.md", "images/preview.png", "notes.json",
    ])
    catalog = await classify_repository(repo, no_json)
    models = [group for group in catalog.groups if group["kind"] == "model"]
    assert len(models) == 2
    assert {tuple(group["files"]) for group in models} == {
        ("first/config.json", "first/model-00001-of-00002.safetensors", "first/model-00002-of-00002.safetensors", "first/tokenizer.json"),
        ("second/config.json", "second/model.safetensors"),
    }
    roles = {file.path: file.role for file in catalog.files}
    assert all(roles[path] == "other" for path in ("second/mtp-head.safetensors", "README.md", "images/preview.png", "notes.json"))
    assert roles["mmproj-f16.gguf"] == "vision"
    plan = await build_plan(DownloadRequest(source="hf", files=["first/model-00001-of-00002.safetensors"]), catalog, no_json)
    assert plan["groups"] == catalog.groups
    assert plan["model_groups"] == [models[0]]
    assert {file["path"] for file in plan["files"]} == set(models[0]["files"])


@pytest.mark.asyncio
@pytest.mark.parametrize("engram", ["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", "engram-IQ4_NL.gguf"])
@pytest.mark.parametrize("quants", [("IQ1_M",), ("IQ1_M", "IQ2_XXS", "IQ2_XS")])
async def test_deepseek_plans_share_one_required_engram_without_another_model(engram, quants):
    variants = ("IQ1_M", "IQ2_XXS", "IQ2_XS")
    draft = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
    repo = repository([*(path for quant in variants for path in deepseek_shards(quant)), engram, draft])
    catalog = await classify_repository(repo, no_json)
    assert len(catalog.groups) == 3
    assert all(group["kind"] == "model" and len(group["files"]) == 4 and group["size"] == 32 for group in catalog.groups)
    engram_file = next(file for file in catalog.files if file.path == engram)
    assert engram_file.role == "required" and engram_file.group_id is None
    assert next(file for file in catalog.files if file.path == draft).group_id is None
    selected = [deepseek_shards(quant)[0] for quant in quants]
    plan = await build_plan(DownloadRequest(source="hf", files=selected), catalog, no_json)
    expected = {path for quant in quants for path in deepseek_shards(quant)} | {engram}
    assert {file["path"] for file in plan["files"]} == expected
    assert len(plan["files"]) == 4 * len(quants) + 1
    assert plan["total_bytes"] == 8 * len(expected)
    assert len(plan["model_groups"]) == len(quants)
    assert all(engram not in group["files"] and draft not in group["files"] for group in plan["model_groups"])
    assert sum(file["path"] == engram for file in plan["automatic_files"]) == 1
    assert next(file for file in plan["automatic_files"] if file["path"] == engram)["reason"] == "DeepSeek V4.1 必需的共享 Engram"
    assert [file["path"] for file in plan["optional_files"]] == [draft]
    assert plan["optional_files"][0]["reason"] == "可选 DSpark 草稿"


@pytest.mark.asyncio
@pytest.mark.parametrize("engram", ["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", "engram-IQ4_NL.gguf"])
async def test_selecting_only_shared_deepseek_engram_does_not_select_any_models(engram):
    repo = repository([*deepseek_shards("IQ2_XXS"), *deepseek_shards("IQ2_XS"), engram])
    plan = await build_plan(DownloadRequest(source="hf", files=[engram]), repo, no_json)
    assert [file["path"] for file in plan["files"]] == [engram]
    assert plan["model_groups"] == plan["automatic_files"] == []
    assert plan["total_bytes"] == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("with_model", [False, True])
async def test_deepseek_dspark_is_manual_and_never_a_primary_model(with_model):
    engram = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
    draft = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
    shards = deepseek_shards("IQ2_XXS")
    repo = repository([*shards, engram, draft])
    selected = [draft, *([shards[0]] if with_model else [])]
    plan = await build_plan(DownloadRequest(source="hf", files=selected), repo, no_json)
    assert {file["path"] for file in plan["files"]} == ({*shards, engram, draft} if with_model else {draft})
    assert len(plan["model_groups"]) == int(with_model)
    assert all(draft not in group["files"] for group in plan["groups"])
    assert all(file["path"] != draft for file in plan["automatic_files"])
    assert plan["optional_files"] == []


@pytest.mark.asyncio
async def test_deepseek_required_engram_can_be_chosen_without_replacing_the_manual_file():
    modern, legacy = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf", "engram-IQ4_NL.gguf"
    shards = deepseek_shards("IQ1_M", "IQ1_M/")
    repo = repository([*shards, modern, legacy])
    default = await build_plan(DownloadRequest(source="hf", files=[shards[0]]), repo, no_json)
    assert {file["path"] for file in default["files"]} == {*shards, modern}
    chosen = await build_plan(DownloadRequest(source="hf", files=[shards[0], legacy]), repo, no_json)
    assert {file["path"] for file in chosen["files"]} == {*shards, legacy}
    assert all(file["path"] != legacy for file in chosen["automatic_files"])
    assert chosen["model_groups"][0]["files"] == shards


@pytest.mark.asyncio
async def test_deepseek_shared_engram_preserves_unknown_size_and_does_not_require_dspark():
    engram = "shared/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
    shards = deepseek_shards("IQ2_XS")
    repo = repository([*shards, engram])
    repo.files[-1].size = None
    plan = await build_plan(DownloadRequest(source="hf", files=[shards[0]]), repo, no_json)
    assert plan["unknown_size_files"] == [engram]
    assert plan["total_bytes"] == 32
    assert plan["optional_files"] == [] and not plan["warnings"]


@pytest.mark.asyncio
async def test_deepseek_missing_engram_and_incomplete_shards_have_actionable_errors():
    shards = deepseek_shards("IQ2_XXS")
    with pytest.raises(SourceError) as error:
        await build_plan(DownloadRequest(source="hf", files=[shards[0]]), repository([*shards, "Other-Engram-IQ4_NL.gguf"]), no_json)
    assert error.value.code == "missing_dependency" and "Engram-IQ4_NL.gguf" in str(error.value)
    with pytest.raises(SourceError) as error:
        await build_plan(DownloadRequest(source="hf", files=[shards[0]]), repository([shards[0], "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"]), no_json)
    assert error.value.code == "missing_shard"


@pytest.mark.asyncio
async def test_multiple_deepseek_engrams_require_a_manual_file_choice():
    shards = deepseek_shards("IQ2_XS")
    engrams = [folder + "/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" for folder in ("first", "second")]
    repo = repository([*shards, *engrams])
    with pytest.raises(SourceError) as error:
        await build_plan(DownloadRequest(source="hf", files=[shards[0]]), repo, no_json)
    assert error.value.code == "dependency_selection_required"
    plan = await build_plan(DownloadRequest(source="hf", files=[shards[0], engrams[1]]), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {*shards, engrams[1]}


@pytest.mark.asyncio
async def test_engram_names_outside_deepseek_v41_keep_their_primary_model_behavior():
    names = ["Engram-IQ4_NL.gguf", "Other-Engram-IQ4_NL.gguf", "DeepSeek-V3-Engram-IQ4_NL.gguf", "dspark-Other-Q8_0.gguf", "Qwen3.8-27B-NVFP4-MTP-HIGH.gguf"]
    catalog = await classify_repository(repository(names), no_json)
    assert {group["files"][0] for group in catalog.groups} == set(names)
    plan = await build_plan(DownloadRequest(source="hf", files=[names[-1]]), catalog, no_json)
    assert [file["path"] for file in plan["files"]] == [names[-1]]
    assert len(plan["model_groups"]) == 1 and plan["optional_files"] == []


@pytest.mark.asyncio
async def test_multiple_model_selection_has_no_fixed_group_limit_and_preserves_unknown_sizes():
    repo = repository([f"model-{index}.gguf" for index in range(140)])
    repo.files[0].size = None
    plan = await build_plan(DownloadRequest(source="hf", files=[file.path for file in repo.files]), repo, no_json)
    assert len(plan["model_groups"]) == 140
    assert len(plan["files"]) == 140
    assert plan["unknown_size_files"] == [repo.files[0].path]
    assert next(group for group in plan["groups"] if repo.files[0].path in group["files"])["size"] is None


@pytest.mark.asyncio
async def test_all_files_can_download_explicit_document_or_mtp_without_importing_a_model():
    repo = repository(["model.gguf", "mtp-head.safetensors", "README.md"])
    plan = await build_plan(DownloadRequest(source="hf", files=["README.md", "mtp-head.safetensors"]), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {"README.md", "mtp-head.safetensors"}
    assert plan["model_groups"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["HIGH", "LOW", "HIGHEST"])
async def test_complete_qwen_gguf_with_embedded_mtp_remains_a_primary_model(variant):
    name = f"Qwen3.8-27B-NVFP4-MTP-{variant}.gguf"
    repo = repository([name, "mtp-Qwen3.8-27B.gguf", "mtp/model.gguf", "strata/rt/predictor.gguf", "strata/rt/gate_up.bin"])
    catalog = await classify_repository(repo, no_json)
    assert [group["files"] for group in catalog.groups] == [[name]]
    assert {file.path: file.role for file in catalog.files} == {name: "model", **{file.path: "other" for file in repo.files if file.path != name}}
    plan = await build_plan(DownloadRequest(source="hf", files=[name]), catalog, no_json)
    assert [file["path"] for file in plan["files"]] == [name]
    assert len(plan["model_groups"]) == 1


@pytest.mark.asyncio
async def test_unsupported_weight_formats_and_strata_runtime_files_stay_in_all_files():
    extras = ["pytorch_model.bin", "pytorch_model.bin.index.json", "weights.pt", "weights.pth", "strata/rt/predictor.safetensors", "strata/rt/model.safetensors.index.json", "strata/rt/gate_up.bin"]
    repo = repository(["config.json", "model.safetensors", *extras])
    catalog = await classify_repository(repo, no_json)
    assert len(catalog.groups) == 1
    assert catalog.groups[0]["files"] == ["config.json", "model.safetensors"]
    assert all(file.role == "other" for file in catalog.files if file.path in extras)
    plan = await build_plan(DownloadRequest(source="hf", files=extras), catalog, no_json)
    assert {file["path"] for file in plan["files"]} == set(extras)
    assert plan["model_groups"] == []
    assert plan["automatic_files"] == []


@pytest.mark.asyncio
async def test_weight_index_adds_referenced_shards_and_tokenizer():
    repo = repository(["weights/model.safetensors.index.json", "weights/first.safetensors", "weights/second.safetensors", "weights/config.json", "weights/tokenizer.model", "alternate.gguf"])

    async def read(path):
        return {"weight_map": {"one": "first.safetensors", "two": "second.safetensors"}} if path.endswith("index.json") else {}

    plan = await build_plan(DownloadRequest(source="hf", files=["weights/model.safetensors.index.json"]), repo, read)
    assert {file["path"] for file in plan["files"]} == set(repo_file.path for repo_file in repo.files if repo_file.path != "alternate.gguf")
    assert {file["role"] for file in plan["automatic_files"]} == {"shard", "required"}
    assert all(file["reason"] for file in plan["automatic_files"])


@pytest.mark.asyncio
async def test_optional_sidecars_remain_user_choices():
    required = {"model.safetensors", "config.json", "tokenizer.json", "generation_config.json", "chat_template.jinja"}
    optional = {"chat_template.json", "params.json"}
    repo = repository(sorted(required | optional))

    async def read(_):
        return {}

    catalog = await classify_repository(repo, read)
    assert set(catalog.groups[0]["files"]) == required
    assert all(file.role == "other" and file.group_id is None for file in catalog.files if file.path in optional)
    for selected in ([], sorted(optional)):
        plan = await build_plan(DownloadRequest(source="hf", files=["model.safetensors", *selected]), catalog, read)
        assert {file["path"] for file in plan["files"]} == required | set(selected)
        assert not any(file["path"] in optional for file in plan["automatic_files"])


@pytest.mark.asyncio
async def test_selected_nonstandard_shard_uses_its_index_without_other_quantization():
    repo = repository(["weights.00.safetensors", "weights.01.safetensors", "model.safetensors.index.json", "other.safetensors", "other.safetensors.index.json"])

    async def read(path):
        return {"weight_map": {"layer": "other.safetensors"}} if path.startswith("other") else {"weight_map": {"first": "weights.00.safetensors", "second": "weights.01.safetensors"}}

    plan = await build_plan(DownloadRequest(source="hf", files=["weights.00.safetensors"]), repo, read)
    assert {file["path"] for file in plan["files"]} == {"weights.00.safetensors", "weights.01.safetensors", "model.safetensors.index.json"}


@pytest.mark.asyncio
async def test_nested_index_resolves_same_named_shards_relative_to_its_own_directory():
    repo = repository(["part.safetensors", "config.json", "tokenizer.json", "nested/part.safetensors", "nested/model.safetensors.index.json"])

    async def read(_):
        return {"weight_map": {"layer": "part.safetensors"}}

    plan = await build_plan(DownloadRequest(source="hf", files=["nested/model.safetensors.index.json"]), repo, read)
    assert {file["path"] for file in plan["files"]} == {"nested/part.safetensors", "nested/model.safetensors.index.json"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reference,code", [("missing.safetensors", "missing_shard"), ("../escape.safetensors", "unsafe_path"), ("C:/escape.safetensors", "unsafe_path")])
async def test_invalid_index_does_not_make_a_partial_plan(reference, code):
    repo = repository(["model.safetensors.index.json", "model.safetensors"])

    async def read(_):
        return {"weight_map": {"layer": reference}}

    with pytest.raises(SourceError) as error:
        await build_plan(DownloadRequest(source="hf", files=["model.safetensors.index.json"]), repo, read)
    assert error.value.code == code


@pytest.mark.asyncio
async def test_incomplete_shard_group_is_rejected_before_download():
    repo = repository(["model-00001-of-00003.gguf", "model-00003-of-00003.gguf"])
    with pytest.raises(SourceError) as error:
        await build_plan(DownloadRequest(source="hf", files=[repo.files[0].path]), repo, no_json)
    assert error.value.code == "missing_shard"


@pytest.mark.asyncio
async def test_ambiguous_vision_projection_is_an_explicit_choice():
    repo = repository(["model.gguf", "mmproj-f16.gguf", "mmproj-Q8_0.gguf"])
    plan = await build_plan(DownloadRequest(source="hf", files=["model.gguf"]), repo, no_json)
    assert [file["path"] for file in plan["files"]] == ["model.gguf"]
    assert {file["path"] for file in plan["optional_files"]} == {"mmproj-f16.gguf", "mmproj-Q8_0.gguf"}
    assert not plan["requires_vision_selection"]
    enabled = await build_plan(DownloadRequest(source="hf", files=["model.gguf"], include_vision=True), repo, no_json)
    assert enabled["requires_vision_selection"]
    assert enabled["warnings"]
    chosen = await build_plan(DownloadRequest(source="hf", files=["model.gguf", "mmproj-Q8_0.gguf"]), repo, no_json)
    assert {file["path"] for file in chosen["files"]} == {"model.gguf", "mmproj-Q8_0.gguf"}


@pytest.mark.asyncio
@pytest.mark.parametrize("include_vision", [False, True])
async def test_local_vision_tower_is_included_when_config_requires_it(include_vision):
    repo = repository(["model.safetensors", "config.json", "vision_tower/model.safetensors", "vision_tower/config.json", "other-Q4.gguf"])

    async def read(_):
        return {"vision_config": {"model_type": "clip"}}

    plan = await build_plan(DownloadRequest(source="hf", files=["model.safetensors"], include_vision=include_vision), repo, read)
    assert {file["path"] for file in plan["files"]} == {"model.safetensors", "config.json", "vision_tower/model.safetensors", "vision_tower/config.json"}
    assert len(plan["model_groups"]) == len(plan["vision_groups"]) == 1
    assert plan["required_vision_groups"] == plan["vision_groups"]
    assert plan["vision_assignments"] == {plan["model_groups"][0]["id"]: [plan["vision_groups"][0]["id"]]}
    assert all(file["reason"].startswith("HF配置要求") for file in plan["automatic_files"] if file["path"].startswith("vision_tower/"))
    assert not plan["requires_vision_selection"]


@pytest.mark.asyncio
async def test_pure_text_hf_config_does_not_require_a_nearby_vision_tower():
    repo = repository(["model.safetensors", "config.json", "vision_tower/model.safetensors", "vision_tower/config.json"])
    plan = await build_plan(DownloadRequest(source="hf", files=["model.safetensors"]), repo, no_json)
    assert {file["path"] for file in plan["files"]} == {"model.safetensors", "config.json"}
    assert plan["vision_groups"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["vision_tower", "mm_vision_tower"])
async def test_explicit_hf_vision_reference_requires_its_local_directory(key):
    repo = repository(["language/model.safetensors", "language/config.json", "language/encoder/model.safetensors", "language/encoder/config.json"])

    async def read(path):
        assert path == "language/config.json"
        return {key: "./encoder"}

    plan = await build_plan(DownloadRequest(source="hf", files=["language/model.safetensors"]), repo, read)
    assert {file["path"] for file in plan["files"]} == {file.path for file in repo.files}
    assert len(plan["model_groups"]) == len(plan["vision_groups"]) == 1
    assert plan["vision_groups"][0]["kind"] == "vision"
    assert plan["vision_assignments"][plan["model_groups"][0]["id"]] == [plan["vision_groups"][0]["id"]]
    assert all(file["role"] == "vision" and file["reason"].startswith("HF配置要求") for file in plan["automatic_files"] if "/encoder/" in file["path"])


@pytest.mark.asyncio
async def test_ambiguous_required_hf_vision_tower_needs_an_explicit_choice_with_vision_off():
    repo = repository(["model.safetensors", "config.json", "vision_tower/a.gguf", "vision_tower/b.gguf"])

    async def read(_):
        return {"vision_config": {"model_type": "clip"}}

    plan = await build_plan(DownloadRequest(source="hf", files=["model.safetensors"]), repo, read)
    assert plan["requires_vision_selection"]
    assert {file["path"] for file in plan["files"]} == {"model.safetensors", "config.json"}
    assert {file["path"] for file in plan["optional_files"]} == {"vision_tower/a.gguf", "vision_tower/b.gguf"}
    chosen = await build_plan(DownloadRequest(source="hf", files=["model.safetensors", "vision_tower/a.gguf"]), repo, read)
    assert not chosen["requires_vision_selection"]
    assert chosen["required_vision_groups"] == chosen["vision_groups"]
    assert {file["path"] for file in chosen["files"]} == {"model.safetensors", "config.json", "vision_tower/a.gguf"}
    assert chosen["vision_assignments"][chosen["model_groups"][0]["id"]] == [chosen["vision_groups"][0]["id"]]


@pytest.mark.asyncio
async def test_external_hf_vision_reference_warns_without_selecting_an_unrelated_local_tower():
    repo = repository(["model.safetensors", "config.json", "vision_tower/model.safetensors", "vision_tower/config.json"])

    async def read(path):
        assert path == "config.json"
        return {"vision_config": {"model_type": "clip"}, "vision_tower": "external/clip-model"}

    plan = await build_plan(DownloadRequest(source="hf", files=["model.safetensors"], include_vision=True), repo, read)
    assert {file["path"] for file in plan["files"]} == {"model.safetensors", "config.json"}
    assert plan["vision_groups"] == []
    assert any("外部视觉编码器" in warning for warning in plan["warnings"])


@pytest.fixture
def service(tmp_path):
    return SourceService(AppPaths(tmp_path / "data"), lambda _: None, lambda: None)


@pytest.mark.asyncio
async def test_deepseek_download_result_registers_only_selected_main_shard_groups(service, monkeypatch):
    from modloader.app import Controller

    engram = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
    draft = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
    variants = [deepseek_shards(quant, quant + "/") for quant in ("IQ2_XXS", "IQ2_XS")]
    names = [*(path for shards in variants for path in shards), engram, draft]
    # Minimal GGUFs exercise the real registration path without model weights.
    key = b"general.architecture"
    def gguf(architecture):
        value = architecture.encode()
        return b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, len(key)) + key + struct.pack("<IQ", 8, len(value)) + value
    contents = {path: gguf("engram" if path == engram else "dflash" if path == draft else "deepseek41") for path in names}
    repo = repository(names)
    for file in repo.files:
        file.size = len(contents[file.path])
    downloads = []
    async def worker(payload, source, progress=None):
        if payload["action"] == "resolve":
            return repo.model_dump()
        assert payload["action"] == "download"
        downloads.append([file["path"] for file in payload["files"]])
        directory = Path(payload["destination"])
        for file in payload["files"]:
            target = directory / file["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[file["path"]])
        return {"path": str(directory), "revision": payload["revision"], "files": downloads[-1]}
    monkeypatch.setattr(service, "_worker", worker)
    controller = Controller(service.paths.root, "management-secret")
    try:
        request = DownloadRequest(source="hf", repo_id="test/model", files=[shards[0] for shards in variants] + [draft],
                                  destination=str(service.paths.models / "deepseek"))
        plan = await service.plan(request)
        request.revision = plan["revision"]
        request.checked_files = [SourceFile.model_validate(file) for file in plan["files"]]
        result = await service.download(request, collect, asyncio.Event())
        assert result["model_groups"] == plan["model_groups"]
        assert downloads[-1].count(engram) == 1 and len(downloads[-1]) == 10
        ids, errors = controller.register_download(request.model_dump(), result)
        assert len(ids) == 2 and errors == []
        models = [controller.store.get("models", identifier) for identifier in ids]
        assert {Path(model["path"]).relative_to(request.destination).as_posix() for model in models} == {shards[0] for shards in variants}
        assert all(len(model["files"]) == 4 and model["architecture"] == "deepseek41" for model in models)
        support = DownloadRequest(source="hf", repo_id="test/model", files=[draft], destination=str(service.paths.models / "draft-only"))
        result = await service.download(support, collect, asyncio.Event())
        assert result["model_groups"] == [] and downloads[-1] == [draft]
        assert controller.register_download(support.model_dump(), result) == ([], [])
        assert len(controller.store.all("models")) == 2
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_resolve_and_download_reuse_classifier_and_keep_the_checked_file_list(service, monkeypatch):
    calls = []

    async def worker(payload, source, progress=None):
        calls.append(payload)
        if payload["action"] == "resolve":
            return repository(["nested/model.safetensors.index.json", "nested/a.safetensors", "nested/b.safetensors", "nested/config.json", "nested/tokenizer.json", "README.md"]).model_dump()
        if payload["action"] == "read_json":
            return {"weight_map": {"one": "a.safetensors", "two": "b.safetensors"}} if payload["file"].endswith("index.json") else {}
        assert payload["action"] == "download"
        return {"files": [file["path"] for file in payload["files"]], "revision": payload["revision"]}

    monkeypatch.setattr(service, "_worker", worker)
    catalog = await service.resolve("hf", "test/model")
    request = DownloadRequest(source="hf", repo_id="test/model", files=["nested/a.safetensors"])
    plan = await service.plan(request)
    assert plan["groups"] == catalog.groups
    request.revision = plan["revision"]
    request.checked_files = [SourceFile.model_validate(file) for file in plan["files"]]
    request.destination = str(service.paths.models / "checked")
    result = await service.download(request, collect, asyncio.Event())
    assert result["files"] == [file["path"] for file in plan["files"]]
    assert calls[-1]["files"] == result["plan_files"] == plan["files"]
    assert result["model_groups"] == plan["model_groups"]
    assert result["vision_assignments"] == plan["vision_assignments"]
    assert calls[-1]["revision"] == plan["revision"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["path", "size", "revision"])
async def test_changed_checked_plan_fails_before_the_download_worker(service, monkeypatch, change):
    files = [{"path": "model.gguf", "size": 8}]
    current = [{"path": "other.gguf" if change == "path" else "model.gguf", "size": 9 if change == "size" else 8}]

    async def plan(_):
        return {"source": "hf", "repo_id": "test/model", "revision": "changed" if change == "revision" else "commit", "files": current, "warnings": []}

    async def worker(*_):
        pytest.fail("Changed plans must not start a download worker")

    monkeypatch.setattr(service, "plan", plan)
    monkeypatch.setattr(service, "_worker", worker)
    request = DownloadRequest(source="hf", repo_id="test/model", revision="commit", files=["model.gguf"], checked_files=files, destination=str(service.paths.models / "changed"))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "resource_changed"
    assert not Path(request.destination).exists()


@pytest.mark.asyncio
async def test_ambiguous_vision_plan_cannot_start_downloading(service, monkeypatch):
    async def plan(_):
        return {"requires_vision_selection": True}

    monkeypatch.setattr(service, "plan", plan)
    request = DownloadRequest(source="hf", repo_id="test/model", files=["model.gguf"], include_vision=True, destination=str(service.paths.models / "ambiguous"))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "vision_selection_required"
    assert not Path(request.destination).exists()


@pytest.mark.asyncio
async def test_ambiguous_required_hf_vision_cannot_start_with_the_default_toggle(service, monkeypatch):
    repo = repository(["model.safetensors", "config.json", "vision_tower/a.gguf", "vision_tower/b.gguf"])

    async def resolve(*_):
        return repo

    async def worker(payload, *_):
        assert payload["action"] == "read_json", "Unresolved required dependencies must not start the download worker"
        return {"vision_config": {"model_type": "clip"}}

    monkeypatch.setattr(service, "resolve", resolve)
    monkeypatch.setattr(service, "_worker", worker)
    request = DownloadRequest(source="hf", repo_id="test/model", files=["model.safetensors"], destination=str(service.paths.models / "required-ambiguous"))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "vision_selection_required"
    assert not Path(request.destination).exists()


@pytest.fixture
def readme_hub(service, monkeypatch):
    state = {"files": ["README.md", "README_zh.md", "model.safetensors"], "content": "\ufeff# 模型说明\n\n下载前查看用途与许可证。🦙\n", "revision": "a" * 40, "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self):
            path = unquote(urlsplit(self.path).path)
            state["requests"].append((self.command, path, self.headers.get("Authorization")))
            if state["status"] != 200:
                data, status = b'{"error":"private-fixture-token"}', state["status"]
            elif path.startswith("/api/models/test/model/revision/"):
                data = json.dumps({"id": "test/model", "sha": state["revision"], "gated": False, "private": False, "pipeline_tag": "text-generation", "siblings": [{"rfilename": name, "size": 8} for name in state["files"]]}).encode()
                status = 200
            elif path.startswith(f"/test/model/resolve/{state['revision']}/") and path.removeprefix(f"/test/model/resolve/{state['revision']}/") in state["files"]:
                data, status = state["content"].encode("utf-8"), 200
            else:
                data, status = b"", 404
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", '"fixture-readme"')
            self.send_header("X-Repo-Commit", state["revision"])
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        do_HEAD = respond
        do_GET = respond

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    # Keep the real subprocess protocol and installed HF SDK. Only redirect its
    # pinned platform endpoint to this local Hub-compatible HTTP fixture.
    bootstrap = """
import sys
import huggingface_hub
from modloader.sources import worker
original_init = worker._HuggingFace.__init__
original_download = huggingface_hub.hf_hub_download
def initialize(self, cache):
    original_init(self, cache)
    self.api.endpoint = sys.argv[1]
def download(**kwargs):
    kwargs['endpoint'] = sys.argv[1]
    return original_download(**kwargs)
worker._HuggingFace.__init__ = initialize
huggingface_hub.hf_hub_download = download
raise SystemExit(worker.main())
"""
    original_launch = asyncio.create_subprocess_exec

    async def launch(*_, **kwargs):
        return await original_launch(sys.executable, "-c", bootstrap, endpoint, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(service, "credential_provider", lambda _: "private-fixture-token")
    yield state
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.mark.parametrize("files,expected_path,requested_revision,file_path", [
    (["README_zh.md", "readme.md", "README.md", "model.safetensors"], "README.md", None, None),
    (["ReadMe.MD", "model.safetensors"], "ReadMe.MD", "b" * 40, None),
    (["docs/README.md", "README_cn.md", "model.safetensors"], "README_cn.md", None, None),
    (["docs/README.md", "model.safetensors"], None, None, None),
    (["README.md", "README.zh-CN.md", "model.safetensors"], "README.zh-CN.md", "a" * 40, "README.zh-CN.md"),
    (["README.md", "strata/README.md", "model.safetensors"], "strata/README.md", "a" * 40, "strata/README.md"),
])
async def test_readme_api_uses_real_sdk_worker_without_downloading_weights(service, readme_hub, files, expected_path, requested_revision, file_path):
    from modloader.app import Controller, build_app

    readme_hub["files"] = files
    controller = Controller(service.paths.root, "management-secret")
    controller.sources = service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(controller)), base_url="http://testserver", headers={"Authorization": "Bearer management-secret"}) as client:
            body = {"source": "hf", "repo_id": "test/model"}
            if requested_revision:
                body["revision"] = requested_revision
            if file_path:
                body["file_path"] = file_path
            response = await client.post("/api/sources/readme", json=body)
        assert response.status_code == 200
        assert response.json() == {"content": readme_hub["content"].lstrip("\ufeff") if expected_path else "", "path": expected_path, "url": "https://huggingface.co/test/model", "revision": readme_hub["revision"]}
        assert readme_hub["requests"][0][1] == f"/api/models/test/model/revision/{requested_revision or 'main'}"
        assert all(auth == "Bearer private-fixture-token" for _, _, auth in readme_hub["requests"])
        files_requested = [path for _, path, _ in readme_hub["requests"] if "/resolve/" in path]
        expected_requests = [f"/test/model/resolve/{readme_hub['revision']}/{expected_path}"] * 2 if expected_path else []
        assert files_requested == expected_requests
        assert not list(service.paths.models.iterdir())
        assert controller.store.all("models") == []
        assert controller.store.all("jobs") == []
    finally:
        await controller.close()


@pytest.mark.parametrize("file_path,code", [
    ("../README.md", "unsafe_path"),
    ("https://example.com/README.md", "unsafe_path"),
    ("C:/README.md", "unsafe_path"),
    ("docs/README.md", "not_found"),
    ("model.safetensors", "invalid_readme"),
    ("README.safetensors", "invalid_readme"),
    ("docs/guide.md", "invalid_readme"),
])
def test_explicit_readme_rejects_unsafe_missing_or_non_document_before_file_fetch(file_path, code):
    from modloader.sources import worker

    fetched = []

    def get_file(*args):
        fetched.append(args)
        raise AssertionError("Invalid README requests must not fetch a file")

    repo = {"files": [{"path": name} for name in ("README.md", "model.safetensors", "README.safetensors", "docs/guide.md")]}
    client = SimpleNamespace(resolve=lambda *_: repo, get_file=get_file)
    with pytest.raises(SourceError) as error:
        worker._readme(client, "test/model", "fixed-commit", file_path)
    assert error.value.code == code
    assert fetched == []


@pytest.mark.parametrize("file_path", ["README", "docs/README-en.markdown", "docs/readme_zh.rst", "docs/README.txt"])
def test_explicit_readme_accepts_listed_document_variants(tmp_path, file_path):
    from modloader.sources import worker

    document = tmp_path / "document"
    document.write_text("Model documentation", encoding="utf-8")
    fetched = []

    def get_file(repo_id, revision, path):
        fetched.append((repo_id, revision, path))
        return document

    repo = {"files": [{"path": file_path}], "revision": "fixed-commit", "url": "https://huggingface.co/test/model"}
    client = SimpleNamespace(resolve=lambda *_: repo, get_file=get_file)
    result = worker._readme(client, "test/model", "main", file_path)
    assert result == {"content": "Model documentation", "path": file_path, "revision": "fixed-commit", "url": repo["url"]}
    assert fetched == [("test/model", "fixed-commit", file_path)]


async def test_readme_worker_authorization_failure_is_finite_and_sanitized(service, readme_hub):
    readme_hub["status"] = 403
    with pytest.raises(SourceError) as error:
        await service.readme("hf", "test/model", readme_hub["revision"])
    assert error.value.code == "authorization_required"
    assert "private-fixture-token" not in str(error.value)
    assert readme_hub["requests"] == [("GET", f"/api/models/test/model/revision/{readme_hub['revision']}", "Bearer private-fixture-token")]


@pytest.fixture
def http_source():
    state = {"data": bytes(range(256)) * 8192, "etag": '"version-one"', "mode": "range", "requests": [], "stop": threading.Event()}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["data"])))
            if state.get('hub_commit'):
                self.send_header('ETag', state['etag'])
                self.send_header('X-Repo-Commit', state['hub_commit'])
            self.end_headers()

        def do_GET(self):
            request_range = self.headers.get("Range")
            state["requests"].append(request_range)
            if state["mode"] == "error":
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = int(request_range.removeprefix("bytes=").removesuffix("-")) if request_range else 0
            if state["mode"] == "ignore":
                start = 0
            if start >= len(state["data"]):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(state['data'])}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            data = state["data"][start:]
            partial = bool(request_range) and state["mode"] != "ignore"
            self.send_response(206 if partial else 200)
            self.send_header("Content-Length", str(len(data)))
            if state["etag"]:
                self.send_header("ETag", state["etag"])
            if partial:
                range_start = 0 if state["mode"] == "bad_range" else start
                self.send_header("Content-Range", f"bytes {range_start}-{len(state['data']) - 1}/{len(state['data'])}")
            self.end_headers()
            try:
                chunk_size = state.get('chunk_size', 16 * 1024)
                for offset in range(0, len(data), chunk_size):
                    if state["stop"].is_set():
                        break
                    self.wfile.write(data[offset: offset + chunk_size])
                    self.wfile.flush()
                    time.sleep(state.get('chunk_delay', 0.004))
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/model.gguf?signature=private-link-value"
    yield state
    state["stop"].set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


async def collect(_):
    pass


async def partial_download(service, source):
    destination = service.paths.models / "direct"
    request = DownloadRequest(source="url", url=source["url"], destination=str(destination))
    cancel = asyncio.Event()

    async def progress(event):
        if event.get("downloaded_bytes", 0) >= 64 * 1024:
            cancel.set()

    with pytest.raises(asyncio.CancelledError):
        await service.download(request, progress, cancel)
    part = destination / "model.gguf.modloader-part"
    assert 0 < part.stat().st_size < len(source["data"])
    size = part.stat().st_size
    await asyncio.sleep(0.05)
    assert part.stat().st_size == size
    journal = (destination / "model.gguf.modloader-meta.json").read_text()
    assert "signature" not in journal and "private-link-value" not in journal
    return request, part, size


@pytest.mark.asyncio
async def test_download_planning_event_precedes_transfer_and_replaces_waiting_state(service, http_source):
    events = []
    cancel = asyncio.Event()

    async def progress(event):
        events.append(event)
        if event.get("phase") == "planning":
            cancel.set()

    request = DownloadRequest(source="url", url=http_source["url"], destination=str(service.paths.models / "planning"))
    with pytest.raises(asyncio.CancelledError):
        await service.download(request, progress, cancel)

    assert events == [{
        "phase": "planning", "message": "正在准备下载清单", "downloaded_bytes": 0,
        "total_bytes": None, "can_pause": False,
    }]
    assert http_source["requests"] == []


@pytest.mark.asyncio
async def test_http_url_progress_events_include_file_stage_and_transfer_feedback(service, http_source):
    events = []

    async def progress(event):
        events.append(event)

    request = DownloadRequest(source="url", url=http_source["url"], destination=str(service.paths.models / "feedback"))
    await service.download(request, progress, asyncio.Event())

    transfers = [event for event in events if event.get("phase") == "downloading"]
    assert transfers
    assert transfers[0]["file"] == "model.gguf"
    assert transfers[0]["message"] == "正在下载 model.gguf"
    assert transfers[0]["file_downloaded_bytes"] == 0
    assert transfers[0]["file_total_bytes"] == len(http_source["data"])
    assert any(event["file_downloaded_bytes"] > 0 for event in transfers)
    assert all(event["file_downloaded_bytes"] == event["downloaded_bytes"] for event in transfers)

    complete = next(event for event in events if event.get("phase") == "complete")
    assert complete["file"] == "model.gguf"
    assert complete["message"] == "文件下载完成：model.gguf"
    assert complete["file_downloaded_bytes"] == len(http_source["data"])
    assert complete["file_total_bytes"] == len(http_source["data"])


@pytest.mark.asyncio
async def test_real_http_cancel_and_etag_range_resume(service, http_source):
    plan = await service.plan(DownloadRequest(source="url", url=http_source["url"]))
    assert plan["total_bytes"] == len(http_source["data"])
    request, part, offset = await partial_download(service, http_source)
    events = []

    async def progress(event):
        events.append(event)

    result = await service.download(request, progress, asyncio.Event())
    assert http_source["requests"][-1] == f"bytes={offset}-"
    assert any(event.get("resumed") for event in events)
    resumed = next(event for event in events if event.get("resumed"))
    assert resumed["message"] == "正在继续下载 model.gguf"
    assert resumed["file"] == "model.gguf" and resumed["file_downloaded_bytes"] == offset
    assert (Path(result["path"]) / "model.gguf").read_bytes() == http_source["data"]
    assert not part.exists()
    assert "signature" not in result["url"]


@pytest.mark.parametrize('stop', [False, True])
async def test_http_pause_stops_writes_and_resumes_same_transfer(service, http_source, stop):
    from modloader.sources.control import DownloadPause

    paused = asyncio.Event()
    async def acknowledged():
        paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    destination = service.paths.models / 'paused-http'
    request = DownloadRequest(source='url', url=http_source['url'], destination=str(destination))

    async def progress(event):
        if event.get('downloaded_bytes', 0) >= 64 * 1024 and pause.generation == 0:
            pause.request()

    task = asyncio.create_task(service.download(request, progress, cancel, pause=pause))
    try:
        await asyncio.wait_for(paused.wait(), 3)
        part = destination / 'model.gguf.modloader-part'
        size = part.stat().st_size
        assert 0 < size < len(http_source['data'])
        await asyncio.sleep(.18)
        assert part.stat().st_size == size and not task.done()
        if stop:
            cancel.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            await asyncio.sleep(.05)
            assert part.stat().st_size == size
        else:
            pause.resume()
            await asyncio.wait_for(task, 3)
            assert (destination / 'model.gguf').read_bytes() == http_source['data']
            assert len(http_source['requests']) == 1
    finally:
        cancel.set()
        pause.resume()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('stop', [False, True])
async def test_environment_package_pause_waits_for_transfer_thread(http_source, tmp_path, stop):
    from modloader.backends.installers import _download
    from modloader.sources.control import DownloadPause

    paused = asyncio.Event()
    async def acknowledged():
        paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    target = tmp_path / 'environment.zip'
    events = []
    async def progress(event):
        events.append(event)
        if event.get('downloaded', 0) >= 1024 * 1024 and pause.generation == 0:
            pause.request()
    task = asyncio.create_task(_download(http_source['url'], target, progress, cancel, pause=pause))
    try:
        await asyncio.wait_for(paused.wait(), 3)
        size = target.stat().st_size
        assert 0 < size < len(http_source['data'])
        await asyncio.sleep(.18)
        assert target.stat().st_size == size and not task.done()
        if stop:
            cancel.set()
            pause.resume()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            await asyncio.sleep(.05)
            assert target.stat().st_size == size
        else:
            pause.resume()
            await asyncio.wait_for(task, 3)
            assert target.read_bytes() == http_source['data']
            assert len(http_source['requests']) == 1
        assert events[-1].get('can_pause') is False
    finally:
        cancel.set()
        pause.resume()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('outcome', ['resume', 'stop', 'failure'])
async def test_environment_concurrent_transfers_join_writers_and_preserve_cache(http_source, tmp_path, outcome):
    import hashlib
    from modloader.backends import catalog, installers
    from modloader.sources.control import DownloadPause
    paths, events, paused = AppPaths(tmp_path), [], asyncio.Event()
    async def acknowledged(): paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    data = http_source['data']
    digest = 'sha256:' + hashlib.sha256(data).hexdigest()
    files = [await catalog.manifest_file(paths, f'{index}.whl', f'File {index}', http_source['url'], digest, len(data)) for index in range(7)]
    cache = catalog.cache_path(paths, files[0])
    cache.parent.mkdir(parents=True)
    cache.write_bytes(data)
    if outcome == 'failure':
        files[1]['digest'] = 'sha256:' + 'a' * 64
    async def progress(event):
        events.append(event)
        active = [f for f in event.get('manifest', {}).get('files', []) if f.get('status') == 'downloading']
        if outcome != 'failure' and pause.generation == 0 and len(active) == 4 and all(f.get('downloaded_bytes', 0) for f in active):
            pause.request()
    transfer = installers._Transfers(paths, progress, cancel, pause, None)
    stage = await transfer.add(dict(id='parallel', title='Parallel', files=files))
    task = asyncio.create_task(transfer.fetch(stage))
    try:
        if outcome == 'failure':
            with pytest.raises(ValueError, match='SHA256'):
                await asyncio.wait_for(task, 5)
        else:
            await asyncio.wait_for(paused.wait(), 5)
            partials = list(paths.runtimes.rglob('*.part'))
            assert len(partials) == 4
            sizes = [p.stat().st_size for p in partials]
            await asyncio.sleep(.15)
            assert [p.stat().st_size for p in partials] == sizes
            assert not task.done() and len(http_source['requests']) == 4
            if outcome == 'stop':
                cancel.set()
                pause.resume()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            else:
                pause.resume()
                result = await asyncio.wait_for(task, 5)
                assert list(result) == [f['id'] for f in files]
                assert all(p.read_bytes() == data for p in result.values())
                assert len(http_source['requests']) == 6  # Resume keeps the existing connections.
                assert events[-1]['downloaded_bytes'] == len(data) * 6
                assert events[-1]['total_bytes'] == len(data) * 6
                assert events[-1]['completed_files'] == 7
        assert max(sum(f.get('status') == 'downloading' for f in e['manifest']['files']) for e in events) == 4
        counts = [e['downloaded_bytes'] for e in events]
        assert counts == sorted(counts)
        assert not list(paths.runtimes.rglob('*.part'))
        assert cache.read_bytes() == data
        assert events[-1]['can_pause'] is False
        if outcome == 'resume':
            assert not any(e.get('can_pause') is False and any(f.get('status') == 'downloading' for f in e['manifest']['files']) for e in events)
    finally:
        cancel.set()
        pause.resume()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_environment_pause_does_not_wait_to_fill_a_network_buffer(http_source, tmp_path):
    import hashlib
    from modloader.backends import catalog, installers
    from modloader.sources.control import DownloadPause
    # 1 KiB every 50 ms: filling even a 64 KiB buffer would take > 3 seconds.
    http_source.update(data=b'x' * (128 * 1024), chunk_size=1024, chunk_delay=.05)
    paths, paused = AppPaths(tmp_path), asyncio.Event()
    async def acknowledged(): paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    digest = 'sha256:' + hashlib.sha256(http_source['data']).hexdigest()
    files = [await catalog.manifest_file(paths, f'slow-{i}.whl', 'Slow source', http_source['url'], digest, len(http_source['data'])) for i in range(4)]
    transfer = installers._Transfers(paths, None, cancel, pause, None)
    stage = await transfer.add(dict(id='slow', title='Slow streams', files=files))
    task = asyncio.create_task(transfer.fetch(stage))
    try:
        async with asyncio.timeout(2):
            while len(http_source['requests']) < 4:
                await asyncio.sleep(.005)
        started = time.monotonic()
        pause.request()
        await asyncio.wait_for(paused.wait(), 1)
        assert time.monotonic() - started < 1
        partials = list(paths.runtimes.rglob('*.part'))
        sizes = [p.stat().st_size for p in partials]
        assert len(partials) == 4
        assert max(sizes) < 64 * 1024
        await asyncio.sleep(.15)
        assert [p.stat().st_size for p in partials] == sizes
        http_source['chunk_delay'] = .001
        pause.resume()
        result = await asyncio.wait_for(task, 4)
        assert len(http_source['requests']) == 4
        assert all(p.read_bytes() == http_source['data'] for p in result.values())
    finally:
        cancel.set()
        pause.resume()
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def sdk_transfer(service, http_source, monkeypatch):
    """Real installed Hub downloaders and worker protocol against local HTTP."""
    bootstrap = """
import sys
from types import SimpleNamespace
import huggingface_hub
from modloader.sources import worker
endpoint, size = sys.argv[1], int(sys.argv[2])
original_hf = huggingface_hub.hf_hub_download
def download(**kwargs):
    kwargs['endpoint'] = endpoint
    return original_hf(**kwargs)
huggingface_hub.hf_hub_download = download
original_ms_init = worker._ModelScope.__init__
def initialize(self, cache):
    original_ms_init(self, cache)
    self.api._legacy._endpoint = endpoint
    self.api.list_repo_files = lambda *a, **kw: [SimpleNamespace(path='model.gguf', size=size, sha256=None, is_dir=False)]
worker._ModelScope.__init__ = initialize
raise SystemExit(worker.main())
"""
    endpoint = http_source['url'].split('/model.gguf')[0]
    original = asyncio.create_subprocess_exec
    processes = []
    async def launch(*_, **kwargs):
        with (service.paths.root / 'sdk-worker-stderr.log').open('wb') as log:
            kwargs['stderr'] = log
            process = await original(sys.executable, '-c', bootstrap, endpoint, str(len(http_source['data'])), **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', launch)
    # HF metadata HEAD requires a pinned commit, just as the production plan.
    http_source['hub_commit'] = 'a' * 40
    return processes


@pytest.mark.parametrize('source', ['hf', 'modelscope'])
@pytest.mark.parametrize('stop', [False, True])
async def test_sdk_worker_pause_controls_actual_sdk_file_writes(service, http_source, sdk_transfer, source, stop, monkeypatch):
    from modloader.sources.control import DownloadPause

    paused = asyncio.Event()
    async def acknowledged():
        paused.set()
    pause, cancel = DownloadPause(acknowledged), asyncio.Event()
    destination = service.paths.models / ('paused-' + source)
    async def plan(_):
        return {'source': source, 'repo_id': 'test/model', 'revision': 'a' * 40, 'files': [{'path': 'model.gguf', 'size': len(http_source['data'])}], 'warnings': []}
    monkeypatch.setattr(service, 'plan', plan)
    capable = False
    async def progress(event):
        nonlocal capable
        capable = event.get('can_pause', capable)
        if capable and event.get('downloaded_bytes', 0) >= 64 * 1024 and pause.generation == 0:
            pause.request()
    request = DownloadRequest(source=source, repo_id='test/model', files=['model.gguf'], destination=str(destination))
    task = asyncio.create_task(service.download(request, progress, cancel, pause=pause))
    try:
        await asyncio.wait_for(paused.wait(), 8)
        def sizes():
            return {str(path.relative_to(destination)): path.stat().st_size for path in destination.rglob('*') if path.is_file()}
        before = sizes()
        assert any(0 < size < len(http_source['data']) for size in before.values())
        await asyncio.sleep(.2)
        assert sizes() == before and not task.done()
        if stop:
            cancel.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 4)
        else:
            pause.resume()
            try:
                await asyncio.wait_for(task, 6)
            except TimeoutError:
                pytest.fail((service.paths.root / 'sdk-worker-stderr.log').read_text(encoding='utf-8', errors='replace'))
            assert (destination / 'model.gguf').read_bytes() == http_source['data']
            assert len(http_source['requests']) == 1
        assert sdk_transfer[0].returncode is not None
        after = sizes()
        await asyncio.sleep(.05)
        assert sizes() == after
    finally:
        cancel.set()
        pause.resume()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_download_uses_core_assigned_model_root_without_mutating_paths(service, http_source, tmp_path):
    alternate = tmp_path / "configured-model-directory"
    scoped = SourceService(service.paths, lambda _: None, lambda: None, models_root=alternate)
    destination = alternate / "model"
    request = DownloadRequest(source="url", url=http_source["url"], destination=str(destination))
    result = await scoped.download(request, collect, asyncio.Event())
    assert Path(result["path"]) == destination
    assert (destination / "model.gguf").read_bytes() == http_source["data"]
    assert service.paths.models != alternate
    request.destination = str(service.paths.models / "not-this-jobs-root")
    with pytest.raises(SourceError) as error:
        await scoped.download(request, collect, asyncio.Event())
    assert error.value.code == "unsafe_destination"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,etag", [("ignore", '"version-one"'), ("range", None)])
async def test_http_no_range_or_validator_safely_restarts(service, http_source, mode, etag):
    http_source["etag"] = etag
    request, _, offset = await partial_download(service, http_source)
    http_source["mode"] = mode
    result = await service.download(request, collect, asyncio.Event())
    assert (Path(result["path"]) / "model.gguf").read_bytes() == http_source["data"]
    assert http_source["requests"][-1] == (f"bytes={offset}-" if etag else None)


@pytest.mark.asyncio
async def test_changed_etag_does_not_mix_resource_versions(service, http_source):
    request, _, offset = await partial_download(service, http_source)
    http_source["data"] = b"new version" * (len(http_source["data"]) // 11)
    http_source["etag"] = '"version-two"'
    result = await service.download(request, collect, asyncio.Event())
    assert http_source["requests"][-2:] == [f"bytes={offset}-", None]
    assert (Path(result["path"]) / "model.gguf").read_bytes() == http_source["data"]


@pytest.mark.asyncio
async def test_invalid_content_range_cannot_append(service, http_source):
    request, part, offset = await partial_download(service, http_source)
    http_source["mode"] = "bad_range"
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "invalid_range"
    assert part.stat().st_size == offset
    assert not part.with_name("model.gguf").exists()


@pytest.mark.asyncio
async def test_retryable_http_failure_has_a_finite_end(service, http_source):
    http_source["mode"] = "error"
    request = DownloadRequest(source="url", url=http_source["url"], destination=str(service.paths.models / "failed"))
    with pytest.raises(SourceError):
        await service.download(request, collect, asyncio.Event())
    assert len(http_source["requests"]) == NETWORK_ATTEMPTS
    assert not (service.paths.models / "failed" / "model.gguf").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../escape.gguf", "C:/escape.gguf", "folder\\escape.gguf", "CON.gguf"])
async def test_direct_path_escape_is_rejected_without_http(service, http_source, filename):
    request = DownloadRequest(source="url", url=http_source["url"], filename=filename, destination=str(service.paths.models / "safe"))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "unsafe_path"
    assert not http_source["requests"]


@pytest.mark.asyncio
async def test_destination_and_symlink_escape_are_rejected(service, http_source, tmp_path):
    request = DownloadRequest(source="url", url=http_source["url"], destination=str(tmp_path / "outside"))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "unsafe_destination"
    destination = service.paths.models / "safe"
    destination.mkdir()
    outside = tmp_path / "untouched.gguf"
    outside.write_bytes(b"original")
    try:
        (destination / "model.gguf").symlink_to(outside)
    except OSError:
        pytest.skip("This Windows environment does not permit symlink creation")
    request.destination = str(destination)
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "unsafe_path"
    assert outside.read_bytes() == b"original"
    assert not http_source["requests"]


@pytest.mark.asyncio
async def test_directory_junction_cannot_write_outside_download(service, tmp_path, monkeypatch):
    destination = service.paths.models / "junction"
    destination.mkdir()
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "model.gguf").write_bytes(b"original")
    link = destination / "escape"
    if os.name == "nt":
        environment = {**os.environ, "MODLOADER_TEST_LINK": str(link), "MODLOADER_TEST_TARGET": str(outside)}
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", "New-Item -ItemType Junction -Path $env:MODLOADER_TEST_LINK -Target $env:MODLOADER_TEST_TARGET -ErrorAction Stop | Out-Null"], check=True, env=environment, capture_output=True)
    else:
        link.symlink_to(outside, target_is_directory=True)

    async def plan(_):
        return {"source": "hf", "repo_id": "test/model", "revision": "commit", "files": [{"path": "escape/model.gguf", "size": 8}], "warnings": []}

    monkeypatch.setattr(service, "plan", plan)
    request = DownloadRequest(source="hf", repo_id="test/model", files=["escape/model.gguf"], destination=str(destination))
    with pytest.raises(SourceError) as error:
        await service.download(request, collect, asyncio.Event())
    assert error.value.code == "unsafe_path"
    assert (outside / "model.gguf").read_bytes() == b"original"


@pytest.mark.asyncio
async def test_sdk_worker_cancel_terminates_the_real_process(service, tmp_path, monkeypatch):
    import modloader.sources as sources

    original = asyncio.create_subprocess_exec
    processes = []
    captured = []
    marker = tmp_path / "worker-bytes"
    script = "import sys,time,json; f=open(sys.argv[1],'wb');\nfor n in range(1000):\n f.write(b'x'*1024); f.flush(); print(json.dumps({'type':'progress','data':{'downloaded_bytes':n+1}}),flush=True); time.sleep(.03)"

    async def launch(*argv, **kwargs):
        captured.append((argv, kwargs["env"]))
        process = await original(sys.executable, "-u", "-c", script, str(marker), **kwargs)
        processes.append(process)
        return process

    async def plan(_):
        return {"source": "hf", "repo_id": "test/model", "revision": "commit", "files": [{"path": "model.gguf", "size": None}], "warnings": []}

    monkeypatch.setattr(sources.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(service, "plan", plan)
    monkeypatch.setattr(service, "credential_provider", lambda _: "private-test-token")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    cancel = asyncio.Event()

    async def progress(event):
        if event.get("downloaded_bytes", 0) >= 2:
            cancel.set()

    request = DownloadRequest(source="hf", repo_id="test/model", files=["model.gguf"], destination=str(service.paths.models / "worker"))
    with pytest.raises(asyncio.CancelledError):
        await service.download(request, progress, cancel)
    assert processes[0].returncode is not None
    size = marker.stat().st_size
    await asyncio.sleep(0.12)
    assert marker.stat().st_size == size
    assert captured[0][0] == (sys.executable, "--source-worker")
    assert all("private-test-token" not in value for value in captured[0][0])
    assert captured[0][1]["MODLOADER_SOURCE_TOKEN"] == "private-test-token"
    assert Path(captured[0][1]["HF_HOME"]).is_relative_to(service.paths.root)


def test_offline_xet_transfer_and_reconstruction_do_not_overwrite_each_other(service, monkeypatch):
    from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter
    from modloader.sources import worker

    events = []
    monkeypatch.setattr(worker, "emit", lambda kind, data=None, **_: events.append(data) if kind == "progress" else None)

    class FixtureHF(worker._HuggingFace):
        def __init__(self):
            pass

        def get_file(self, repo_id, revision, path, local=None, tqdm_class=None):
            with XetDownloadProgressReporter(
                reconstruction_desc="fixture: reconstructing file", transfer_desc="fixture: downloading bytes",
                total=100, log_level=30, tqdm_class=tqdm_class,
            ) as reporter:
                # Use the actual installed SDK adapter. A single bar must
                # receive both event streams, even if transfer exceeds the
                # declared file size because of encoding or network retries.
                assert reporter.reconstruction_bar is reporter.transfer_bar
                for reconstructed, transferred in [(0, 80), (25, 140), (100, 190)]:
                    reporter.update_progress(SimpleNamespace(
                        total_bytes_completed=reconstructed,
                        total_transfer_bytes_completed=transferred,
                        total_bytes_completion_rate=0,
                        total_transfer_bytes_completion_rate=0,
                        total_bytes=100,
                    ))
                    time.sleep(0.16)
            target = local / path
            target.write_bytes(b"x" * 100)
            return target

    result = worker._download(FixtureHF(), {
        "source": "hf", "repo_id": "fixture/model", "revision": "fixture",
        "models_root": str(service.paths.models), "destination": str(service.paths.models / "offline-xet"),
        "files": [{"path": "model.safetensors", "size": 100}],
    })
    counts = [event["downloaded_bytes"] for event in events]
    assert counts == sorted(counts)
    assert all(count <= 100 for count in counts)
    transfer_events = [event for event in events if event.get("file_transferred_bytes") is not None]
    assert transfer_events[0]["file_transferred_bytes"] == 80
    assert transfer_events[0]["file_downloaded_bytes"] == 0
    assert transfer_events[0]["message"] != "等待开始"
    assert transfer_events[-1]["file_transferred_bytes"] == 190
    assert any(event["file_transferred_bytes"] > event["file_downloaded_bytes"] for event in transfer_events)
    assert result["size_bytes"] == 100


def test_offline_hf_http_progress_keeps_resume_file_bytes(service, monkeypatch):
    from huggingface_hub.file_download import _get_progress_bar_context
    from modloader.sources import worker

    events = []
    monkeypatch.setattr(worker, "emit", lambda kind, data=None, **_: events.append(data) if kind == "progress" else None)

    class FixtureHF(worker._HuggingFace):
        def __init__(self):
            pass

        def get_file(self, repo_id, revision, path, local=None, tqdm_class=None):
            with _get_progress_bar_context(desc="fixture", log_level=30, total=100, initial=20, tqdm_class=tqdm_class) as bar:
                bar.update(30)
                assert bar.n == 50
                bar.update(50)
                assert bar.n == 100
            target = local / path
            target.write_bytes(b"x" * 100)
            return target

    result = worker._download(FixtureHF(), {
        "source": "hf", "repo_id": "fixture/model", "revision": "fixture",
        "models_root": str(service.paths.models), "destination": str(service.paths.models / "offline-http-progress"),
        "files": [{"path": "model.safetensors", "size": 100}],
    })
    counts = [event["downloaded_bytes"] for event in events]
    assert counts == sorted(counts)
    assert 20 in counts and 50 in counts and counts[-1] == 100
    assert all(event.get("file_transferred_bytes") is None for event in events)
    assert result["size_bytes"] == 100
