from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Awaitable, Callable

from ..models import DownloadRequest, SourceRepository
from ._common import SourceError, relative_path


_SHARD = re.compile(r"^(?P<base>.+)-(?P<number>\d+)-of-(?P<total>\d+)(?P<ext>\.(?:gguf|safetensors))$", re.I)
_WEIGHT_SUFFIXES = {".gguf", ".safetensors"}
_INDEX_SUFFIXES = (".safetensors.index.json",)
_VISION_DIRECTORIES = {"vision_tower", "vision_model", "vision", "mm_projector", "projector"}
_ANCILLARY = {
    "config.json", "configuration.json", "generation_config.json", "adapter_config.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "tokenizer.bpe",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "image_processor_config.json",
    "chat_template.jinja", "tekken.json",
}
_DEEPSEEK41_ISTA = re.compile(r"DeepSeek-V4\.1-Flash-1017-GSQ-RCO-(?:IQ1_M|IQ2_XXS|IQ2_XS)\.gguf", re.I)
_DEEPSEEK41_ENGRAM = "deepseek-v4.1-flash-engram-iq4_nl.gguf"
_DEEPSEEK41_LEGACY_ENGRAM = "engram-iq4_nl.gguf"
_DEEPSEEK41_DSPARK = "dspark-deepseek-v4.1-flash-1017-q8_0.gguf"


def _is_accessory(path: str) -> bool:
    parsed = PurePosixPath(path)
    parents = tuple(part.lower() for part in parsed.parts[:-1])
    # Complete GGUF names can describe embedded MTP (e.g. -MTP-HIGH).
    # Only standalone MTP files and known accessory directories are excluded.
    return bool(re.match(r"^mtp(?:[_.-]|$)", parsed.name, re.I)) or "mtp" in parents or any(
        parents[index:index + 2] == ("strata", "rt") for index in range(len(parents) - 1)
    )


def _is_projector(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return "mmproj" in name or name.startswith(("mm_projector", "projector"))


def _is_vision(path: str) -> bool:
    return _is_projector(path) or bool(set(part.lower() for part in PurePosixPath(path).parts[:-1]) & _VISION_DIRECTORIES)


def _is_weight(path: str) -> bool:
    parsed = PurePosixPath(path)
    return parsed.suffix.lower() in _WEIGHT_SUFFIXES and parsed.name.lower() != "tokenizer.model"


def _family(path: str) -> str:
    parsed = PurePosixPath(path)
    match = _SHARD.fullmatch(parsed.name)
    return (parsed.parent / (match["base"] + match["ext"])).as_posix() if match else path


def _is_deepseek41_ista(path: str) -> bool:
    return bool(_DEEPSEEK41_ISTA.fullmatch(PurePosixPath(_family(path)).name))


def _deepseek41_accessory(path: str, *, legacy: bool = False) -> str | None:
    name = PurePosixPath(path).name.casefold()
    if name == _DEEPSEEK41_ENGRAM or (legacy and name == _DEEPSEEK41_LEGACY_ENGRAM):
        return "engram"
    if name == _DEEPSEEK41_DSPARK:
        return "dspark"
    return None


def _quantization(path: str) -> str | None:
    match = re.search(r"(?:^|[-_.])(IQ\d[\w]*|Q\d[\w]*|BF16|F16|F32)(?=[-_.]|$)", PurePosixPath(path).stem, re.I)
    return match[1].upper() if match else None


def _ancillary(path: str) -> bool:
    parsed = PurePosixPath(path)
    return parsed.name.lower() in _ANCILLARY or parsed.name.lower().startswith("tokenizer.") or parsed.suffix.lower() == ".tiktoken"


def _dependencies(directory: PurePosixPath, available: dict) -> set[str]:
    # Configuration/tokenizer belongs to this HF model directory. A nearby
    # model, or a GGUF variant in another folder, is not dependency evidence.
    return {
        path for path in available
        if (PurePosixPath(path).parent == directory and _ancillary(path))
        or (PurePosixPath(path).parent.name == "chat_templates" and PurePosixPath(path).parent.parent == directory and path.lower().endswith(".jinja"))
    }


async def classify_repository(repository: SourceRepository, read_json: Callable[[str], Awaitable[dict]],
                              *, progress: Callable[[dict], None] | None = None) -> SourceRepository:
    """The shared catalog for both the selection UI and download completion."""
    available = {file.path: file for file in repository.files}
    groups: list[dict] = []
    has_deepseek = any(_is_deepseek41_ista(path) for path in available)
    accessories = {path: kind for path in available if (kind := _deepseek41_accessory(path, legacy=has_deepseek))}
    roles: dict[str, tuple[str, str | None]] = {
        path: ("required" if kind == "engram" else "other", None) for path, kind in accessories.items()
    }
    errors: dict[str, dict] = {}
    indexed: set[str] = set()

    def add_group(identity: str, name: str, format: str, paths: set[str], weights: set[str], kind: str, error: SourceError | None = None) -> None:
        files = sorted(paths)
        sizes = [available[path].size for path in files]
        groups.append({
            "id": identity, "name": name, "format": format, "kind": kind, "files": files,
            "size": sum(size or 0 for size in sizes) if all(size is not None for size in sizes) else None,
            "quantization": _quantization(name),
        })
        if error:
            errors[identity] = {"code": error.code, "message": str(error)}
        for path in files:
            if kind == "vision":
                role = "vision"
            elif path in weights:
                role = "shard" if _SHARD.fullmatch(PurePosixPath(path).name) or len(weights) > 1 else "model"
            else:
                role = "required"
            if path not in roles or path in weights:
                roles[path] = (role, identity)

    indexes = [path for path in sorted(available) if path.lower().endswith(_INDEX_SUFFIXES) and not _is_accessory(path)]
    for number, path in enumerate(indexes):
        if progress:
            progress({"stage": "weight_indexes", "message": f"检查权重索引 · {path}", "completed": number, "total": len(indexes)})
        parsed = PurePosixPath(path)
        weights: set[str] = set()
        error = None
        try:
            document = await read_json(path)
            mapping = document.get("weight_map") if isinstance(document, dict) else None
            if not isinstance(mapping, dict) or not mapping or not all(isinstance(value, str) for value in mapping.values()):
                raise SourceError(f"权重索引没有有效的 weight_map：{path}", "invalid_index")
            for value in mapping.values():
                value = relative_path(value)
                local = relative_path((parsed.parent / value).as_posix())
                candidate = local if local in available else value
                if candidate not in available:
                    raise SourceError(f"权重索引引用了不存在的分片：{candidate}", "missing_shard")
                if not _is_weight(candidate) or _is_accessory(candidate) or candidate in accessories:
                    raise SourceError(f"权重索引引用了不支持的模型文件：{candidate}", "invalid_index")
                weights.add(candidate)
        except SourceError as caught:
            error = caught
        indexed.update(weights)
        kind = "vision" if _is_vision(path) else "model"
        paths = weights | {path} | _dependencies(parsed.parent, available)
        name = parsed.parent.as_posix() if parsed.parent != PurePosixPath(".") else repository.name
        # Different indexes in the same directory remain separate choices.
        name += " / " + parsed.name.removesuffix(".index.json")
        add_group("hf:" + path, name, "hf", paths, weights, kind, error)
        if progress:
            progress({"stage": "weight_indexes", "message": f"检查权重索引 · {path}", "completed": number + 1, "total": len(indexes)})

    families: dict[tuple[str, str], set[str]] = {}
    for path in sorted(available):
        if path in indexed or not _is_weight(path) or _is_accessory(path) or path in accessories:
            continue
        parsed = PurePosixPath(path)
        kind = "vision" if _is_vision(path) else "model"
        if parsed.suffix.lower() == ".gguf":
            identity = "gguf:" + _family(path)
        elif any((parsed.parent / name).as_posix() in available for name in ("config.json", "adapter_config.json")):
            identity = "hf:" + parsed.parent.as_posix() + ":" + parsed.suffix.lower()
        else:
            identity = "hf:" + _family(path)
        families.setdefault((identity, kind), set()).add(path)
    for (identity, kind), weights in families.items():
        first = sorted(weights)[0]
        parsed = PurePosixPath(first)
        gguf = parsed.suffix.lower() == ".gguf"
        paths = weights if gguf else weights | _dependencies(parsed.parent, available)
        name = PurePosixPath(_family(first)).name if gguf or len(weights) == 1 else (parsed.parent.as_posix() if parsed.parent != PurePosixPath(".") else repository.name)
        add_group(identity, name, "gguf" if gguf else "hf", paths, weights, kind)

    files = [file.model_copy(update={
        "role": roles.get(file.path, ("other", None))[0],
        "group_id": roles.get(file.path, ("other", None))[1],
        "reason": ("DeepSeek V4.1 必需的共享 Engram" if accessories[file.path] == "engram" else "可选 DSpark 草稿") if file.path in accessories else None,
    }) for file in repository.files]
    return repository.model_copy(update={"groups": groups, "files": files, "metadata": {**repository.metadata, "group_errors": errors}})


def _validate_shards(paths: set[str], available: dict) -> None:
    for path in paths:
        match = _SHARD.fullmatch(PurePosixPath(path).name)
        if not match:
            continue
        total = int(match["total"])
        members: dict[int, str] = {}
        for candidate in available:
            other = _SHARD.fullmatch(PurePosixPath(candidate).name)
            if other and _family(candidate).casefold() == _family(path).casefold() and int(other["total"]) == total:
                number = int(other["number"])
                if number in members:
                    raise SourceError("同一权重组出现重复编号的分片。", "invalid_shards")
                members[number] = candidate
        if total <= 0 or len(members) != total or any(number < 1 or number > total for number in members):
            raise SourceError(f"所选权重的分片不完整：{path}", "missing_shard")


def _vision_candidates(model: dict, groups: list[dict]) -> list[dict]:
    weights = [path for path in model["files"] if _is_weight(path)]
    directory = PurePosixPath(weights[0]).parent if weights else PurePosixPath(".")
    candidates = []
    for group in groups:
        if group["kind"] != "vision":
            continue
        if model["format"] == "gguf" and group["format"] != "gguf":
            continue
        vision_weights = [path for path in group["files"] if _is_weight(path)]
        if not vision_weights:
            continue
        vision_path = PurePosixPath(vision_weights[0])
        if vision_path.parent in {directory, PurePosixPath(".")} or vision_path.parent.parent == directory:
            candidates.append(group)
    local = [group for group in candidates if any(PurePosixPath(path).parent == directory for path in group["files"] if _is_weight(path))]
    return local or candidates


def _required_vision_choices(config_path: str, config: dict, groups: list[dict], available: dict, warnings: list[str]) -> list[list[dict]]:
    directory = PurePosixPath(config_path).parent
    targets: set[str] = set()
    references = [config[key] for key in ("vision_tower", "mm_vision_tower") if isinstance(config.get(key), str) and config[key]]
    for value in references:
        try:
            reference = relative_path(value.removeprefix("./").rstrip("/"))
            local = (directory / reference).as_posix()
            target = next((candidate for candidate in (local, reference) if any(path == candidate or path.startswith(candidate + "/") for path in available)), None)
        except SourceError:
            target = None
        if target:
            targets.add(target)
        else:
            warnings.append("模型配置引用了外部视觉编码器，当前仓库下载不能保证包含该编码器。")
    if not references and "vision_config" in config:
        targets.update((directory / name).as_posix() for name in _VISION_DIRECTORIES)
    choices = []
    for target in sorted(targets):
        matches = [group for group in groups if any(
            _is_weight(path) and (path == target or PurePosixPath(path).parent.as_posix() == target)
            for path in group["files"]
        )]
        if matches:
            choices.append(matches)
    return choices


async def build_plan(request: DownloadRequest, repository: SourceRepository, read_json: Callable[[str], Awaitable[dict]],
                     *, progress: Callable[[dict], None] | None = None) -> dict:
    if not repository.groups:
        repository = await classify_repository(repository, read_json, progress=progress)
    available = {file.path: file for file in repository.files}
    explicit = {relative_path(path) for path in request.files}
    if not explicit:
        raise SourceError("请先选择要下载的模型或文件。", "selection_required")
    missing = sorted(explicit - available.keys())
    if missing:
        raise SourceError(f"仓库中没有所选文件：{', '.join(missing)}", "file_not_found")
    selected = set(explicit)
    warnings = list(repository.metadata.get("warnings", []))
    groups = repository.groups

    def selected_group(group: dict) -> bool:
        # Shared tokenizer/configuration does not select every model using it.
        return any(path in explicit and (_is_weight(path) or path.lower().endswith(_INDEX_SUFFIXES)) for path in group["files"])

    models = [group for group in groups if group["kind"] == "model" and selected_group(group)]
    visions = [group for group in groups if group["kind"] == "vision" and selected_group(group)]
    reasons = {path: "手动选择" for path in explicit}

    def include(group: dict, reason: str) -> None:
        error = repository.metadata.get("group_errors", {}).get(group["id"])
        if error:
            raise SourceError(error["message"], error["code"])
        _validate_shards(set(group["files"]), available)
        for path in group["files"]:
            selected.add(path)
            if path not in explicit:
                reasons[path] = reason if group["kind"] == "vision" else "同一模型的必要分片" if _is_weight(path) else "权重索引" if path.lower().endswith(_INDEX_SUFFIXES) else reason

    for group in models:
        include(group, "模型配置与 tokenizer 等必要文件")
    deepseek = any(_is_deepseek41_ista(path) for group in models for path in group["files"])
    drafts: set[str] = set()
    if deepseek:
        engrams = {path for path in available if _deepseek41_accessory(path, legacy=True) == "engram"}
        chosen = engrams & explicit
        if not chosen:
            preferred = {path for path in engrams if PurePosixPath(path).name.casefold() == _DEEPSEEK41_ENGRAM} or engrams
            if not preferred:
                raise SourceError("当前仓库缺少 DeepSeek V4.1 必需的 Engram-IQ4_NL.gguf。", "missing_dependency")
            if len(preferred) > 1:
                raise SourceError("仓库包含多个 DeepSeek V4.1 Engram 文件，请手动选择所需的一项后重新检查下载。", "dependency_selection_required")
            chosen = preferred
        # Keep shared GGUF dependencies outside model groups: selecting one must
        # not select every quantization, and registration must start at a shard.
        for path in chosen:
            selected.add(path)
            if path not in explicit:
                reasons[path] = "DeepSeek V4.1 必需的共享 Engram"
        drafts = {path for path in available if _deepseek41_accessory(path) == "dspark"}
    for group in visions:
        include(group, "所选视觉组件的必要文件")
    candidates: dict[str, dict] = {}
    required_visions: dict[str, dict] = {}
    assignments: dict[str, list[str]] = {}
    requires_vision_selection = False
    required_models: set[str] = set()

    # Separate HF towers can be required to load even a text-only request.
    # This is a configuration dependency, unlike an optional GGUF projector.
    config_models: dict[str, list[dict]] = {}
    for model in models:
        if model["format"] == "hf":
            for path in model["files"]:
                if PurePosixPath(path).name in {"config.json", "adapter_config.json"}:
                    config_models.setdefault(path, []).append(model)
    for number, (path, related_models) in enumerate(config_models.items()):
        if progress:
            progress({"stage": "model_config", "message": f"检查模型配置 · {path}", "completed": number, "total": len(config_models)})
        config = await read_json(path)
        if not isinstance(config, dict):
            raise SourceError(f"配置必须是 JSON 对象：{path}", "invalid_index")
        if config.get("auto_map"):
            warnings.append("此模型包含自定义加载代码；运行时启用远程代码前应检查模型来源。")
        if config.get("base_model_name_or_path"):
            warnings.append("此文件是适配器，加载时还需要配置指定的基础模型。")
        if any(isinstance(config.get(key), str) and config[key] for key in ("vision_tower", "mm_vision_tower")):
            required_models.update(model["id"] for model in related_models)
        for options in _required_vision_choices(path, config, groups, available, warnings):
            options = [group for group in options if group not in related_models]
            if not options:
                continue
            required_models.update(model["id"] for model in related_models)
            vision_options = [{**group, "kind": "vision"} for group in options]
            candidates.update((group["id"], group) for group in vision_options)
            option_ids = {group["id"] for group in options}
            groups = [next((vision for vision in vision_options if vision["id"] == group["id"]), group) for group in groups]
            models = [model for model in models if model["id"] not in option_ids]
            for group in vision_options:
                for member in group["files"]:
                    available[member] = available[member].model_copy(update={"role": "vision", "group_id": group["id"]})
            chosen = [group for group in vision_options if selected_group(group)]
            if not chosen and len(vision_options) == 1:
                chosen = vision_options
            if len(chosen) != 1:
                requires_vision_selection = True
                warnings.append("HF配置要求视觉组件；此目录有多个权重方案，请明确选择匹配的一项后重新检查下载。")
            for group in chosen:
                include(group, "HF配置要求的视觉组件与配置")
                required_visions[group["id"]] = group
                if not any(vision["id"] == group["id"] for vision in visions):
                    visions.append(group)
            for model in related_models:
                assignments[model["id"]] = list(dict.fromkeys([*assignments.get(model["id"], []), *(group["id"] for group in chosen)]))
        if progress:
            progress({"stage": "model_config", "message": f"检查模型配置 · {path}", "completed": number + 1, "total": len(config_models)})

    for model in models:
        choices = _vision_candidates(model, groups)
        candidates.update((group["id"], group) for group in choices)
        chosen = [group for group in choices if any(vision["id"] == group["id"] for vision in visions)]
        if request.include_vision and not chosen and model["id"] not in required_models:
            if len(choices) == 1:
                chosen = choices
                for group in chosen:
                    include(group, "已启用视觉：匹配的视觉组件")
                    if group not in visions:
                        visions.append(group)
            elif len(choices) > 1:
                requires_vision_selection = True
                warnings.append("有多个视觉组件，请先选择与模型匹配的一项，再检查下载。")
            else:
                warnings.append(f"{model['name']} 未找到可自动补齐的视觉组件，请查看模型说明。")
        assignments[model["id"]] = list(dict.fromkeys([*assignments.get(model["id"], []), *(group["id"] for group in chosen)]))

    files = [{**available[path].model_dump(), "reason": reasons[path]} for path in sorted(selected)]
    optional = ({path for group in candidates.values() for path in group["files"]} | drafts) - selected
    return {
        "source": repository.source, "repo_id": repository.repo_id, "revision": repository.revision,
        "files": files, "total_bytes": sum(file["size"] or 0 for file in files),
        "unknown_size_files": [file["path"] for file in files if file["size"] is None],
        "warnings": list(dict.fromkeys(warnings)), "groups": groups,
        "automatic_files": [file for file in files if file["path"] not in explicit],
        "optional_files": [{**available[path].model_dump(), "reason": "可选 DSpark 草稿" if path in drafts else "可选视觉组件"} for path in sorted(optional)],
        "model_groups": models, "vision_groups": visions, "required_vision_groups": list(required_visions.values()), "vision_candidates": list(candidates.values()),
        "vision_assignments": assignments, "requires_vision_selection": requires_vision_selection,
        "include_vision": request.include_vision, "url": repository.url,
    }

