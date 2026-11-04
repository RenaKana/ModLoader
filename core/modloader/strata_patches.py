"""Apply the reviewed Control 0.2.10 continuation patch to a new install only."""
from pathlib import Path
import re
import sys

CONTROL_REF = 'ffa80d73133ea5c548aefe57889a16e8085a938f'
FILES = {'serve/server.py', 'serve/raw_continuation.py', 'serve/test_raw_token_continuation.py'}


def apply_server_patch(checkout, source_ref):
    """Exact pinned source/context; prepare all outputs before any write.

    Returns False for other releases, True for this reviewed patch (including
    an already applied copy). Called after official archive verification.
    """
    if source_ref != CONTROL_REF:
        return False
    root = Path(checkout).resolve()
    bundle = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
    patch = (bundle / 'patches/strata-token-continuation/server-0.2.10.patch').read_text('utf-8').splitlines(keepends=True)
    sections = []
    for line in patch:
        if line.startswith('--- '):
            sections.append([line])
        else:
            if not sections:
                raise ValueError('Strata 服务补丁格式无效')
            sections[-1].append(line)
    outputs, originals = {}, {}
    for section in sections:
        if not section[1].startswith('+++ b/'):
            raise ValueError('Strata 服务补丁目标无效')
        name = section[1][6:].strip()
        if name not in FILES:
            raise ValueError('Strata 服务补丁包含未授权路径')
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Strata 服务补丁路径越界')
        original = path.read_text('utf-8') if path.exists() else None
        originals[path] = original
        lines = (original or '').splitlines(keepends=True)
        hunks = []
        for line in section[2:]:
            if line.startswith('@@ '):
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if not match:
                    raise ValueError('Strata 服务补丁 hunk 无效')
                hunks.append((int(match[1]), int(match[2]), [], []))
            elif hunks and line[:1] in (' ', '+', '-'):
                if line[0] != '+':
                    hunks[-1][2].append(line[1:])
                if line[0] != '-':
                    hunks[-1][3].append(line[1:])
            else:
                raise ValueError('Strata 服务补丁内容无效')
        if not hunks:
            raise ValueError('Strata 服务补丁为空')
        if original is not None and all(lines[max(0, new - 1):max(0, new - 1) + len(after)] == after
                                        for old, new, before, after in hunks):
            continue
        result, cursor = [], 0
        for old, new, before, after in hunks:
            start = max(0, old - 1)
            if start < cursor or lines[start:start + len(before)] != before:
                raise ValueError(f'Strata {name} 与已核对的 0.2.10 服务补丁上下文不符')
            result += lines[cursor:start] + after
            cursor = start + len(before)
        if section[0].startswith('--- /dev/null') and original is not None:
            raise ValueError(f'Strata {name} 已存在不同内容')
        outputs[path] = ''.join(result + lines[cursor:])
    if set(path.relative_to(root).as_posix() for path in originals) != FILES:
        raise ValueError('Strata 服务补丁文件集合不完整')
    written = []
    try:
        for path, content in outputs.items():
            path.write_text(content, encoding='utf-8', newline='\n')
            written.append(path)
    except OSError:
        for path in reversed(written):
            if originals[path] is None:
                path.unlink()
            else:
                path.write_text(originals[path], encoding='utf-8', newline='\n')
        raise
    return True
