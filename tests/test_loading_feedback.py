"""Model loading feedback uses scrubbed startup output, never inference output."""
import asyncio
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

from modloader.models import RuntimeInstance
from modloader.runtime import RuntimeManager, loading_summary, scrub


def instance():
    return RuntimeInstance(id='loading-fixture', model_id='model', profile_id='profile', profile_revision=1,
                           alias='fixture', backend='fixture', runtime_id='runtime', port=1234,
                           started_at=datetime.now(timezone.utc).isoformat(), state='starting', phase='loading')


async def test_stdout_updates_loading_summary_and_stops_at_ready():
    manager = RuntimeManager(None, {})
    model = instance()
    manager.logs[model.id] = deque(maxlen=1000)
    stream = asyncio.StreamReader()
    reader = asyncio.create_task(manager._read_output(model, SimpleNamespace(stdout=stream)))
    stream.feed_data(b'\x1b[32mload tensors 12/64\x1b[0m Authorization: Bearer fixture-secret\n')
    await asyncio.sleep(0)
    assert model.message == 'load tensors 12/64 Authorization: [REDACTED]'
    assert 'fixture-secret' not in manager.logs[model.id][0]
    assert model.phase == 'loading' and model.state == 'starting'

    model.state, model.phase, model.message = 'ready', 'ready', '模型已就绪'
    stream.feed_data(b'private prompt and inference output\n')
    stream.feed_data(b'MODLOADER_GUEST_PID=4321\n')
    stream.feed_eof()
    await reader
    assert model.message == '模型已就绪' and len(manager.logs[model.id]) == 1
    assert model.guest_pid == 4321


async def test_launch_failure_has_terminal_phase_and_scrubbed_message(tmp_path, monkeypatch):
    from modloader import runtime
    manager = RuntimeManager(SimpleNamespace(work=tmp_path, root=tmp_path), {})
    model = instance()
    manager.logs[model.id] = deque(maxlen=1000)
    def fail(*args):
        raise OSError('launch failed api_key=fixture-secret')
    monkeypatch.setattr(runtime, 'prepare_launch', fail)

    await manager._run(model, SimpleNamespace())
    assert model.state == 'failed' and model.phase == 'failed'
    assert model.message == model.error == 'launch failed api_key=[REDACTED]'
    assert list(manager.logs[model.id]) == [model.error]


def test_compact_readout_preserves_real_counts_without_making_a_percentage():
    assert loading_summary('  loading shard 2/7  ') == 'loading shard 2/7'
    assert loading_summary('loading model') == 'loading model'
    assert len(loading_summary('tensor ' + 'x' * 1000)) <= 240
    assert scrub('Authorization: Basic fixture-secret') == 'Authorization: [REDACTED]'
    assert 'fixture-secret' not in scrub("{'Authorization': 'Bearer fixture-secret'}")
    assert 'with spaces' not in scrub('api_key="fixture-secret with spaces"')
