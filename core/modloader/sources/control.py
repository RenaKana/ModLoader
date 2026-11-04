"""Cooperative pause points shared by a job's download stages."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Awaitable, Callable


PAUSE_CHUNK_SIZE = 64 * 1024


class DownloadPause:
    def __init__(self, paused: Callable[[], Awaitable[None]] | None = None):
        self.requested = False
        self.generation = 0
        self.changed = asyncio.Event()
        self._gate = asyncio.Event()
        self._gate.set()
        self._paused = paused
        self.acknowledged = False
        self._participants: dict[object, int | None] = {}

    def request(self):
        self.generation += 1
        self.requested = True
        self.acknowledged = False
        self._gate.clear()
        self.changed.set()

    def resume(self):
        self.generation += 1
        self.requested = False
        self.acknowledged = False
        self._gate.set()
        self.changed.set()

    async def acknowledge(self, generation: int):
        # A late worker acknowledgement must not revive a previous pause.
        if self.requested and generation == self.generation and not self.acknowledged and all(value == generation for value in self._participants.values()):
            self.acknowledged = True
            if self._paused:
                await self._paused()

    async def checkpoint(self, cancel: asyncio.Event):
        await self._checkpoint(cancel)

    async def _checkpoint(self, cancel: asyncio.Event, participant=None):
        if cancel.is_set():
            raise asyncio.CancelledError
        try:
            while self.requested:
                if participant is not None:
                    self._participants[participant] = self.generation
                await self.acknowledge(self.generation)
                await self._gate.wait()
                if cancel.is_set():
                    raise asyncio.CancelledError
        finally:
            if participant in self._participants:
                self._participants[participant] = None

    @asynccontextmanager
    async def participant(self):
        token = object()
        self._participants[token] = None
        try:
            yield _PauseParticipant(self, token)
        finally:
            self._participants.pop(token, None)
            await self.acknowledge(self.generation)


class _PauseParticipant:
    def __init__(self, parent, token):
        self.parent, self.token = parent, token

    async def checkpoint(self, cancel):
        await self.parent._checkpoint(cancel, self.token)

    def resume(self):
        self.parent.resume()
