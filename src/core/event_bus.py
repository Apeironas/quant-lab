"""
Olay veri yolu (Event Bus) - modülleri birbirine bağlayan tek nokta.

asyncio.Queue üzerine kurulu basit bir yayınla/abone ol (pub/sub) mekanizması.
Modüller birbirini çağırmaz; herkes bus'a olay bırakır, ilgilenen dinler.
Yeni bir modül eklemek (örn. Telegram bildirimi) = sadece subscribe etmek.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from .events import Event, EventType

logger = logging.getLogger("event_bus")

# Bir olay işleyici: async fonksiyon, olayı alır, bir şey döndürmez
Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Bir olay tipine abone ol. main.py'de sistem kurulurken çağrılır."""
        self._handlers[event_type].append(handler)
        logger.debug("Abonelik: %s -> %s", event_type, getattr(handler, "__qualname__", handler))

    async def publish(self, event: Event) -> None:
        """Olayı kuyruğa bırak (bloklamaz)."""
        await self._queue.put(event)

    async def run(self) -> None:
        """
        Ana dağıtım döngüsü. Kuyruktan olayları sırayla çeker ve abonelere
        SIRAYLA await eder. Bu bilinçli bir tasarım: aynı olaya bağlı
        işleyiciler arasında yarış durumu (race condition) oluşmaz.
        """
        self._running = True
        logger.info("EventBus başladı")
        while self._running:
            event = await self._queue.get()
            for handler in self._handlers.get(event.type, []):
                try:
                    await handler(event)
                except Exception:
                    # Tek bir işleyicinin hatası tüm botu düşürmesin;
                    # hatayı tam iziyle logla ve devam et.
                    logger.exception("Handler hatası: %s olayında", event.type)
            self._queue.task_done()

    async def drain(self) -> None:
        """
        Kuyruk tamamen boşalana kadar tüm olayları işle (backtest için).
        Bir mum yayınlanır, drain çağrılır: mum -> sinyal -> emir -> fill
        zinciri o mum içinde deterministik olarak tamamlanır. Canlıda
        run() kullanılır; drain sadece backtest sürücüsü içindir.
        """
        while not self._queue.empty():
            event = self._queue.get_nowait()
            for handler in self._handlers.get(event.type, []):
                try:
                    await handler(event)
                except Exception:
                    logger.exception("Handler hatası: %s olayında", event.type)
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False
