"""
PİYASA VERİSİ MODÜLÜ

Görevi tek ve net: borsadan veri çek, MarketEvent olarak bus'a bırak.
Strateji veya emirlerle HİÇBİR ilgisi yoktur.

İki çalışma yolu:
  - WebSocket (ccxt.pro watch_ohlcv): düşük gecikme, tercih edilen yol
  - REST polling (fetch_ohlcv):       websocket sorun çıkarırsa yedek yol

Ayrıca başlangıçta `history_bars` kadar geçmiş mum çekip stratejinin
indikatör tamponunu "ısıtır" (SMA-50 için en az 50 mum gerekir vb.).
"""
from __future__ import annotations

import asyncio
import logging

import ccxt.pro as ccxtpro

from ..core.event_bus import EventBus
from ..core.events import MarketEvent

logger = logging.getLogger("data_fetcher")


def _candle_dict(ohlcv: list) -> dict:
    """ccxt OHLCV listesini [ts, o, h, l, c, v] okunur sözlüğe çevir."""
    return {
        "ts": ohlcv[0],
        "open": ohlcv[1],
        "high": ohlcv[2],
        "low": ohlcv[3],
        "close": ohlcv[4],
        "volume": ohlcv[5],
    }


class DataFetcher:
    def __init__(self, config: dict, bus: EventBus) -> None:
        self.bus = bus
        md = config["market_data"]
        self.symbols: list[str] = md["symbols"]
        self.timeframe: str = md["timeframe"]
        self.history_bars: int = md["history_bars"]
        self.use_websocket: bool = md["use_websocket"]

        ex_cfg = config["exchange"]
        exchange_class = getattr(ccxtpro, ex_cfg["id"])
        self.exchange = exchange_class({
            "apiKey": ex_cfg["api_key"],
            "secret": ex_cfg["api_secret"],
            "enableRateLimit": True,   # ccxt istek sınırlarını otomatik yönetir
        })
        if ex_cfg.get("testnet", True):
            self.exchange.set_sandbox_mode(True)

    async def warmup(self) -> dict[str, list[dict]]:
        """
        Başlangıç için geçmiş mumları çek. main.py bunları stratejiye verir ki
        bot açılır açılmaz indikatörler hesaplanabilsin.
        """
        history: dict[str, list[dict]] = {}
        for symbol in self.symbols:
            ohlcvs = await self.exchange.fetch_ohlcv(
                symbol, self.timeframe, limit=self.history_bars
            )
            history[symbol] = [_candle_dict(c) for c in ohlcvs]
            logger.info("Isınma: %s için %d mum yüklendi", symbol, len(ohlcvs))
        return history

    async def run(self) -> None:
        """Her sembol için ayrı bir dinleme görevi başlat."""
        tasks = [
            asyncio.create_task(self._stream_symbol(symbol), name=f"stream:{symbol}")
            for symbol in self.symbols
        ]
        await asyncio.gather(*tasks)

    async def _stream_symbol(self, symbol: str) -> None:
        """
        Tek sembol için sonsuz veri döngüsü.
        Sadece KAPANAN mumları yayınlarız: strateji, oluşumu bitmemiş mumla
        karar verirse aynı mum içinde sinyal girip çıkabilir (repaint sorunu).
        """
        last_ts: int | None = None
        while True:
            try:
                if self.use_websocket:
                    ohlcvs = await self.exchange.watch_ohlcv(symbol, self.timeframe)
                else:
                    ohlcvs = await self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=2)
                    await asyncio.sleep(5)  # REST polling aralığı

                # watch_ohlcv güncellenen son mum(lar)ı döndürür.
                # Sondan bir önceki mum "kapanmış" demektir; onu yayınla.
                if len(ohlcvs) >= 2:
                    closed = ohlcvs[-2]
                    if last_ts is None or closed[0] > last_ts:
                        last_ts = closed[0]
                        await self.bus.publish(MarketEvent(
                            symbol=symbol,
                            timeframe=self.timeframe,
                            candle=_candle_dict(closed),
                        ))
                        logger.debug("%s yeni mum: close=%.2f", symbol, closed[4])

            except Exception as exc:
                # Ağ kopması, borsa bakımı vb. - botu düşürme, bekle ve yeniden bağlan
                logger.warning("%s veri akışı hatası: %s - 10 sn sonra tekrar", symbol, exc)
                await asyncio.sleep(10)

    async def close(self) -> None:
        await self.exchange.close()
