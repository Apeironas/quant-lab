"""
STRATEJİ MOTORU - temel sınıf

Kendi stratejini yazmak için:
  1. Bu sınıftan türeyen yeni bir dosya oluştur (örn. src/strategy/benim_strateji.py)
  2. Sadece `generate_signal()` metodunu doldur
  3. config.yaml'da strategy.name'i değiştir ve main.py'deki STRATEGIES
     sözlüğüne sınıfını ekle

Strateji İZOLEDİR: borsayı, emirleri, bakiyeyi bilmez. Tek girdisi mum verisi,
tek çıktısı SignalEvent (veya None). Pozisyon büyüklüğü, komisyon, SL/TP
uygulaması başka modüllerin işidir. Bu izolasyon sayesinde aynı strateji
hem canlıda hem backtest'te değişmeden çalışır.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

from ..core.event_bus import EventBus
from ..core.events import MarketEvent, SignalEvent

logger = logging.getLogger("strategy")


class BaseStrategy(ABC):
    #: İndikatör hesaplamak için bellekte tutulacak azami mum sayısı
    MAX_BUFFER = 500

    #: Alt sınıflar bunu tanımlar; config.yaml'daki parametre bloğunun adıdır
    #: (strategy.<NAME>: ...). main.py'deki STRATEGIES anahtarıyla aynı tutulur.
    NAME: str = ""

    def __init__(self, config: dict, bus: EventBus) -> None:
        self.bus = bus
        strat_cfg: dict = config["strategy"]
        # Kendi adına ait parametre bloğunu al; eski düz "params" da desteklenir
        self.params: dict = strat_cfg.get(self.NAME) or strat_cfg.get("params") or {}
        # Sembol başına mum tamponu (DataFrame: ts, open, high, low, close, volume)
        self._buffers: dict[str, pd.DataFrame] = {}

    def warmup(self, history: dict[str, list[dict]]) -> None:
        """Başlangıçta geçmiş mumlarla tamponu doldur (DataFetcher.warmup çıktısı)."""
        for symbol, candles in history.items():
            self._buffers[symbol] = pd.DataFrame(candles)
            logger.info("Strateji ısındı: %s, %d mum", symbol, len(candles))

    async def on_market_event(self, event: MarketEvent) -> None:
        """EventBus tarafından her yeni mumda çağrılır. Bunu değiştirmene gerek yok."""
        df = self._buffers.get(event.symbol, pd.DataFrame())
        df = pd.concat([df, pd.DataFrame([event.candle])], ignore_index=True)
        self._buffers[event.symbol] = df.tail(self.MAX_BUFFER).reset_index(drop=True)

        signal = self.generate_signal(event.symbol, self._buffers[event.symbol])
        if signal is not None:
            # Sinyal zamanı = mumun zamanı: canlıda ~şimdi, backtest'te tarihsel an.
            # RiskManager'ın günlük devre kesicisi bu damgayı kullanır.
            signal.timestamp_ms = int(event.candle["ts"])
            logger.info("SİNYAL: %s %s @ %.2f (güç=%.2f)",
                        signal.symbol, signal.direction, signal.price, signal.strength)
            await self.bus.publish(signal)

    async def on_fill(self, event) -> None:
        """
        DURUM FARKINDALIĞI: gerçekleşen emirler (FILL) stratejiye geri beslenir.
        Varsayılan: hiçbir şey yapma. Pozisyon durumu takip etmesi gereken
        stratejiler (örn. MLStrategy) bunu override eder. main.py ve backtest
        sürücüleri stratejiyi FILL olaylarına otomatik abone eder.
        """

    @abstractmethod
    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> SignalEvent | None:
        """
        Al/sat mantığının uygulandığı yer; alt sınıflar bunu doldurur.

        Girdi:
            symbol  - "BTC/USDT" gibi parite adı
            candles - kronolojik sıralı mum DataFrame'i
                      (sütunlar: ts, open, high, low, close, volume;
                       son satır = en yeni KAPANMIŞ mum)
        Çıktı:
            SignalEvent -> işlem niyeti (LONG / SHORT / EXIT)
            None        -> bu mumda işlem yok

        Notlar:
          - İndikatörler 'ta' kütüphanesinden: ta.momentum.rsi(close, window=14)
          - ML modelleri __init__'te yüklenir; olasılık SignalEvent.strength'e yazılır
          - Pozisyon boyutu burada hesaplanmaz, RiskManager'a aittir
        """
        raise NotImplementedError
