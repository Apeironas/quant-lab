"""
ÖRNEK STRATEJİ: SMA Kesişimi (Golden/Death Cross)

Hızlı ortalama yavaşı yukarı keserse AL, aşağı keserse ÇIK.

DİKKAT: Bu strateji kâr etmek için değil, iskeletin uçtan uca çalıştığını
göstermek için var. Kendi stratejini yazarken bu dosyayı şablon olarak
kopyala; asıl değiştireceğin yer generate_signal() metodudur.
"""
from __future__ import annotations

import pandas as pd

from ..core.events import SignalDirection, SignalEvent
from .base_strategy import BaseStrategy


class SmaCrossoverStrategy(BaseStrategy):
    NAME = "sma_crossover"

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> SignalEvent | None:
        fast_n = self.params.get("fast_period", 20)
        slow_n = self.params.get("slow_period", 50)

        # İndikatör ısınması: yeterli mum yoksa işlem yapma
        if len(candles) < slow_n + 2:
            return None

        close = candles["close"]
        fast = close.rolling(fast_n).mean()
        slow = close.rolling(slow_n).mean()
        # ('ta' kütüphanesiyle eşdeğeri: ta.trend.sma_indicator(close, window=fast_n))

        # Kesişim tespiti: bir önceki mumda altındaydı, şimdi üstünde
        crossed_up = fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]
        crossed_down = fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]

        price = float(close.iloc[-1])

        if crossed_up:
            return SignalEvent(symbol=symbol, direction=SignalDirection.LONG, price=price)
        if crossed_down:
            # Spot piyasada SHORT yerine pozisyondan çıkıyoruz
            return SignalEvent(symbol=symbol, direction=SignalDirection.EXIT, price=price)
        return None
