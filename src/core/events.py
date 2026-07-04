"""
Olay (Event) tanımları - sistemin ortak dili.

Akış şu şekildedir:

    DataFetcher ──MarketEvent──> Strategy ──SignalEvent──> RiskManager
                                                               │
    Portfolio <──FillEvent── ExecutionEngine <──OrderEvent────┘

Her modül yalnızca olay alır ve olay yayar; birbirinin iç yapısını bilmez.
Bu sayede stratejiyi, borsayı veya risk kurallarını birbirinden bağımsız
değiştirebilirsin.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    MARKET = "MARKET"    # Yeni mum / emir defteri verisi geldi
    SIGNAL = "SIGNAL"    # Strateji bir al/sat/çık niyeti üretti
    ORDER = "ORDER"      # Risk yönetimi onayladı, boyutlandırılmış emir hazır
    FILL = "FILL"        # Emir gerçekleşti (gerçek veya simüle)


class SignalDirection(str, Enum):
    LONG = "LONG"        # Al (yükseliş beklentisi)
    SHORT = "SHORT"      # Sat/açığa sat (spot'ta genelde sadece eldekini satmak demektir)
    EXIT = "EXIT"        # Mevcut pozisyonu kapat


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Event:
    """Tüm olayların temel sınıfı."""
    type: EventType = field(init=False)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class MarketEvent(Event):
    """Yeni kapanan mum. candle sözlüğü: open/high/low/close/volume/ts."""
    symbol: str = ""
    timeframe: str = ""
    candle: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    """
    Stratejinin ürettiği ham niyet. Miktar İÇERMEZ - pozisyon büyüklüğüne
    strateji değil RiskManager karar verir. Bu ayrım bilinçlidir.
    """
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    price: float = 0.0            # Sinyal anındaki referans fiyat
    strength: float = 1.0         # 0-1 arası güven skoru (ML modeli olasılığı vb. için)
    stop_loss: float | None = None    # Strateji özel SL önerirse; None ise config'deki % kullanılır
    take_profit: float | None = None  # Strateji özel TP önerirse
    time_stop_bars: int = 0       # Dikey bariyer: >0 ise pozisyon bu kadar bar sonra zorla kapanır

    def __post_init__(self) -> None:
        self.type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    """Risk kontrolünden geçmiş, borsaya iletilmeye hazır emir."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    amount: float = 0.0           # Baz varlık cinsinden miktar (örn. BTC)
    price: float | None = None    # Limit emirse fiyat
    stop_loss: float = 0.0        # ZORUNLU - SL'siz emir ExecutionEngine tarafından reddedilir
    take_profit: float = 0.0      # ZORUNLU
    time_stop_bars: int = 0       # Dikey bariyer (0 = zaman durdurucu yok)
    reason: str = ""              # Log için: "sma_crossover LONG", "STOP_LOSS tetiklendi" vb.

    def __post_init__(self) -> None:
        self.type = EventType.ORDER


@dataclass
class FillEvent(Event):
    """Gerçekleşen işlem - komisyon ve slippage DAHİL nihai rakamlar."""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    amount: float = 0.0
    fill_price: float = 0.0       # Slippage sonrası gerçekleşme fiyatı
    fee_quote: float = 0.0        # Komisyon (USDT cinsinden)
    order_id: str = ""
    reason: str = ""
    latency_ms: float = 0.0       # Emir gönderimi -> gerçekleşme gecikmesi (izleme için)

    def __post_init__(self) -> None:
        self.type = EventType.FILL
