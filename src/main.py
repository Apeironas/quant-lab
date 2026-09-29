"""
ANA GİRİŞ NOKTASI - tüm modülleri kurar ve olay akışını bağlar.

Çalıştırma (proje kök dizininden):
    python -m src.main

Olay akışı:
    DataFetcher ─MarketEvent─> Strategy ─SignalEvent─> RiskManager
                                                            │
    Portfolio <──FillEvent── ExecutionEngine <──OrderEvent──┘
"""
from __future__ import annotations

import asyncio
import logging

from .core.event_bus import EventBus
from .core.events import EventType, SignalEvent
from .data.data_fetcher import DataFetcher
from .execution.execution_engine import ExecutionEngine
from .portfolio.portfolio import Portfolio
from .risk.risk_manager import RiskManager
from .strategy.ml_strategy import MLStrategy
from .strategy.sma_crossover import SmaCrossoverStrategy
from .utils.config_loader import load_config
from .utils.logger import setup_logging

logger = logging.getLogger("main")

# =====================================================================
# Strateji kaydı: yeni strateji import edilip bu sözlüğe eklenir,
# ardından config.yaml'daki strategy.name ile seçilir.
# =====================================================================
STRATEGIES = {
    "sma_crossover": SmaCrossoverStrategy,
    "ml": MLStrategy,   # önce eğit: python -m src.ml.train_example
    # "benim_stratejim": BenimStratejim,
}


async def periodic_report(portfolio: Portfolio, interval_s: int = 300) -> None:
    """5 dakikada bir portföy özeti logla - bot yaşıyor mu, durum ne?"""
    while True:
        await asyncio.sleep(interval_s)
        logger.info("DURUM | %s", portfolio.summary())


async def run() -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], config["logging"]["level"])
    logger.info("Bot başlıyor | mod=%s | testnet=%s | semboller=%s",
                config["mode"], config["exchange"]["testnet"],
                config["market_data"]["symbols"])

    # --- Modülleri kur ---
    bus = EventBus()
    portfolio = Portfolio(config)
    risk = RiskManager(config, portfolio)
    strategy_cls = STRATEGIES[config["strategy"]["name"]]
    strategy = strategy_cls(config, bus)
    execution = ExecutionEngine(config, bus, portfolio)
    fetcher = DataFetcher(config, bus)

    # --- Olay aboneliklerini bağla (sıra önemli: portföy fiyatı önce görsün) ---
    bus.subscribe(EventType.MARKET, portfolio.on_market_event)   # fiyat aynası
    bus.subscribe(EventType.MARKET, execution.on_market_event)   # SL/TP kontrolü
    bus.subscribe(EventType.MARKET, strategy.on_market_event)    # sinyal üretimi

    async def on_signal(event: SignalEvent) -> None:
        """Sinyal -> risk süzgeci -> (onaylanırsa) emir."""
        order = risk.evaluate(event)
        if order is not None:
            await bus.publish(order)

    bus.subscribe(EventType.SIGNAL, on_signal)
    bus.subscribe(EventType.ORDER, execution.on_order)
    bus.subscribe(EventType.FILL, portfolio.on_fill)
    bus.subscribe(EventType.FILL, strategy.on_fill)   # durum farkındalığı (v2.1)

    # --- Isınma: geçmiş veriyi çek, stratejiye ver ---
    await execution.connect()
    history = await fetcher.warmup()
    strategy.warmup(history)
    for symbol, candles in history.items():
        if candles:
            portfolio.last_prices[symbol] = candles[-1]["close"]

    # --- Ana döngüler ---
    try:
        await asyncio.gather(
            bus.run(),                     # olay dağıtımı
            fetcher.run(),                 # canlı veri akışı
            periodic_report(portfolio),    # durum raporu
        )
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Kapanıyor... Son durum: %s", portfolio.summary())
        await fetcher.close()
        await execution.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nBot kullanıcı tarafından durduruldu.")


if __name__ == "__main__":
    main()
