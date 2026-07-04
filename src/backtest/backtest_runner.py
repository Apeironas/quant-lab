"""
BACKTEST MODÜLÜ - stratejiyi geçmiş veri üzerinde, canlıymış gibi koşturur.

Çalıştırma (proje kök dizininden):
    python -m src.backtest.backtest_runner --days 365 --timeframe 1h
    python -m src.backtest.backtest_runner --days 90 --timeframe 15m --verbose

Nasıl çalışır:
  1. Geçmiş OHLCV verisi Binance PRODUCTION genel uçlarından indirilir
     (API anahtarı GEREKMEZ; testnet kullanılmaz çünkü testnet geçmişi kısadır
     ve periyodik sıfırlanır). İndirilen veri data/cache/ altına kaydedilir;
     aynı parametrelerle ikinci çalıştırma indirme yapmaz.
  2. Mumlar TEK TEK MarketEvent olarak EventBus'a akıtılır - canlıdaki akışın
     birebir aynısı. Strategy, RiskManager, ExecutionEngine, Portfolio
     DEĞİŞMEDEN aynı kodla çalışır. Komisyon + slippage tavizsiz uygulanır.
  3. Sonunda performans raporu basılır: işlem sayısı, kazanma oranı,
     max drawdown, brüt/net PnL, al-ve-tut kıyası.

Bilinçli basitleştirme: emir, sinyal mumunun kapanış fiyatından (+slippage)
dolar. Daha da katı olmak istersen "bir sonraki mumun açılışından doldur"
kuralına geçilebilir - farklıysa stratejin zaten kırılgandır.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import ccxt.async_support as accxt
import pandas as pd

from ..core.event_bus import EventBus
from ..core.events import (EventType, MarketEvent, OrderEvent, OrderSide,
                           OrderType, SignalEvent)
from ..execution.execution_engine import ExecutionEngine
from ..portfolio.portfolio import Portfolio
from ..risk.risk_manager import RiskManager
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging

logger = logging.getLogger("backtest")

CACHE_DIR = Path("data/cache")


# ===================================================================== #
#  1) VERİ İNDİRME (sayfalı, önbellekli)
# ===================================================================== #
async def download_ohlcv(exchange_id: str, symbol: str, timeframe: str,
                         days: int) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{exchange_id}_{symbol.replace('/', '')}_{timeframe}_{days}d.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        logger.info("Önbellekten yüklendi: %s (%d mum)", cache_file, len(df))
        return df

    exchange = getattr(accxt, exchange_id)({"enableRateLimit": True})
    rows: list[list] = []
    try:
        since = exchange.milliseconds() - days * 86_400_000
        batch_no = 0
        while True:
            batch = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 1
            batch_no += 1
            if batch_no % 10 == 0:
                logger.info("İndiriliyor... %d mum (son: %s)", len(rows),
                            pd.to_datetime(batch[-1][0], unit="ms"))
            if len(batch) < 1000:
                break
    finally:
        await exchange.close()

    df = (pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
          .drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    logger.info("İndirme bitti: %d mum -> %s", len(df), cache_file)
    return df


# ===================================================================== #
#  2) BACKTEST DÖNGÜSÜ
# ===================================================================== #
async def run_backtest(args: argparse.Namespace) -> None:
    config = load_config()
    config["mode"] = "paper"  # backtest DAİMA paper: emir asla dışarı gitmez

    if args.strategy:  # config'i değiştirmeden geçici strateji seçimi
        config["strategy"]["name"] = args.strategy
    symbol = args.symbol or config["market_data"]["symbols"][0]
    timeframe = args.timeframe or config["market_data"]["timeframe"]

    # Rapor temiz kalsın: işlem başına loglar sadece --verbose ile görünür
    setup_logging(config["logging"]["dir"], "INFO" if args.verbose else "WARNING")
    logger.setLevel(logging.INFO)

    df = await download_ohlcv(config["exchange"]["id"], symbol, timeframe, args.days)
    candles: list[dict] = df.to_dict("records")

    warm = config["market_data"]["history_bars"]
    if len(candles) <= warm + 10:
        raise SystemExit(f"Yetersiz veri: {len(candles)} mum var, en az {warm + 10} gerekli. "
                         f"--days değerini artır veya daha küçük timeframe seç.")

    # --- Sistemi canlıdakiyle AYNI şekilde kur ---
    from ..main import STRATEGIES  # döngüsel import olmasın diye burada
    bus = EventBus()
    portfolio = Portfolio(config)
    risk = RiskManager(config, portfolio)
    strategy = STRATEGIES[config["strategy"]["name"]](config, bus)
    execution = ExecutionEngine(config, bus, portfolio)

    bus.subscribe(EventType.MARKET, portfolio.on_market_event)
    bus.subscribe(EventType.MARKET, execution.on_market_event)
    bus.subscribe(EventType.MARKET, strategy.on_market_event)

    async def on_signal(event: SignalEvent) -> None:
        order = risk.evaluate(event)
        if order is not None:
            await bus.publish(order)

    bus.subscribe(EventType.SIGNAL, on_signal)
    bus.subscribe(EventType.ORDER, execution.on_order)
    bus.subscribe(EventType.FILL, portfolio.on_fill)
    bus.subscribe(EventType.FILL, strategy.on_fill)   # durum farkındalığı (v2.1)

    # --- Isınma + akış ---
    strategy.warmup({symbol: candles[:warm]})
    portfolio.last_prices[symbol] = candles[warm - 1]["close"]

    equity_curve: list[float] = []
    stream = candles[warm:]
    logger.info("Backtest başlıyor: %s %s, %d mum akıtılacak", symbol, timeframe, len(stream))

    for i, candle in enumerate(stream):
        await bus.publish(MarketEvent(symbol=symbol, timeframe=timeframe, candle=candle))
        await bus.drain()  # mum -> sinyal -> emir -> fill zinciri burada tamamlanır
        equity_curve.append(portfolio.equity())
        if (i + 1) % 5000 == 0:
            logger.info("İlerleme: %d/%d mum, equity=%.2f", i + 1, len(stream), equity_curve[-1])

    # --- Açık pozisyon kaldıysa son fiyattan kapat (istatistikler net olsun) ---
    if symbol in portfolio.positions:
        pos = portfolio.positions[symbol]
        await bus.publish(OrderEvent(
            symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
            amount=pos["amount"], reason="backtest sonu - zorunlu kapanış",
        ))
        await bus.drain()
        equity_curve.append(portfolio.equity())

    print_report(config, portfolio, equity_curve, stream, symbol, timeframe, args.days)


# ===================================================================== #
#  3) PERFORMANS RAPORU
# ===================================================================== #
def _max_drawdown(equity_curve: list[float]) -> float:
    """Tepe noktadan en derin düşüş (oran olarak, 0.15 = %15)."""
    peak, max_dd = float("-inf"), 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
    return max_dd


def print_report(config: dict, portfolio: Portfolio, equity_curve: list[float],
                 stream: list[dict], symbol: str, timeframe: str, days: int) -> None:
    initial = config["portfolio"]["initial_cash"]
    final = equity_curve[-1] if equity_curve else initial
    trades = portfolio.trades
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    gross_pnl = sum(t["pnl"] + t["fees"] for t in trades)   # komisyonsuz fiyat hareketi kârı
    net_pnl = portfolio.realized_pnl                          # komisyon + slippage sonrası
    total_fees = portfolio.total_fees
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    max_dd = _max_drawdown(equity_curve) * 100
    profit_factor = (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
                     if losses and sum(t["pnl"] for t in losses) != 0 else float("inf"))

    # Kıyas: hiç strateji olmasa, ilk mumda alıp sonuna kadar tutsaydık?
    buy_hold_pct = 100 * (stream[-1]["close"] / stream[0]["close"] - 1)
    strategy_pct = 100 * (final / initial - 1)

    start = pd.to_datetime(stream[0]["ts"], unit="ms").date()
    end = pd.to_datetime(stream[-1]["ts"], unit="ms").date()

    w = 58
    line = "=" * w
    print()
    print(line)
    print(f"  BACKTEST RAPORU  |  {symbol}  {timeframe}  |  {start} -> {end}")
    print(line)
    strat_name = config["strategy"]["name"]
    strat_params = config["strategy"].get(strat_name) or config["strategy"].get("params", {})
    print(f"  Strateji            : {strat_name} {strat_params}")
    print(f"  Mum sayısı          : {len(stream):,}")
    print(f"  Başlangıç sermayesi : {initial:>12,.2f} USDT")
    print(f"  Bitiş sermayesi     : {final:>12,.2f} USDT")
    print("-" * w)
    print(f"  Toplam işlem        : {len(trades)} kapanan ({len(wins)} kazanç / {len(losses)} kayıp)")
    print(f"  Kazanma oranı       : %{win_rate:.1f}")
    print(f"  Profit factor       : {profit_factor:.2f}" if trades else "  Profit factor       : -")
    print(f"  Brüt PnL            : {gross_pnl:>+12,.2f} USDT  (komisyon/slippage öncesi)")
    print(f"  Toplam komisyon     : {total_fees:>12,.2f} USDT")
    print(f"  NET PnL             : {net_pnl:>+12,.2f} USDT  (%{strategy_pct:+.2f})")
    print(f"  Max drawdown        : %{max_dd:.2f}")
    print("-" * w)
    print(f"  Al-ve-tut kıyası    : %{buy_hold_pct:+.2f}  "
          f"({'strateji ÖNDE' if strategy_pct > buy_hold_pct else 'al-ve-tut ÖNDE'})")
    print(line)
    if trades and total_fees > abs(net_pnl):
        print("  UYARI: komisyonlar net PnL'den büyük - strateji aşırı işlem yapıyor olabilir.")
    if not trades:
        print("  Hiç işlem oluşmadı: sinyal koşulları bu dönemde hiç tetiklenmedi.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratejiyi geçmiş veride test et")
    parser.add_argument("--days", type=int, default=365, help="Kaç günlük veri (varsayılan: 365)")
    parser.add_argument("--timeframe", default=None, help="Mum periyodu (varsayılan: config.yaml)")
    parser.add_argument("--symbol", default=None, help="Parite (varsayılan: config.yaml)")
    parser.add_argument("--strategy", default=None,
                        help="Strateji seçimi (config'i ezmez; örn: --strategy ml)")
    parser.add_argument("--verbose", action="store_true", help="İşlem başına logları göster")
    args = parser.parse_args()
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()
