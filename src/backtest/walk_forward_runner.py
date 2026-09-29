"""
WALK-FORWARD (KAYAN PENCERE) ANALİZİ - dürüst performans ölçümü

Çalıştırma (proje kökünden):
    python -m src.backtest.walk_forward_runner --days 540 --timeframe 1h --train-days 180 --test-days 30
    python -m src.backtest.walk_forward_runner --days 720 --timeframe 1h --train-days 180 --test-days 90 --verbose

Döngü mantığı:
    [--- train (180 gün) ---][embargo][-- test/OOS (30 gün) --]
                 [--- train (180 gün) ---][embargo][-- test/OOS --]
                              ... pencere test kadar kayar ...

    1. Train diliminde model SIFIRDAN eğitilir (features.build_dataset)
    2. Model, MLStrategy'ye ENJEKTE edilir (model dosyası yazılmaz)
    3. Test diliminde tam backtest koşulur: komisyon + slippage + SL/TP +
       risk yönetimi - canlı akışın birebir aynısı
    4. Pencere kayar; sermaye bir sonraki pencereye DEVREDER (bileşik)
    5. Tüm OOS equity eğrileri uç uca eklenir (stitching) -> tek rapor

SIZINTI KORUMASI (iki katman):
    a) build_dataset pencere İÇİNDE çalışır: etiketlerin baktığı gelecek
       (horizon bar) pencere sonunda NaN olur ve atılır - train etiketi
       test dönemini asla göremez.
    b) Yine de train sonu ile test başı arasına `horizon` bar EMBARGO konur.
    Strateji tamponunun test öncesi mumlarla ısıtılması sızıntı DEĞİLDİR:
    t anında geçmiş zaten bilinir; embargo etiket ufku içindir.

Bu dosya SADECE orkestrasyondur: features.py, ml_strategy.py, portfolio,
risk, execution kodlarına dokunmaz - hepsini olduğu gibi kullanır.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import logging

import pandas as pd

from ..core.event_bus import EventBus
from ..core.events import (EventType, MarketEvent, OrderEvent, OrderSide,
                           OrderType, SignalEvent)
from ..execution.execution_engine import ExecutionEngine
from ..ml.features import MAX_LOOKBACK, build_dataset
from ..ml.train_example import make_model
from ..portfolio.portfolio import Portfolio
from ..risk.risk_manager import RiskManager
from ..strategy.ml_strategy import MLStrategy
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging
from .backtest_runner import download_ohlcv

logger = logging.getLogger("walkforward")


def timeframe_minutes(tf: str) -> int:
    unit, n = tf[-1], int(tf[:-1])
    return n * {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]


# ===================================================================== #
#  TEK PENCERE: enjekte modelle test diliminde tam backtest
# ===================================================================== #
async def run_window(config: dict, model, symbol: str, timeframe: str,
                     warm_candles: list[dict], test_candles: list[dict],
                     initial_cash: float) -> tuple[Portfolio, list[float]]:
    win_cfg = copy.deepcopy(config)
    win_cfg["mode"] = "paper"
    win_cfg["portfolio"]["initial_cash"] = initial_cash  # sermaye devri (bileşik)

    bus = EventBus()
    portfolio = Portfolio(win_cfg)
    risk = RiskManager(win_cfg, portfolio)
    strategy = MLStrategy(win_cfg, bus, model=model)  # <- dinamik enjeksiyon
    execution = ExecutionEngine(win_cfg, bus, portfolio)

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

    strategy.warmup({symbol: warm_candles})
    portfolio.last_prices[symbol] = warm_candles[-1]["close"]

    equity_curve: list[float] = []
    for candle in test_candles:
        await bus.publish(MarketEvent(symbol=symbol, timeframe=timeframe, candle=candle))
        await bus.drain()
        equity_curve.append(portfolio.equity())

    # Pencere sonunda pozisyon açık kalamaz: bir sonraki pencerenin modeli
    # farklı olacak; devreden sermaye net/nakit olmalı.
    if symbol in portfolio.positions:
        pos = portfolio.positions[symbol]
        await bus.publish(OrderEvent(
            symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
            amount=pos["amount"], reason="pencere sonu - zorunlu kapanış",
        ))
        await bus.drain()
        equity_curve.append(portfolio.equity())

    return portfolio, equity_curve


# ===================================================================== #
#  ORKESTRASYON
# ===================================================================== #
async def run_walk_forward(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], "INFO" if args.verbose else "WARNING")
    logger.setLevel(logging.INFO)

    symbol = args.symbol or config["market_data"]["symbols"][0]
    timeframe = args.timeframe

    # v2.1: dikey bariyer = etiket horizon'u (bariyer-sadık işlem yönetimi)
    config["strategy"].setdefault("ml", {})["time_stop_bars"] = args.horizon

    bars_per_day = 1440 // timeframe_minutes(timeframe)
    train_bars = args.train_days * bars_per_day
    test_bars = args.test_days * bars_per_day
    embargo = args.horizon
    warmup_bars = max(MAX_LOOKBACK + 10,
                      config["strategy"].get("ml", {}).get("min_bars", 60))

    df = await download_ohlcv(config["exchange"]["id"], symbol, timeframe, args.days)
    if getattr(args, "funding", False):
        from ..data.derivatives import download_funding_history, merge_derivatives
        funding_df = await download_funding_history(config["exchange"]["id"], symbol, args.days)
        df = merge_derivatives(df, funding_df=funding_df)
        logger.info("Funding rate özellikleri AKTİF (%d kayıt birleştirildi)", len(funding_df))
    if getattr(args, "orderbook", False):
        from ..data.orderbook_store import merge_orderbook
        df = merge_orderbook(df, symbol, bar_ms=timeframe_minutes(timeframe) * 60_000)
        logger.info("Order book (OBI) özellikleri AKTİF")
    candles: list[dict] = df.to_dict("records")

    # Pencere sınırlarını çıkar: (train0, train1, test0, test1)
    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while start + train_bars + embargo + test_bars <= len(candles):
        t1 = start + train_bars
        windows.append((start, t1, t1 + embargo, t1 + embargo + test_bars))
        start += test_bars  # pencere, test dilimi kadar kayar
    if not windows:
        raise SystemExit(
            f"Veri yetmiyor: {len(candles)} mum var; 1 pencere için "
            f"{train_bars + embargo + test_bars} gerekli. --days'i artır "
            f"veya --train-days/--test-days'i küçült.")

    logger.info("Walk-forward: %d pencere | train=%dg (%d bar) + embargo=%d bar + test=%dg (%d bar)",
                len(windows), args.train_days, train_bars, embargo, args.test_days, test_bars)

    # --- Pencere döngüsü ---
    initial_cash = config["portfolio"]["initial_cash"]
    carry = initial_cash                    # sermaye pencereden pencereye devreder
    stitched_equity: list[float] = []       # uç uca eklenen OOS equity eğrisi
    all_trades: list[dict] = []
    window_rows: list[dict] = []
    total_fees = 0.0

    for i, (t0, t1, s0, s1) in enumerate(windows, 1):
        train_df = df.iloc[t0:t1]
        X, y = build_dataset(train_df, horizon=args.horizon, threshold=args.threshold)
        model = make_model()
        model.fit(X, y)

        warm = candles[s0 - warmup_bars:s0]
        test = candles[s0:s1]
        portfolio, curve = await run_window(
            config, model, symbol, timeframe, warm, test, initial_cash=carry)

        start_dt = pd.to_datetime(test[0]["ts"], unit="ms").date()
        end_dt = pd.to_datetime(test[-1]["ts"], unit="ms").date()
        window_pnl = curve[-1] - carry
        window_rows.append({
            "no": i, "start": start_dt, "end": end_dt,
            "trades": len(portfolio.trades),
            "pnl": window_pnl,
            "pct": 100 * window_pnl / carry,
            "bh_pct": 100 * (test[-1]["close"] / test[0]["close"] - 1),
        })
        logger.info("Pencere %d/%d | %s -> %s | %d işlem | PnL %+.2f (%%%+.2f) | B&H %%%+.2f",
                    i, len(windows), start_dt, end_dt, len(portfolio.trades),
                    window_pnl, window_rows[-1]["pct"], window_rows[-1]["bh_pct"])

        all_trades.extend(portfolio.trades)
        total_fees += portfolio.total_fees
        stitched_equity.extend(curve)       # uç uca ekleme (stitching)
        carry = curve[-1]                    # sermaye devri

    print_wf_report(config, args, symbol, timeframe, windows, window_rows,
                    all_trades, stitched_equity, total_fees, initial_cash, candles)


# ===================================================================== #
#  RAPOR - yalnızca OUT-OF-SAMPLE rakamlar
# ===================================================================== #
def _max_drawdown(curve: list[float]) -> float:
    peak, mdd = float("-inf"), 0.0
    for eq in curve:
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return mdd


def print_wf_report(config, args, symbol, timeframe, windows, rows,
                    trades, equity, total_fees, initial_cash, candles) -> None:
    final = equity[-1] if equity else initial_cash
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    gross = sum(t["pnl"] + t["fees"] for t in trades)
    net = final - initial_cash
    max_dd = _max_drawdown(equity) * 100
    loss_sum = sum(t["pnl"] for t in losses)
    pf = (sum(t["pnl"] for t in wins) / abs(loss_sum)) if loss_sum else float("inf")
    profitable_windows = sum(1 for r in rows if r["pnl"] > 0)

    # OOS dönemin tamamı için al-ve-tut kıyası
    first_test = candles[windows[0][2]]
    last_test = candles[windows[-1][3] - 1]
    bh_total = 100 * (last_test["close"] / first_test["close"] - 1)
    strat_total = 100 * (final / initial_cash - 1)

    w = 66
    print()
    print("=" * w)
    print(f"  WALK-FORWARD RAPORU (SADECE OUT-OF-SAMPLE)  |  {symbol}  {timeframe}")
    print("=" * w)
    print(f"  Pencere yapısı      : train {args.train_days}g + embargo {args.horizon} bar "
          f"+ test {args.test_days}g")
    print(f"  Pencere sayısı      : {len(rows)}  (sermaye pencereler arası devreder)")
    print()
    print("  --- Pencere bazında OOS performans (tutarlılık kontrolü) ---")
    print(f"  {'#':>2}  {'Dönem':<25} {'İşlem':>5} {'Net PnL':>12} {'Getiri':>9} {'B&H':>9}")
    for r in rows:
        print(f"  {r['no']:>2}  {str(r['start'])} -> {str(r['end']):<12} "
              f"{r['trades']:>5} {r['pnl']:>+12,.2f} {r['pct']:>+8.2f}% {r['bh_pct']:>+8.2f}%")
    print()
    print("  --- Birleşik (stitched) OOS sonuçları ---")
    print(f"  Başlangıç -> Bitiş  : {initial_cash:,.2f} -> {final:,.2f} USDT")
    print(f"  Toplam işlem        : {len(trades)} kapanan ({len(wins)} kazanç / {len(losses)} kayıp)")
    print(f"  Kazanma oranı       : %{win_rate:.1f}")
    print(f"  Profit factor       : {pf:.2f}" if trades else "  Profit factor       : -")
    print(f"  Brüt PnL            : {gross:>+12,.2f} USDT  (komisyon/slippage öncesi)")
    print(f"  Toplam komisyon     : {total_fees:>12,.2f} USDT")
    print(f"  NET PnL             : {net:>+12,.2f} USDT  (%{strat_total:+.2f})")
    print(f"  Max drawdown        : %{max_dd:.2f}  (birleşik eğri üzerinde)")
    print(f"  Kârlı pencere       : {profitable_windows}/{len(rows)}")
    print("-" * w)
    print(f"  Al-ve-tut kıyası    : %{bh_total:+.2f}  "
          f"({'strateji ÖNDE' if strat_total > bh_total else 'al-ve-tut ÖNDE'})")
    print("=" * w)
    if rows and profitable_windows <= len(rows) // 2:
        print("  YORUM: Pencerelerin yarısından azı kârlı - strateji dönemsel")
        print("  şansa yaslanıyor olabilir; özellik/eşik/horizon üzerinde çalış.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward (kayan pencere) analizi")
    parser.add_argument("--days", type=int, default=540,
                        help="İndirilecek toplam veri (gün, varsayılan: 540)")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--train-days", type=int, default=180,
                        help="Eğitim penceresi (gün, varsayılan: 180)")
    parser.add_argument("--test-days", type=int, default=30,
                        help="Test/OOS penceresi ve kayma adımı (gün, varsayılan: 30)")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Etiket ufku = embargo bar sayısı (varsayılan: 6)")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--funding", action="store_true",
                        help="Funding rate özelliklerini ekle (perpetual verisi)")
    parser.add_argument("--orderbook", action="store_true",
                        help="Order book (OBI) özelliklerini ekle (toplanan L2 verisi)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_walk_forward(args))


if __name__ == "__main__":
    main()
