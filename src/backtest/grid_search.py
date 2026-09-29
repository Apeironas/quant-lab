"""
NESTED GRID-SEARCH - walk-forward içinde sızıntısız hiperparametre arama

Çalıştırma (proje kökünden):
    python -m src.backtest.grid_search --days 540 --timeframe 1h --train-days 180 --test-days 30
    python -m src.backtest.grid_search --horizons 6,24 --thresholds 0.60:0.40,0.65:0.35 --days 320 --test-days 45

Her walk-forward penceresinde akış (NESTED yapı):

    [============ TRAIN (180g) ============][emb][== TEST/OOS (30g) ==]
    [-- iç-train %70 --][emb][-- iç-val --]        ^
            |                     |                |
            |  her horizon için   |                |
            +--> model eğit ------+                |
                 her eşik çifti için iç-val'de     |
                 NET (komisyon+slippage sonrası)   |
                 backtest -> skor                  |
                                                   |
    En iyi skorlu (horizon, eşikler) -> TÜM train ile yeniden eğit --> OOS

SIZINTI GARANTİLERİ:
    - Parametre seçimi YALNIZ iç-validation skoruna bakar; OOS penceresi
      seçime hiçbir şekilde katılmaz (nested cross-validation).
    - İç-train/iç-val arasına horizon kadar, train/test arasına
      max(horizons) kadar embargo konur.
    - Skor, event-motorundan geçen GERÇEKÇİ backtest'ten gelir: komisyon,
      slippage, SL/TP ve risk yönetimi dahil (brüt kâr DEĞİL).

Metrik: --metric sharpe (varsayılan; bar-getirilerinden yıllıklandırılmış
Net Sharpe) veya --metric pnl (net PnL). Hiç işlem üretmeyen kombinasyon
değerlendirilemez sayılır (-inf) ve seçilemez.

Bu dosya yalnız orkestrasyondur: run_window'u walk_forward_runner'dan,
make_model'i train_example'dan, build_dataset'i features'tan aynen kullanır -
mevcut dosyalarda sıfır değişiklik.

Süre notu: varsayılan grid (4 horizon x 3 eşik) x 11 pencere ~30-40 dk sürer;
hızlı deneme için --horizons/--thresholds ile grid'i küçült.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import math

import numpy as np
import pandas as pd

from ..ml.features import MAX_LOOKBACK, build_dataset
from ..ml.train_example import make_model
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging
from .backtest_runner import download_ohlcv
from .walk_forward_runner import _max_drawdown, run_window, timeframe_minutes

logger = logging.getLogger("gridsearch")


def parse_thresholds(s: str) -> list[tuple[float, float]]:
    """'0.60:0.40,0.65:0.35' -> [(0.60, 0.40), (0.65, 0.35)]"""
    pairs = []
    for chunk in s.split(","):
        lo, ex = chunk.split(":")
        lo, ex = float(lo), float(ex)
        if ex >= lo:
            raise SystemExit(f"Geçersiz eşik çifti {lo}:{ex} - çıkış eşiği girişten küçük olmalı (histerezis).")
        pairs.append((lo, ex))
    return pairs


def score_curve(curve: list[float], n_trades: int, metric: str, bars_per_year: int) -> float:
    """Validasyon equity eğrisinden NET skor. İşlemsiz kombinasyon seçilemez."""
    if n_trades == 0 or len(curve) < 2:
        return float("-inf")
    if metric == "pnl":
        return curve[-1] - curve[0]
    eq = np.asarray(curve, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    std = rets.std()
    if std == 0:
        return float("-inf")
    return float(rets.mean() / std * math.sqrt(bars_per_year))  # yıllıklandırılmış Sharpe


def with_thresholds(config: dict, lo: float, ex: float,
                    time_stop_bars: int | None = None) -> dict:
    """
    Eşikleri (ve v2.1: dikey bariyeri) config kopyasına enjekte et.
    time_stop_bars = kombonun horizon'u: etiket hangi dikey bariyerle
    üretildiyse yürütme de aynı bariyeri uygular (bariyer-sadık yönetim).
    """
    cfg = copy.deepcopy(config)
    cfg["strategy"].setdefault("ml", {})
    cfg["strategy"]["ml"]["long_threshold"] = lo
    cfg["strategy"]["ml"]["exit_threshold"] = ex
    if time_stop_bars is not None:
        cfg["strategy"]["ml"]["time_stop_bars"] = time_stop_bars
    return cfg


async def run_grid_search(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], "INFO" if args.verbose else "WARNING")
    logger.setLevel(logging.INFO)

    if args.maker:
        # Maker yürütme modeli: limit giriş (muhafazakâr dolum) + maker TP
        config.setdefault("execution", {})["entry_mode"] = "maker"
        logger.info("MAKER yürütme modeli AKTİF (limit giriş + maker TP çıkışı)")

    horizons = [int(h) for h in args.horizons.split(",")]
    thresholds = parse_thresholds(args.thresholds)
    symbol = args.symbol or config["market_data"]["symbols"][0]
    timeframe = args.timeframe

    bars_per_day = 1440 // timeframe_minutes(timeframe)
    bars_per_year = bars_per_day * 365
    train_bars = args.train_days * bars_per_day
    test_bars = args.test_days * bars_per_day
    embargo = max(horizons)  # dış embargo: en kötü durum (en uzun etiket ufku)
    warmup_bars = max(MAX_LOOKBACK + 10,
                      config["strategy"].get("ml", {}).get("min_bars", 60))

    df = await download_ohlcv(config["exchange"]["id"], symbol, timeframe, args.days)
    if args.funding:
        # Funding rate'i sızıntısız yapıştır; kolon mum sözlükleriyle tüm
        # akışa (eğitim + OOS çıkarım) otomatik taşınır (bkz. derivatives.py)
        from ..data.derivatives import download_funding_history, merge_derivatives
        funding_df = await download_funding_history(config["exchange"]["id"], symbol, args.days)
        df = merge_derivatives(df, funding_df=funding_df)
        logger.info("Funding rate özellikleri AKTİF (%d kayıt birleştirildi)", len(funding_df))
    if args.orderbook:
        # OBI bar istatistiklerini mumlara göm (toplama dönemi dışı NaN kalır
        # ve eğitimden düşer - bkz. orderbook_store.py)
        from ..data.orderbook_store import merge_orderbook
        df = merge_orderbook(df, symbol, bar_ms=timeframe_minutes(timeframe) * 60_000)
        logger.info("Order book (OBI) özellikleri AKTİF")
    candles: list[dict] = df.to_dict("records")

    windows: list[tuple[int, int, int, int]] = []
    start = 0
    while start + train_bars + embargo + test_bars <= len(candles):
        t1 = start + train_bars
        windows.append((start, t1, t1 + embargo, t1 + embargo + test_bars))
        start += test_bars
    if not windows:
        raise SystemExit("Veri yetmiyor - --days'i artır veya pencereleri küçült.")

    n_combo = len(horizons) * len(thresholds)
    logger.info("Nested grid-search: %d pencere x %d kombinasyon (%s horizon x %s eşik) | metrik=%s",
                len(windows), n_combo, horizons, [f"{a}:{b}" for a, b in thresholds], args.metric)

    initial_cash = config["portfolio"]["initial_cash"]
    carry = initial_cash
    stitched: list[float] = []
    all_trades: list[dict] = []
    rows: list[dict] = []
    total_fees = 0.0

    for i, (t0, t1, s0, s1) in enumerate(windows, 1):
        # ---------- İÇ DÖNGÜ: yalnız train dilimi içinde parametre seçimi ----------
        it1 = t0 + int((t1 - t0) * 0.7)   # iç-train sonu (%70)
        best: dict | None = None

        for h in horizons:
            # Horizon etiketi değiştirir -> horizon başına model eğitilir
            X, y = build_dataset(df.iloc[t0:it1], horizon=h, threshold=args.label_threshold)
            model = make_model()
            model.fit(X, y)

            val_start = it1 + h               # iç embargo = bu horizon'un ufku
            warm = candles[val_start - warmup_bars:val_start]
            val = candles[val_start:t1]

            for lo, ex in thresholds:
                cfg = with_thresholds(config, lo, ex, time_stop_bars=h)
                pf_val, curve = await run_window(
                    cfg, model, symbol, timeframe, warm, val, initial_cash=initial_cash)
                s = score_curve(curve, len(pf_val.trades), args.metric, bars_per_year)
                if best is None or s > best["score"]:
                    best = {"h": h, "lo": lo, "ex": ex, "score": s,
                            "val_trades": len(pf_val.trades)}

        if best is None or best["score"] == float("-inf"):
            # Hiçbir kombinasyon iç-val'de işlem üretmedi: ilkine düş, not düş
            best = {"h": horizons[0], "lo": thresholds[0][0], "ex": thresholds[0][1],
                    "score": float("nan"), "val_trades": 0}
            logger.warning("Pencere %d: iç-val'de hiçbir kombinasyon işlem üretmedi", i)

        # ---------- DIŞ ADIM: kazanan parametrelerle dokunulmamış OOS ----------
        X, y = build_dataset(df.iloc[t0:t1], horizon=best["h"], threshold=args.label_threshold)
        model = make_model()
        model.fit(X, y)

        cfg = with_thresholds(config, best["lo"], best["ex"], time_stop_bars=best["h"])
        warm = candles[s0 - warmup_bars:s0]
        test = candles[s0:s1]
        pf_oos, curve = await run_window(
            cfg, model, symbol, timeframe, warm, test, initial_cash=carry)

        start_dt = pd.to_datetime(test[0]["ts"], unit="ms").date()
        end_dt = pd.to_datetime(test[-1]["ts"], unit="ms").date()
        pnl = curve[-1] - carry
        rows.append({
            "no": i, "start": start_dt, "end": end_dt,
            "h": best["h"], "lo": best["lo"], "ex": best["ex"],
            "val_score": best["score"],
            "trades": len(pf_oos.trades), "pnl": pnl, "pct": 100 * pnl / carry,
            "bh_pct": 100 * (test[-1]["close"] / test[0]["close"] - 1),
        })
        logger.info("Pencere %d/%d | seçim: h=%d eşik=%.2f/%.2f (val %s=%.2f) | OOS: %d işlem, PnL %+.2f (%%%+.2f)",
                    i, len(windows), best["h"], best["lo"], best["ex"],
                    args.metric, best["score"], len(pf_oos.trades), pnl, rows[-1]["pct"])

        all_trades.extend(pf_oos.trades)
        total_fees += pf_oos.total_fees
        stitched.extend(curve)
        carry = curve[-1]

    print_gs_report(args, symbol, timeframe, windows, rows, all_trades,
                    stitched, total_fees, initial_cash, candles)


def print_gs_report(args, symbol, timeframe, windows, rows, trades,
                    equity, total_fees, initial_cash, candles) -> None:
    final = equity[-1] if equity else initial_cash
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = 100 * len(wins) / len(trades) if trades else 0.0
    net = final - initial_cash
    strat_total = 100 * (final / initial_cash - 1)
    max_dd = _max_drawdown(equity) * 100 if equity else 0.0
    loss_sum = sum(t["pnl"] for t in losses)
    pf = (sum(t["pnl"] for t in wins) / abs(loss_sum)) if loss_sum else float("inf")
    profitable = sum(1 for r in rows if r["pnl"] > 0)

    first_test = candles[windows[0][2]]
    last_test = candles[windows[-1][3] - 1]
    bh_total = 100 * (last_test["close"] / first_test["close"] - 1)

    # En sık seçilen kombinasyon (parametre kararlılığı göstergesi)
    combos = [(r["h"], r["lo"], r["ex"]) for r in rows]
    top_combo = max(set(combos), key=combos.count)
    stability = combos.count(top_combo)

    w = 78
    print()
    print("=" * w)
    print(f"  NESTED GRID-SEARCH RAPORU (OOS)  |  {symbol}  {timeframe}  |  metrik={args.metric}")
    print("=" * w)
    print(f"  Grid                : horizon={args.horizons}  eşikler={args.thresholds}")
    print(f"  Pencere yapısı      : train {args.train_days}g (içinde %70/%30 val) + test {args.test_days}g")
    print()
    print("  --- Pencere bazında: iç-val seçimi -> dokunulmamış OOS sonucu ---")
    print(f"  {'#':>2}  {'Dönem':<25} {'h':>3} {'Eşik':>10} {'ValSkor':>8} {'İşl':>4} {'OOS PnL':>11} {'OOS%':>8} {'B&H%':>8}")
    for r in rows:
        print(f"  {r['no']:>2}  {str(r['start'])} -> {str(r['end']):<12} {r['h']:>3} "
              f"{r['lo']:.2f}/{r['ex']:.2f} {r['val_score']:>8.2f} {r['trades']:>4} "
              f"{r['pnl']:>+11,.2f} {r['pct']:>+7.2f}% {r['bh_pct']:>+7.2f}%")
    print()
    print("  --- Birleşik (stitched) OOS sonuçları ---")
    print(f"  Başlangıç -> Bitiş  : {initial_cash:,.2f} -> {final:,.2f} USDT")
    print(f"  Toplam işlem        : {len(trades)} kapanan ({len(wins)} kazanç / {len(losses)} kayıp)")
    print(f"  Kazanma oranı       : %{win_rate:.1f}")
    print(f"  Profit factor       : {pf:.2f}" if trades else "  Profit factor       : -")
    print(f"  Toplam komisyon     : {total_fees:>12,.2f} USDT")
    print(f"  NET PnL             : {net:>+12,.2f} USDT  (%{strat_total:+.2f})")
    print(f"  Max drawdown        : %{max_dd:.2f}")
    print(f"  Kârlı pencere       : {profitable}/{len(rows)}")
    print(f"  Parametre kararlılığı: en sık seçim h={top_combo[0]} {top_combo[1]:.2f}/{top_combo[2]:.2f} "
          f"({stability}/{len(rows)} pencere)")
    print("-" * w)
    print(f"  Al-ve-tut kıyası    : %{bh_total:+.2f}  "
          f"({'strateji ÖNDE' if strat_total > bh_total else 'al-ve-tut ÖNDE'})")
    print("=" * w)
    print("  Not: Parametreler her pencerede YALNIZ iç-validation ile seçildi;")
    print("  OOS pencereleri seçime katılmadı (nested CV). Bu tablo dürüsttür.")
    if stability < max(2, len(rows) // 3):
        print("  UYARI: Seçilen parametreler pencereden pencereye çok değişiyor -")
        print("  istikrarsız seçim genellikle gerçek bir edge olmadığına işarettir.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward içinde nested grid-search")
    parser.add_argument("--days", type=int, default=540)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--horizons", default="6,12,24,48",
                        help="Taranacak etiket ufukları (bar), örn: 6,12,24,48")
    parser.add_argument("--thresholds", default="0.60:0.40,0.65:0.35,0.70:0.30",
                        help="Giriş:çıkış eşik çiftleri, örn: 0.60:0.40,0.65:0.35")
    parser.add_argument("--metric", choices=["sharpe", "pnl"], default="sharpe",
                        help="İç-validation seçim metriği (NET, komisyon sonrası)")
    parser.add_argument("--label-threshold", type=float, default=0.0,
                        help="Pozitif etiket için asgari getiri")
    parser.add_argument("--funding", action="store_true",
                        help="Funding rate özelliklerini ekle (perpetual verisi)")
    parser.add_argument("--orderbook", action="store_true",
                        help="Order book (OBI) özelliklerini ekle (toplanan L2 verisi)")
    parser.add_argument("--maker", action="store_true",
                        help="Maker yürütme modeli: limit giriş + maker TP (paper)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_grid_search(args))


if __name__ == "__main__":
    main()
