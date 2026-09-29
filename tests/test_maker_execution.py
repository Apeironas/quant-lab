"""
Maker (limit emir) yürütme modeli entegrasyon testleri.

İki sentetik senaryo:
    1. Zigzag piyasa: limit giriş bir sonraki barda dolar (low < limit),
       pozisyon dikey bariyerle kapanır -> tam döngüler oluşur.
    2. Tek yönlü yükselen piyasa: fiyat limite asla geri inmez ->
       emir DOLMAZ, iptal olur, hiç işlem oluşmaz (muhafazakâr dolum kanıtı).
"""
import asyncio

import numpy as np

from src.backtest.walk_forward_runner import run_window

N_WARM, N_TEST = 350, 30


class AlwaysBullish:
    def predict_proba(self, X):
        return np.tile([0.30, 0.70], (len(X), 1))


def base_config(fill_window: int = 2):
    return {
        "mode": "paper",
        "fees": {"maker_pct": 0.02, "taker_pct": 0.05},
        "slippage": {"bps": 3},
        "execution": {"entry_mode": "maker", "fill_window_bars": fill_window},
        "risk": {"risk_per_trade_pct": 1.0, "max_position_pct": 20.0,
                 "max_open_positions": 1, "stop_loss_pct": 1.5,
                 "take_profit_pct": 3.0, "max_daily_loss_pct": 50.0},
        "portfolio": {"initial_cash": 10_000.0},
        "strategy": {"name": "ml", "ml": {
            "long_threshold": 0.60, "exit_threshold": 0.45,
            "min_bars": 310, "tp_atr": 2.0, "sl_atr": 1.0,
            "barrier_exit_only": True, "time_stop_bars": 5,
        }},
    }


def zigzag_candles(n, t0=0):
    """Salınan piyasa: limit her zaman bir sonraki barda dolar."""
    out = []
    for i in range(n):
        close = 100.0 + (0.2 if i % 2 else 0.0)
        out.append({"ts": (t0 + i) * 14_400_000, "open": close,
                    "high": close + 0.3, "low": close - 0.3,
                    "close": close, "volume": 1000.0})
    return out


def rising_candles(n, t0=0, start=100.0):
    """Tek yönlü yükseliş: fiyat önceki kapanışın altına asla inmez."""
    out = []
    for i in range(n):
        close = start + 0.1 * (t0 - N_WARM + i if t0 else i)
        out.append({"ts": (t0 + i) * 14_400_000, "open": close,
                    "high": close + 0.05, "low": close - 0.05,
                    "close": close, "volume": 1000.0})
    return out


def test_maker_fill_and_cycle():
    """Zigzag: limit dolar, pozisyon zaman bariyeriyle kapanır, döngü tekrarlar."""
    portfolio, _ = asyncio.run(run_window(
        base_config(), AlwaysBullish(), "TEST/USDT", "4h",
        zigzag_candles(N_WARM), zigzag_candles(N_TEST, t0=N_WARM),
        initial_cash=10_000.0))
    assert len(portfolio.trades) >= 2, "maker dolum döngüsü çalışmıyor"
    assert all("TIME_STOP" in t["reason"] for t in portfolio.trades
               if "zorunlu" not in t["reason"])
    # Giriş maker ücretiyle olmalı: tur ücreti, çift-taker turundan ucuz
    for t in portfolio.trades:
        notional = t["amount"] * t["entry_price"]
        assert t["fees"] < notional * 2 * 0.0005, "maker ücreti uygulanmamış görünüyor"


def test_maker_no_fill_in_runaway_market():
    """Yükselen piyasa: fiyat limite inmez -> dolum yok -> sıfır işlem, sıfır ücret."""
    warm = [{"ts": i * 14_400_000, "open": 100 + 0.1 * i, "high": 100 + 0.1 * i + 0.05,
             "low": 100 + 0.1 * i - 0.05, "close": 100 + 0.1 * i, "volume": 1000.0}
            for i in range(N_WARM)]
    test = [{"ts": (N_WARM + i) * 14_400_000, "open": 100 + 0.1 * (N_WARM + i),
             "high": 100 + 0.1 * (N_WARM + i) + 0.05,
             "low": 100 + 0.1 * (N_WARM + i) - 0.05,
             "close": 100 + 0.1 * (N_WARM + i), "volume": 1000.0}
            for i in range(N_TEST)]
    portfolio, curve = asyncio.run(run_window(
        base_config(), AlwaysBullish(), "TEST/USDT", "4h", warm, test,
        initial_cash=10_000.0))
    assert len(portfolio.trades) == 0, "dolmaması gereken limit dolmuş!"
    assert abs(curve[-1] - 10_000.0) < 1e-6, "işlemsiz dönemde equity değişmemeli"
