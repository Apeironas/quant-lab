"""
v2.1 bariyer-sadık işlem yönetimi entegrasyon testi.

Sahte bir "hep %70 eminim" modeli + bariyerlerin vurulamayacağı düz sentetik
piyasa ile tam olay zinciri (sinyal -> risk -> emir -> fill -> bariyer izleme)
uçtan uca doğrulanır:
    1. Olasılık-bazlı erken çıkış (proba-exit) devre dışı
    2. Dikey bariyer (time stop) tam N bar sonra mum kapanışından çıkar
    3. FILL geri beslemesi: pozisyon kapanınca strateji yeniden girebilir
"""
import asyncio

import numpy as np

from src.backtest.walk_forward_runner import run_window

N_WARM, N_TEST = 350, 30
TIME_STOP = 5


def make_candles(n, t0=0):
    out = []
    for i in range(n):
        close = 100.0 + (0.2 if i % 2 else 0.0)  # hafif zigzag: ATR>0, bariyer vurulmaz
        out.append({
            "ts": (t0 + i) * 14_400_000,
            "open": close, "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": 1000.0,
        })
    return out


class AlwaysBullish:
    """predict_proba her koşulda p(başarı)=0.70 döndürür."""
    def predict_proba(self, X):
        return np.tile([0.30, 0.70], (len(X), 1))


CONFIG = {
    "mode": "paper",
    "fees": {"maker_pct": 0.02, "taker_pct": 0.05},
    "slippage": {"bps": 3},
    "risk": {"risk_per_trade_pct": 1.0, "max_position_pct": 20.0,
             "max_open_positions": 1, "stop_loss_pct": 1.5,
             "take_profit_pct": 3.0, "max_daily_loss_pct": 50.0},
    "portfolio": {"initial_cash": 10_000.0},
    "strategy": {"name": "ml", "ml": {
        "long_threshold": 0.60, "exit_threshold": 0.45,
        "min_bars": 310, "tp_atr": 2.0, "sl_atr": 1.0,
        "barrier_exit_only": True, "time_stop_bars": TIME_STOP,
    }},
}


def test_barrier_faithful_trade_management():
    warm = make_candles(N_WARM)
    test = make_candles(N_TEST, t0=N_WARM)
    portfolio, _ = asyncio.run(run_window(
        CONFIG, AlwaysBullish(), "TEST/USDT", "4h", warm, test,
        initial_cash=10_000.0))

    reasons = [t["reason"] for t in portfolio.trades]
    # 1) Proba-exit tamamen kapalı
    assert not any("EXIT sinyali" in r for r in reasons)
    # 2) Tüm kapanışlar dikey bariyerden (SL/TP vurulamayacak şekilde kuruldu)
    assert all("TIME_STOP" in r for r in reasons if "zorunlu" not in r)
    # 3) Fill geri beslemesiyle yeniden giriş: 30 barda ~5 tam döngü
    assert len(portfolio.trades) >= 4
