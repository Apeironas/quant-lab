"""Triple-barrier etiketleme birim testleri (sentetik senaryolar)."""
import numpy as np
import pandas as pd

from src.ml.features import build_labels_triple_barrier

# ATR ısınması için 20 bar düz seyir (range=2 -> ATR ~2), ardından senaryo
BASE_C = [100.0] * 20
BASE_H = [101.0] * 20
BASE_L = [99.0] * 20


def make_df(closes, highs, lows):
    n = len(closes)
    return pd.DataFrame({
        "ts": np.arange(n) * 14_400_000,
        "open": closes, "close": closes, "high": highs, "low": lows,
        "volume": np.ones(n),
    })


def test_take_profit_hit():
    """Fiyat üst bariyeri (entry + 2*ATR) aşarsa etiket 1."""
    df = make_df(BASE_C + [100, 100, 106, 106], BASE_H + [101, 101, 107, 107],
                 BASE_L + [99, 99, 105, 105])
    labels = build_labels_triple_barrier(df, horizon=10)
    assert labels.iloc[20] == 1.0


def test_stop_loss_hit():
    """Fiyat alt bariyerin (entry - 1*ATR) altına inerse etiket 0."""
    df = make_df(BASE_C + [100, 100, 94, 94], BASE_H + [101, 101, 95, 95],
                 BASE_L + [99, 99, 93, 93])
    labels = build_labels_triple_barrier(df, horizon=10)
    assert labels.iloc[20] == 0.0


def test_same_bar_conservative_sl_first():
    """Aynı barda iki bariyer birden değerse muhafazakâr kural: SL önce -> 0."""
    df = make_df(BASE_C + [100, 110, 110], BASE_H + [101, 120, 120],
                 BASE_L + [99, 90, 90])
    labels = build_labels_triple_barrier(df, horizon=10)
    assert labels.iloc[20] == 0.0


def test_vertical_barrier():
    """Hiçbir bariyer vurulmazsa dikey bariyerde kapanış > giriş ise 1."""
    df = make_df(BASE_C + [100.0, 100.5, 100.8, 101.0],
                 BASE_H + [100.6, 100.9, 101.0, 101.2],
                 BASE_L + [99.9, 100.2, 100.5, 100.8])
    labels = build_labels_triple_barrier(df, horizon=3)
    assert labels.iloc[20] == 1.0


def test_tail_is_nan():
    """Akıbeti veri içinde belirlenemeyen kuyruk satırları NaN kalmalı."""
    df = make_df(BASE_C + [100.0, 100.5, 100.8, 101.0],
                 BASE_H + [100.6, 100.9, 101.0, 101.2],
                 BASE_L + [99.9, 100.2, 100.5, 100.8])
    labels = build_labels_triple_barrier(df, horizon=3)
    assert np.isnan(labels.iloc[-1]) and np.isnan(labels.iloc[-2])
