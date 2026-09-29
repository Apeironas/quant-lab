"""Mikroyapı metrikleri birim testleri (sentetik emir defterleri)."""
import numpy as np
import pandas as pd

from src.ml.microstructure import (aggregate_to_bars, micro_price,
                                   order_book_imbalance, snapshot_metrics)

# Sentetik tahta: bids azalan, asks artan
BIDS = [[100.0, 3.0], [99.9, 2.0], [99.8, 1.0]]
ASKS = [[100.2, 1.0], [100.3, 1.0], [100.4, 1.0]]


def test_obi_buy_pressure():
    """Bid hacmi 6, ask hacmi 3 -> OBI = (6-3)/9 = +0.333 (alıcı baskısı)."""
    assert abs(order_book_imbalance(BIDS, ASKS, depth=3) - 1 / 3) < 1e-12


def test_obi_depth_slicing():
    """depth=1: (3-1)/4 = 0.5 - yalnız en iyi kademeler sayılmalı."""
    assert abs(order_book_imbalance(BIDS, ASKS, depth=1) - 0.5) < 1e-12


def test_obi_symmetric_book_is_zero():
    sym = [[100.0, 2.0]], [[100.2, 2.0]]
    assert order_book_imbalance(*sym, depth=1) == 0.0


def test_micro_price_leans_toward_thin_side():
    """
    bb=100 (vol 3), ba=100.2 (vol 1): alış tarafı kalın -> fiyat yukarı
    itilir -> microprice mid'in ÜSTÜNDE olmalı.
    micro = (100*1 + 100.2*3) / 4 = 100.15 > mid = 100.10
    """
    micro = micro_price(BIDS, ASKS)
    mid = (100.0 + 100.2) / 2
    assert abs(micro - 100.15) < 1e-12
    assert micro > mid


def test_snapshot_metrics_fields_and_signs():
    m = snapshot_metrics(BIDS, ASKS, depths=(1, 3))
    assert m["spread_bps"] > 0
    assert m["micro_basis_bps"] > 0          # alıcı baskısı -> pozitif basis
    assert m["obi_1"] == order_book_imbalance(BIDS, ASKS, 1)
    assert m["bid_vol_10"] == 6.0 and m["ask_vol_10"] == 3.0


def test_aggregate_to_bars_grouping():
    """Metrik akışı bar açılış ts'ine gruplanmalı; kolonlar tutarlı olmalı."""
    ts = np.array([0, 20_000, 59_000, 60_000, 90_000])  # ilk 3 -> bar 0, son 2 -> bar 60000
    df = pd.DataFrame({
        "ts": ts,
        "obi_10": [0.1, 0.3, 0.2, -0.4, -0.6],
        "micro_basis_bps": [1.0, 2.0, 3.0, -1.0, -2.0],
        "spread_bps": [0.5] * 5,
    })
    bars = aggregate_to_bars(df, bar_ms=60_000)
    assert list(bars["ts"]) == [0, 60_000]
    assert abs(bars["obi10_mean"].iloc[0] - 0.2) < 1e-12
    assert bars["obi10_last"].iloc[1] == -0.6
    assert bars["n_snapshots"].iloc[0] == 3
