"""
Sızıntı (look-ahead bias) testleri - sistemin en kritik güvencesi.

İlke: t satırının özellikleri yalnız t ve öncesinden hesaplanıyorsa, verinin
GELECEĞİNİ silmek geçmiş satırların özelliklerini DEĞİŞTİRMEMELİDİR.
Tek bir hücre bile değişirse gelecek bilgisi sızıyor demektir.
"""
import numpy as np
import pandas as pd

from src.data.derivatives import merge_derivatives
from src.ml.features import MAX_LOOKBACK, build_features


def synthetic_ohlcv(n=800, seed=42, bar_ms=14_400_000):
    """Rastgele yürüyüş OHLCV (4h taban) - ağ gerektirmeyen test verisi."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    return pd.DataFrame({
        "ts": np.arange(n) * bar_ms,
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.uniform(100, 1000, n),
    })


def _assert_no_leak(df, cuts):
    full = build_features(df)
    for cut in cuts:
        part = build_features(df.iloc[:cut].copy())
        a = full.iloc[:cut].to_numpy(dtype=float)
        b = part.to_numpy(dtype=float)
        identical = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        assert identical.all(), f"SIZINTI: kesim t={cut}, {int((~identical).sum())} hücre farklı"


def test_features_no_lookahead_4h_base():
    """4h taban: bar sınırına denk gelen/gelmeyen kesimlerde özellikler sabit."""
    _assert_no_leak(synthetic_ohlcv(), cuts=(400, 557, 701))


def test_features_no_lookahead_1h_base():
    """1h taban (MTF 4h+1d yolu) için aynı garanti."""
    _assert_no_leak(synthetic_ohlcv(n=900, bar_ms=3_600_000), cuts=(500, 683))


def test_features_no_lookahead_with_funding():
    """Funding kolonu eklendiğinde de sızıntı olmamalı."""
    df = synthetic_ohlcv()
    rng = np.random.default_rng(7)
    funding = pd.DataFrame({
        "ts": np.arange(0, df["ts"].iloc[-1], 8 * 3_600_000),  # 8 saatte bir
    })
    funding["funding_rate"] = rng.normal(0.0001, 0.0002, len(funding))
    merged = merge_derivatives(df, funding_df=funding)
    _assert_no_leak(merged, cuts=(400, 601))


def test_funding_merge_alignment():
    """As-of join: mum, yalnız açılışına <= olan son funding kaydını görmeli."""
    df = synthetic_ohlcv(n=10, bar_ms=14_400_000)          # ts: 0,4h,8h,...
    funding = pd.DataFrame({"ts": [8 * 3_600_000],          # tek kayıt: 08:00
                            "funding_rate": [0.005]})
    merged = merge_derivatives(df, funding_df=funding)
    assert np.isnan(merged["funding_rate"].iloc[0])          # 00:00 mumu: henüz yok
    assert np.isnan(merged["funding_rate"].iloc[1])          # 04:00 mumu: henüz yok
    assert merged["funding_rate"].iloc[2] == 0.005           # 08:00 mumu: ilk görüş


def test_warmup_within_max_lookback():
    """İlk NaN'sız satır MAX_LOOKBACK sınırının içinde olmalı."""
    full = build_features(synthetic_ohlcv())
    first_valid = int(full.notna().all(axis=1).idxmax())
    assert first_valid <= MAX_LOOKBACK
