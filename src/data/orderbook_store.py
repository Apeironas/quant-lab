"""
EMİR DEFTERİ DEPOSU - toplanan L2 metriklerini araştırmaya bağlayan köprü.

Akış:
    collector (1 Hz CSV'ler) -> load_orderbook_metrics() -> aggregate_to_bars()
    -> merge_orderbook(ohlcv) -> features.py OBI bloğu (kolon-varsa-aktif)

ZAMAN HİZALAMASI / SIZINTI:
    aggregate_to_bars çıktısında ts = BAR AÇILIŞI ve istatistikler o barın
    [ts, ts+bar) aralığından gelir - mumun kendi OHLCV'siyle aynı konvansiyon
    (t satırının verisi t mumu kapandığında bilinir; features.py kural 2).
    Bu yüzden birleştirme merge_asof değil, ts üzerinde DOĞRUDAN eşleşmedir.

KAPSAM NOTU: OBI kolonları yalnız veri toplamanın açık olduğu dönemde
doludur; öncesi NaN kalır ve build_dataset o satırları atar.
Yani --days'i geniş vermek zarar vermez, etkin veri = toplanan dönemdir.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..ml.microstructure import aggregate_to_bars

logger = logging.getLogger("ob_store")

OB_DIR = Path("data/orderbook")


def load_orderbook_metrics(symbol: str) -> pd.DataFrame:
    """Sembolün tüm günlük metrik CSV'lerini yükle (kronolojik, tekilleştirilmiş)."""
    fsym = symbol if ":" in symbol else f"{symbol}:USDT"
    safe = fsym.replace("/", "").replace(":", "")
    files = sorted(OB_DIR.glob(f"{safe}_metrics_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"{OB_DIR} altında {safe}_metrics_*.csv bulunamadı - collector çalıştı mı?")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    logger.info("Order book metrikleri: %d satır (%s -> %s)", len(df),
                pd.to_datetime(df["ts"].iloc[0], unit="ms").date(),
                pd.to_datetime(df["ts"].iloc[-1], unit="ms").date())
    return df


def merge_orderbook(ohlcv_df: pd.DataFrame, symbol: str, bar_ms: int,
                    min_snapshots_ratio: float = 0.5) -> pd.DataFrame:
    """
    OHLCV mumlarına OBI bar istatistiklerini ekle.

    min_snapshots_ratio: bir barın geçerli sayılması için gereken asgari
    1 Hz örnek doluluğu (0.5 = barın en az yarısı kayıtlı olmalı). Veri
    boşluğuna denk gelen barların yarım-yamalak istatistikleri modele
    gürültü olarak girmesin diye NaN'lanır (build_dataset onları atar).
    """
    metrics = load_orderbook_metrics(symbol)
    bars = aggregate_to_bars(metrics, bar_ms=bar_ms)

    # Doluluk filtresi: eksik barların istatistikleri güvenilmez -> NaN
    expected = bar_ms / 1000  # 1 Hz örnekleme varsayımı
    thin = bars["n_snapshots"] < expected * min_snapshots_ratio
    ob_cols = [c for c in bars.columns if c not in ("ts", "n_snapshots")]
    bars.loc[thin, ob_cols] = float("nan")
    if int(thin.sum()):
        logger.info("Doluluk filtresi: %d/%d bar NaN'landı (kapsama < %%%d)",
                    int(thin.sum()), len(bars), int(100 * min_snapshots_ratio))

    out = ohlcv_df.merge(bars.drop(columns=["n_snapshots"]), on="ts", how="left")
    covered = out["obi10_mean"].notna().sum()
    logger.info("OBI birleştirme: %d/%d mumda OBI verisi var", covered, len(out))
    return out
