"""
MİKROYAPI METRİKLERİ - emir defteri (L2) anlık görüntülerinden özellikler.

Saf fonksiyonlar: girdi bids/asks listeleri ([fiyat, miktar], bids azalan,
asks artan sıralı), çıktı sayılar. Collector canlıda, araştırma kodu ise
kayıtlı snapshotlarda AYNI fonksiyonları çağırır (train/serve tutarlılığı,
features.py ile aynı ilke).

Kavramlar:
    OBI (Order Book Imbalance): tahtadaki alıcı/satıcı baskı dengesi.
        +1'e yakın = alış duvarı baskın, -1'e yakın = satış duvarı baskın.
    Micro-price (hacim ağırlıklı orta fiyat): en iyi kademe hacimleriyle
        ağırlıklanmış "adil" fiyat. Satış tarafı kalınsa mid'in altına kayar -
        fiyatın kısa vadede nereye "çekildiğinin" mikro göstergesi.
"""
from __future__ import annotations

import pandas as pd

Level = list  # [price, amount]


def order_book_imbalance(bids: list[Level], asks: list[Level], depth: int = 10) -> float:
    """
    OBI = (bid_hacmi - ask_hacmi) / (bid_hacmi + ask_hacmi), ilk `depth` kademede.
    Aralık [-1, +1]; 0 = dengeli tahta.
    """
    bid_vol = sum(a for _, a in bids[:depth])
    ask_vol = sum(a for _, a in asks[:depth])
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0


def micro_price(bids: list[Level], asks: list[Level]) -> float:
    """
    Hacim ağırlıklı orta fiyat (VWMP / microprice):
        (best_bid * ask_hacmi + best_ask * bid_hacmi) / (bid_hacmi + ask_hacmi)
    Karşı tarafın hacmiyle ağırlıklama bilinçli: ask tarafı kalınsa fiyat
    bid'e doğru itilir -> microprice mid'in altına düşer.
    """
    (bb, bv), (ba, av) = bids[0], asks[0]
    total = bv + av
    return (bb * av + ba * bv) / total if total > 0 else (bb + ba) / 2


def snapshot_metrics(bids: list[Level], asks: list[Level],
                     depths: tuple[int, ...] = (5, 10, 20)) -> dict:
    """
    Tek snapshot'tan tüm metrikler - collector'ın kayıt ettiği satır.
    micro_basis_bps: (microprice - mid) / mid, baz puan. İşaretli mikro baskı:
    pozitif = alıcılar fiyatı yukarı çekiyor.
    """
    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2
    micro = micro_price(bids, asks)

    metrics = {
        "mid": mid,
        "micro_price": micro,
        "micro_basis_bps": 10_000 * (micro - mid) / mid if mid > 0 else 0.0,
        "spread_bps": 10_000 * (best_ask - best_bid) / mid if mid > 0 else 0.0,
        "bid_vol_10": sum(a for _, a in bids[:10]),
        "ask_vol_10": sum(a for _, a in asks[:10]),
    }
    for d in depths:
        metrics[f"obi_{d}"] = order_book_imbalance(bids, asks, depth=d)
    return metrics


def aggregate_to_bars(metrics_df: pd.DataFrame, bar_ms: int = 60_000) -> pd.DataFrame:
    """
    1 Hz metrik akışını bar'lara (varsayılan 1m) agrega eder - araştırma köprüsü.

    Çıktı satırının ts'i BAR AÇILIŞ zamanıdır ve içerdiği istatistikler o barın
    [ts, ts+bar) aralığından gelir. Mum konvansiyonumuzla aynı: t satırının
    verisi, t mumu KAPANDIĞINDA bilinir (sızıntı yok, bkz. features.py kural 2).
    OHLCV ile birleştirme: ts üzerinde doğrudan join (merge_derivatives deseni).
    """
    df = metrics_df.copy()
    df["bar_ts"] = (df["ts"] // bar_ms) * bar_ms
    g = df.groupby("bar_ts")

    out = pd.DataFrame({
        "obi10_mean": g["obi_10"].mean(),       # barın ortalama baskı dengesi
        "obi10_last": g["obi_10"].last(),       # bar kapanışındaki denge
        "obi10_std": g["obi_10"].std(),         # baskı oynaklığı
        "micro_basis_mean": g["micro_basis_bps"].mean(),
        "spread_bps_mean": g["spread_bps"].mean(),
        "n_snapshots": g["ts"].count(),         # veri kalitesi göstergesi
    }).reset_index().rename(columns={"bar_ts": "ts"})
    return out
