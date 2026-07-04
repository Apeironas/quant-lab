"""
TÜREV PİYASA VERİ HATTI - Funding Rate ve Open Interest (perpetual futures)

Neden: OHLCV "ne oldu"yu söyler; funding/OI ise piyasanın NASIL POZİSYONLANDIĞINI
söyler - OHLCV'de bulunmayan bilgi. Funding aşırı pozitifken kalabalık long'dur
(sıkışma riski), aşırı negatifken kalabalık short'tur.

VERİ KISITLARI (Binance, dürüst envanter):
    - Funding rate geçmişi: YILLARCA geriye gider (8 saatte bir kayıt) -> backtest'e uygun
    - Open Interest geçmişi: SADECE ~30 GÜN geriye gider (borsa kısıtı) ->
      uzun backtest'te KULLANILAMAZ; canlı/paper ve ileriye dönük biriktirme için altyapı hazır.

ZAMAN HİZALAMASI / SIZINTI KİLİDİ:
    merge_asof(direction="backward") ile her 1h mumuna, AÇILIŞ zamanına eşit
    veya ÖNCESİNDEKİ son funding kaydı yazılır. Örn. 08:00 funding'i ilk kez
    08:00 açılışlı muma düşer (o mumun kararı 09:00'da verilir - funding
    çoktan bilinir). Gelecekteki funding'e bakmak yapısal olarak imkansızdır;
    aynı garanti OI için de geçerlidir.
"""
from __future__ import annotations

import logging
from pathlib import Path

import ccxt.async_support as accxt
import pandas as pd

logger = logging.getLogger("derivatives")

CACHE_DIR = Path("data/cache")


def _futures_symbol(symbol: str) -> str:
    """Spot sembolü USDT-perpetual sembolüne çevir: BTC/USDT -> BTC/USDT:USDT"""
    return symbol if ":" in symbol else f"{symbol}:USDT"


def _futures_exchange(exchange_id: str) -> accxt.Exchange:
    """
    Türev uçları için futures'a özel borsa sınıfı (binance -> binanceusdm).
    Neden: genel 'binance' sınıfı load_markets'ta dev spot+futures listesini
    çeker ve zaman aşımına yatkındır; usdm sınıfı yalnız futures'ı yükler.
    """
    fid = "binanceusdm" if exchange_id == "binance" else exchange_id
    return getattr(accxt, fid)({"enableRateLimit": True, "timeout": 30_000})


async def _fetch_with_retry(coro_factory, attempts: int = 3, wait_s: float = 5.0):
    """Geçici ağ hatalarında (timeout vb.) birkaç kez yeniden dene."""
    import asyncio
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if attempt == attempts:
                raise
            logger.warning("İstek hatası (%s) - %d/%d, %.0f sn sonra tekrar",
                           type(exc).__name__, attempt, attempts, wait_s)
            await asyncio.sleep(wait_s)


async def download_funding_history(exchange_id: str, symbol: str, days: int) -> pd.DataFrame:
    """
    Funding rate geçmişini indir (sayfalı, önbellekli).
    Çıktı: DataFrame[ts (ms), funding_rate] - kronolojik, 8 saatte bir kayıt.
    """
    fsym = _futures_symbol(symbol)
    cache_file = CACHE_DIR / f"{exchange_id}_funding_{fsym.replace('/', '').replace(':', '')}_{days}d.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
        logger.info("Funding önbellekten: %s (%d kayıt)", cache_file, len(df))
        return df

    exchange = _futures_exchange(exchange_id)
    rows: list[dict] = []
    try:
        since = exchange.milliseconds() - days * 86_400_000
        while True:
            s = since
            batch = await _fetch_with_retry(
                lambda: exchange.fetch_funding_rate_history(fsym, since=s, limit=1000))
            if not batch:
                break
            rows.extend({"ts": r["timestamp"], "funding_rate": r["fundingRate"]} for r in batch)
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
    finally:
        await exchange.close()

    df = (pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    logger.info("Funding indirildi: %d kayıt (%s -> %s)", len(df),
                pd.to_datetime(df["ts"].iloc[0], unit="ms").date(),
                pd.to_datetime(df["ts"].iloc[-1], unit="ms").date())
    return df


async def download_open_interest_history(exchange_id: str, symbol: str,
                                         timeframe: str = "1h") -> pd.DataFrame:
    """
    Open Interest geçmişi - DİKKAT: Binance yalnız SON ~30 GÜNÜ verir.
    Uzun backtest için kullanma; canlı sinyal ve ileriye dönük veri
    biriktirme (periyodik çağırıp cache'e ekleme) için kullan.
    Çıktı: DataFrame[ts (ms), open_interest]
    """
    fsym = _futures_symbol(symbol)
    exchange = _futures_exchange(exchange_id)
    rows: list[dict] = []
    try:
        since = exchange.milliseconds() - 30 * 86_400_000  # borsa kısıtı: ~30 gün
        while True:
            s = since
            batch = await _fetch_with_retry(
                lambda: exchange.fetch_open_interest_history(
                    fsym, timeframe=timeframe, since=s, limit=500))
            if not batch:
                break
            rows.extend({"ts": r["timestamp"],
                         "open_interest": r.get("openInterestAmount") or r.get("openInterestValue")}
                        for r in batch)
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 500:
                break
    finally:
        await exchange.close()
    logger.warning("OI geçmişi: %d kayıt (Binance kısıtı gereği en fazla ~30 gün geriye)", len(rows))
    return pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def merge_derivatives(ohlcv_df: pd.DataFrame,
                      funding_df: pd.DataFrame | None = None,
                      oi_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Türev verilerini OHLCV'ye SIZINTISIZ olarak yapıştırır (as-of backward join).

    Her mum satırı, kendi AÇILIŞ zamanına <= olan son kaydı alır. Böylece
    t anındaki satırda yalnız t'de zaten yayınlanmış funding/OI bulunur.
    Kolonlar mum sözlükleriyle birlikte tüm sisteme (MarketEvent -> strateji
    tamponu) otomatik taşınır; features.py kolonları görürse ilgili
    özellikleri üretir, görmezse sessizce atlar.
    """
    out = ohlcv_df.sort_values("ts").reset_index(drop=True)
    for extra_df, col in ((funding_df, "funding_rate"), (oi_df, "open_interest")):
        if extra_df is None or extra_df.empty:
            continue
        extra = extra_df[["ts", col]].dropna().sort_values("ts")
        merged = pd.merge_asof(out[["ts"]], extra, on="ts", direction="backward")
        out[col] = merged[col].to_numpy()
    return out
