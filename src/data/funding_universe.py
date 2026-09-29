"""
FUNDING EVRENİ İNDİRİCİ - çoklu-perp carry araştırmasının veri hattı.

Çalıştırma:
    python -m src.data.funding_universe --top 80 --days 400

Ne yapar: Binance USDT-perp evrenini 24s hacme göre sıralar, en likit
`top` sembolün funding geçmişini indirir (sembol başına cache; mevcutsa
atlar). Çıktılar data/cache/binanceusdm_funding_*.csv olarak birikir ve
funding-harvest backtest'inin girdisidir.

Neden: BTC carry'si mütevazı (~%5/yıl nominal) ama funding AŞIRILIKLARI
alt-perp'lerde yaşar (anlık taramada %10+/yıl ödeyen 82 perp görüldü).
Cross-section backtest için TÜM adayların tarihçesi gerekir - "bugünün
en iyileri"ni geçmişe götürmek look-ahead olur.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from ..utils.logger import setup_logging
from .derivatives import _fetch_with_retry, _futures_exchange

logger = logging.getLogger("funding_universe")

CACHE_DIR = Path("data/cache")


async def download_universe(top: int, days: int) -> None:
    exchange = _futures_exchange("binance")
    try:
        await exchange.load_markets()
        tickers = await exchange.fetch_tickers()
        # USDT-margined perp'leri 24s ciroya göre sırala
        cands = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT:USDT"):
                continue
            qv = t.get("quoteVolume") or 0
            cands.append((sym, qv))
        cands.sort(key=lambda x: -x[1])
        universe = [s for s, _ in cands[:top]]
        logger.info("Evren: %d sembol (24s ciro sıralı, ilk: %s)", len(universe), universe[:5])

        done = skipped = failed = 0
        for i, fsym in enumerate(universe, 1):
            safe = fsym.replace("/", "").replace(":", "")
            cache_file = CACHE_DIR / f"binanceusdm_funding_{safe}_{days}d.csv"
            if cache_file.exists():
                skipped += 1
                continue
            try:
                since = exchange.milliseconds() - days * 86_400_000
                rows: list[dict] = []
                while True:
                    s = since
                    batch = await _fetch_with_retry(
                        lambda: exchange.fetch_funding_rate_history(fsym, since=s, limit=1000))
                    if not batch:
                        break
                    rows.extend({"ts": r["timestamp"], "funding_rate": r["fundingRate"]}
                                for r in batch)
                    since = batch[-1]["timestamp"] + 1
                    if len(batch) < 1000:
                        break
                df = (pd.DataFrame(rows).drop_duplicates("ts")
                      .sort_values("ts").reset_index(drop=True))
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file, index=False)
                done += 1
                if i % 10 == 0:
                    logger.info("İlerleme: %d/%d (%s: %d kayıt)", i, len(universe), fsym, len(df))
            except Exception as exc:
                failed += 1
                logger.warning("%s indirilemedi: %s", fsym, exc)
        logger.info("BİTTİ: %d indirildi, %d önbellekte vardı, %d başarısız",
                    done, skipped, failed)
    finally:
        await exchange.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Perp evreni funding tarihçesi indirici")
    parser.add_argument("--top", type=int, default=80, help="En likit kaç perp (varsayılan 80)")
    parser.add_argument("--days", type=int, default=400)
    args = parser.parse_args()
    setup_logging("logs", "INFO")
    asyncio.run(download_universe(args.top, args.days))


if __name__ == "__main__":
    main()
