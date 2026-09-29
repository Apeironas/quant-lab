"""
EMİR DEFTERİ (L2) VERİ TOPLAYICI - Bölüm 2'nin temel taşı.

L2 geçmişi satın alınamaz; BİRİKTİRİLİR. Bu script "her zaman açık" çalışacak
bağımsız bir süreçtir: bugün başlatılan her saat, gelecekteki backtest'in
hammaddesidir.

Çalıştırma (proje kökünden, ayrı bir terminalde açık bırak):
    python -m src.data.orderbook_collector
    python -m src.data.orderbook_collector --symbol ETH/USDT --depth 20 --interval 1.0

Ne yapar:
    - Binance USDT-Perpetual websocket'inden ilk `depth` kademe emir defteri
    - Her `interval` saniyede bir METRİK satırı yazar (OBI@5/10/20,
      micro-price, spread...) -> data/orderbook/SYMBOL_metrics_YYYY-MM-DD.csv
    - Her `raw-interval` saniyede bir HAM top-N snapshot yazar (ileride yeni
      metrik icat edilirse yeniden hesaplama sigortası)
      -> data/orderbook/SYMBOL_raw_YYYY-MM-DD.csv
    - Günlük dosya rotasyonu; buffer'lı yazım (çökme en fazla ~30 sn veri
      kaybettirir); ağ kopunca otomatik yeniden bağlanır.

Boyut beklentisi (BTC, varsayılan ayarlar): metrik ~6 MB/gün, ham ~15 MB/gün.
API anahtarı GEREKMEZ (halka açık akış).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt.pro as ccxtpro

from ..ml.microstructure import snapshot_metrics
from ..utils.logger import setup_logging

logger = logging.getLogger("ob_collector")

OUT_DIR = Path("data/orderbook")
FLUSH_EVERY = 30  # kaç satırda bir diske yazılsın (metrik satırı ~= saniye)


class CsvAppender:
    """Günlük dönen, buffer'lı CSV yazıcı. Başlığı dosya yokken yazar."""

    def __init__(self, prefix: str, columns: list[str],
                 flush_every: int = FLUSH_EVERY) -> None:
        self.prefix = prefix
        self.columns = columns
        self.flush_every = flush_every
        self.buffer: list[list] = []

    def add(self, ts_ms: int, row: list) -> None:
        self.buffer.append([ts_ms, *row])
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        # Rotasyon: dosya adı, buffer'daki İLK satırın gününe göre seçilir;
        # gün ortası geçişlerde birkaç satırlık taşma önemsizdir.
        day = datetime.fromtimestamp(self.buffer[0][0] / 1000, tz=timezone.utc).date()
        path = OUT_DIR / f"{self.prefix}_{day.isoformat()}.csv"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(",".join(["ts", *self.columns]) + "\n")
            for row in self.buffer:
                f.write(",".join(_fmt(v) for v in row) + "\n")
        self.buffer.clear()


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


METRIC_COLS = ["mid", "micro_price", "micro_basis_bps", "spread_bps",
               "bid_vol_10", "ask_vol_10", "obi_5", "obi_10", "obi_20"]


def _raw_cols(depth: int) -> list[str]:
    cols = []
    for i in range(depth):
        cols += [f"bid_p{i}", f"bid_v{i}"]
    for i in range(depth):
        cols += [f"ask_p{i}", f"ask_v{i}"]
    return cols


async def collect(args: argparse.Namespace) -> None:
    symbol = args.symbol if ":" in args.symbol else f"{args.symbol}:USDT"
    safe_name = symbol.replace("/", "").replace(":", "")

    metrics_out = CsvAppender(f"{safe_name}_metrics", METRIC_COLS)
    # Ham kayıt seyrek aktığı için (5 sn'de 1) daha sık flush: kayıp <= ~30 sn
    raw_out = CsvAppender(f"{safe_name}_raw", _raw_cols(args.depth), flush_every=6)

    exchange = ccxtpro.binanceusdm({"enableRateLimit": True})
    logger.info("Toplayıcı başlıyor: %s | derinlik=%d | metrik=%.1fs | ham=%.1fs | çıktı=%s",
                symbol, args.depth, args.interval, args.raw_interval, OUT_DIR)

    last_metric = last_raw = last_report = 0.0
    n_rows = 0
    try:
        while True:
            try:
                ob = await exchange.watch_order_book(symbol, limit=args.depth)
                now = time.time()
                bids, asks = ob.get("bids") or [], ob.get("asks") or []
                if not bids or not asks:
                    continue
                ts_ms = int(ob.get("timestamp") or now * 1000)

                # --- 1 Hz metrik örneklemesi ---
                if now - last_metric >= args.interval:
                    last_metric = now
                    m = snapshot_metrics(bids, asks)
                    metrics_out.add(ts_ms, [m[c] for c in METRIC_COLS])
                    n_rows += 1

                # --- Ham snapshot (yeniden hesaplama sigortası) ---
                if args.raw_interval > 0 and now - last_raw >= args.raw_interval:
                    last_raw = now
                    row: list = []
                    for i in range(args.depth):
                        row += list(bids[i][:2]) if i < len(bids) else [0.0, 0.0]
                    for i in range(args.depth):
                        row += list(asks[i][:2]) if i < len(asks) else [0.0, 0.0]
                    raw_out.add(ts_ms, row)

                # --- Saatlik yaşam raporu ---
                if now - last_report >= 3600:
                    last_report = now
                    logger.info("Yaşıyor: %d metrik satırı toplandı | son OBI@10=%.3f spread=%.2fbps",
                                n_rows, m.get("obi_10", 0), m.get("spread_bps", 0))

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Ağ kopması / borsa gürültüsü: veri hattı ASLA ölmez
                logger.warning("Akış hatası: %s - 10 sn sonra yeniden bağlanılıyor", exc)
                await asyncio.sleep(10)
    finally:
        metrics_out.flush()
        raw_out.flush()
        await exchange.close()
        logger.info("Toplayıcı kapandı. Toplam %d metrik satırı.", n_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 emir defteri veri toplayıcı")
    parser.add_argument("--symbol", default="BTC/USDT", help="Parite (varsayılan: BTC/USDT)")
    parser.add_argument("--depth", type=int, default=20, choices=[5, 10, 20],
                        help="Kademe sayısı (Binance kısmi akış: 5/10/20)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Metrik örnekleme aralığı, saniye (varsayılan: 1.0)")
    parser.add_argument("--raw-interval", type=float, default=5.0,
                        help="Ham snapshot aralığı, saniye (0 = ham kayıt kapalı)")
    args = parser.parse_args()

    setup_logging("logs", "INFO")
    try:
        asyncio.run(collect(args))
    except KeyboardInterrupt:
        print("\nToplayıcı kullanıcı tarafından durduruldu (buffer diske yazıldı).")


if __name__ == "__main__":
    main()
