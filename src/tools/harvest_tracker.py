"""
FUNDING-HARVEST TAKİPÇİSİ - ileriye dönük paper doğrulama modülü.

Çalıştırma:
    python -m src.tools.harvest_tracker --once    # tek koşu (elle/dış zamanlayıcı)
    python -m src.tools.harvest_tracker --loop    # günlük döngü (watchdog ile bırak)

Strateji (backtest'le birebir aynı kurallar):
    - Evren: 24s ciroya göre en likit ~120 USDT-perp (CANLI seçim ->
      backtest'teki hayatta-kalma yanlılığı burada YOKTUR; bu modülün
      varlık sebebi tam olarak o yanlılıksız ileri doğrulamadır)
    - Sinyal: son 3 günün funding toplamı (yıllıklaştırılmış), eşik %10/yıl
    - Sepet: top-N (varsayılan 10) eşit ağırlık; delta-nötr varsayım
      (spot long + perp short) - yalnız funding geliri sayılır
    - Rebalance: 7 günde bir; değişen isim başına %0.3/bacak maliyet düşülür
    - Muhasebe: her koşuda, tutulan isimlerin son koşudan beri GERÇEKLEŞEN
      funding kayıtları toplanır -> data/harvest/harvest_log.csv

Bu bir PAPER takipçidir: emir göndermez, borsa hesabına dokunmaz.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ..data.derivatives import _fetch_with_retry, _futures_exchange
from ..utils.logger import setup_logging

logger = logging.getLogger("harvest")

OUT_DIR = Path("data/harvest")
STATE_FILE = OUT_DIR / "state.json"
LOG_FILE = OUT_DIR / "harvest_log.csv"

TOP_LIQUID = 120        # tarama evreni (gelir sürekliliği için geniş tutulur)
BASKET_N = 10           # sepet büyüklüğü (backtest: NET +%19.4/yıl, MaxDD %1.75)
MIN_ANN = 0.10          # sinyal ALT eşiği: trailing yıllık funding > %10
REBALANCE_DAYS = 7      # backtest'in net galibi
PER_SIDE_COST = 0.003   # %0.3/bacak-değişimi (muhafazakâr)
SIGNAL_WINDOW_MS = 3 * 86_400_000

# --- LİKİDİTE / TUZAK FİLTRELERİ ---
# Canlı takipte sepetin yarısı beş günde likit evrenden düştü: gerçekte
# çıkamama ve delist riski. Backtest bunu modellemiyor, filtreler kapatır.
MIN_VOLUME_USDT = 25_000_000  # likidite tabanı: 24s ciro < bu -> seçilmez (çıkış güvenliği)
MAX_ANN = 1.50                # ekstrem TAVAN: trailing > %150/yıl -> tuzak, elenir
                              # (en sert ortalamaya dönen + en kötü slippage'lı isimler)


def select_basket(trailing_ann: dict[str, float],
                  volumes: dict[str, float] | None = None,
                  n: int = BASKET_N, min_ann: float = MIN_ANN,
                  max_ann: float = MAX_ANN,
                  min_volume: float = MIN_VOLUME_USDT) -> list[str]:
    """
    Saf seçim kuralı (test edilebilir). Filtreler:
      - min_ann < trailing < max_ann  (düşük funding'i ATLA, ekstrem tuzağı ELE)
      - volumes verildiyse 24s ciro >= min_volume  (likidite tabanı)
    Kalanlardan en yüksek funding'li n isim.
    """
    elig = []
    for s, a in trailing_ann.items():
        if not (min_ann < a < max_ann):
            continue
        if volumes is not None and volumes.get(s, 0.0) < min_volume:
            continue
        elig.append((s, a))
    elig.sort(key=lambda x: -x[1])
    return [s for s, _ in elig[:n]]


def rebalance_cost(old: set[str], new: set[str], n: int = BASKET_N,
                   per_side: float = PER_SIDE_COST) -> float:
    """Sepet değişim maliyeti: değişen ağırlık başına bacak maliyeti (oran)."""
    changes = len(new - old) + len(old - new)
    return (changes / max(n, 1)) * per_side


def apply_negative_exit(holdings: list[str], trailing_ann: dict[str, float],
                        volumes: dict[str, float] | None = None,
                        n: int = BASKET_N) -> list[str]:
    """
    NEGATİF-ÇIKIŞ kuralı:
    funding'i negatife dönen VEYA likit evrenden düşen (trailing'de olmayan)
    isimleri 7 günü BEKLEMEDEN at, boşalan slotları taze uygun isimle doldur.
    Tarihsel panelde: saf 7g rebalance +%17.2/MaxDD%1.78 -> +negatif-çıkış
    +%19.0/MaxDD%0.94 (getiri artar, düşüş yarıya iner). Sadece bozulanı
    hedef aldığı için kör sıklaştırmanın komisyon kanamasına düşmez.

    Saf/test edilebilir. Döner: güncellenmiş sepet listesi.
    """
    # Hayatta kalanlar: evrende görünen VE funding'i negatif olmayan
    survivors = [s for s in holdings
                 if trailing_ann.get(s) is not None and trailing_ann[s] >= 0]
    # Boşalan slotları en yüksek funding'li taze isimlerle doldur
    if len(survivors) < n:
        for cand in select_basket(trailing_ann, volumes=volumes, n=n):
            if len(survivors) >= n:
                break
            if cand not in survivors:
                survivors.append(cand)
    return survivors


async def fetch_signals() -> tuple[dict[str, float], dict[str, list], dict[str, float]]:
    """
    Canlı evrenden trailing sinyaller + ham funding kayıtları + 24s ciro.
    Ciro, select_basket'in likidite tabanı filtresi için döndürülür.
    """
    exchange = _futures_exchange("binance")
    try:
        await exchange.load_markets()
        tickers = await exchange.fetch_tickers()
        cands = sorted(
            ((s, t.get("quoteVolume") or 0) for s, t in tickers.items()
             if s.endswith("/USDT:USDT")),
            key=lambda x: -x[1])[:TOP_LIQUID]

        now = exchange.milliseconds()
        trailing: dict[str, float] = {}
        records: dict[str, list] = {}
        volumes: dict[str, float] = {s: v for s, v in cands}
        for i, (fsym, _) in enumerate(cands, 1):
            try:
                batch = await _fetch_with_retry(
                    lambda: exchange.fetch_funding_rate_history(
                        fsym, since=now - SIGNAL_WINDOW_MS, limit=50), attempts=2)
            except Exception as exc:
                logger.debug("%s atlandı: %s", fsym, exc)
                continue
            if not batch:
                continue
            records[fsym] = [(r["timestamp"], r["fundingRate"]) for r in batch]
            total = sum(r["fundingRate"] for r in batch)
            trailing[fsym] = total * (365 * 86_400_000 / SIGNAL_WINDOW_MS)
            if i % 40 == 0:
                logger.info("Tarama: %d/%d", i, len(cands))
        return trailing, records, volumes
    finally:
        await exchange.close()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"holdings": [], "last_rebalance_ts": 0, "last_run_ts": 0, "cum_net": 0.0}


def save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


async def fetch_income_records(symbols: list[str], since_ms: int) -> dict[str, list]:
    """
    Tutulan isimlerin `since_ms`'ten beri GERÇEKLEŞEN funding kayıtları.

    Gerekçe: sinyal taraması yalnız 3 günlük pencere
    çeker. Koşular arası boşluk 3 günü aşarsa gelir EKSİK sayılır; maliyet ise
    tam yazılır -> defter stratejiyi haksız yere ekside gösterir. Bu fonksiyon
    boşluk durumunda devreye girip geliri son koşuya kadar geri toplar.
    (Gerçekte pozisyonlar borsada açık kalır ve funding tahsil edilir; doğru
    muhasebe budur.)
    """
    exchange = _futures_exchange("binance")
    out: dict[str, list] = {}
    try:
        for fsym in symbols:
            try:
                s = since_ms
                batch = await _fetch_with_retry(
                    lambda: exchange.fetch_funding_rate_history(
                        fsym, since=s, limit=1000), attempts=2)
            except Exception as exc:
                logger.debug("%s gelir geçmişi alınamadı: %s", fsym, exc)
                continue
            out[fsym] = [(r["timestamp"], r["fundingRate"]) for r in batch]
    finally:
        await exchange.close()
    return out


async def run_once() -> None:
    state = load_state()
    now_ms = int(time.time() * 1000)
    trailing, records, volumes = await fetch_signals()
    logger.info("Tarama bitti: %d sembolde sinyal", len(trailing))

    # --- 1) GELİR: son koşudan beri, MEVCUT sepetin gerçekleşen funding'i ---
    # BOŞLUĞA DAYANIKLI: koşular arası 3 günü aşarsa sinyal penceresi geliri
    # eksik sayar -> tutulan isimler için geçmişi son koşuya kadar geri çek.
    gap_days = ((now_ms - state["last_run_ts"]) / 86_400_000
                if state["last_run_ts"] else 0.0)
    income_records = records
    if state["holdings"] and gap_days > 2.5:
        logger.info("Boşluk %.1f gün -> gelir için genişletilmiş geçmiş çekiliyor",
                    gap_days)
        extended = await fetch_income_records(state["holdings"], state["last_run_ts"])
        income_records = {**records, **extended}

    income = 0.0
    n = max(len(state["holdings"]), 1)
    for sym in state["holdings"]:
        for ts, fr in income_records.get(sym, []):
            if ts > state["last_run_ts"]:
                income += fr / n

    # --- 2) REBALANCE (7 günde bir) ---
    cost = 0.0
    rebalanced = False
    days_since = (now_ms - state["last_rebalance_ts"]) / 86_400_000
    if days_since >= REBALANCE_DAYS or not state["holdings"]:
        new_basket = select_basket(trailing, volumes=volumes)
        cost = rebalance_cost(set(state["holdings"]), set(new_basket))
        state["holdings"] = new_basket
        state["last_rebalance_ts"] = now_ms
        rebalanced = True
    else:
        # Rebalance günü değil: negatif-çıkış kuralını uygula (bozulanı hemen at)
        refreshed = apply_negative_exit(state["holdings"], trailing, volumes=volumes)
        if set(refreshed) != set(state["holdings"]):
            cost = rebalance_cost(set(state["holdings"]), set(refreshed))
            changed = len(set(state["holdings"]) - set(refreshed))
            logger.info("NEGATİF-ÇIKIŞ: %d isim atıldı/yenilendi (funding negatif "
                        "veya evrenden düştü)", changed)
            state["holdings"] = refreshed

    net = income - cost
    state["cum_net"] += net
    state["last_run_ts"] = now_ms
    save_state(state)

    # --- 3) Deftere yaz ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if is_new:
            f.write("ts,datetime,income_pct,cost_pct,net_pct,cum_net_pct,rebalanced,holdings\n")
        f.write(f"{now_ms},{datetime.now(timezone.utc).isoformat()},"
                f"{income * 100:.5f},{cost * 100:.5f},{net * 100:.5f},"
                f"{state['cum_net'] * 100:.5f},{int(rebalanced)},"
                f"\"{';'.join(state['holdings'])}\"\n")

    # --- 4) Özet ---
    logger.info("=" * 64)
    logger.info("HARVEST | gelir: %+.4f%% | maliyet: %.4f%% | kümülatif NET: %+.3f%%",
                income * 100, cost * 100, state["cum_net"] * 100)
    if rebalanced:
        logger.info("REBALANCE yapıldı. Yeni sepet (trailing %%/yıl):")
        for s in state["holdings"]:
            logger.info("  %-22s %+7.1f%%", s.replace("/USDT:USDT", ""),
                        trailing.get(s, float("nan")) * 100)
    logger.info("=" * 64)


async def daily_loop() -> None:
    while True:
        try:
            await run_once()
        except Exception:
            logger.exception("Harvest koşusu hata verdi - 1 saat sonra tekrar")
            await asyncio.sleep(3600)
            continue
        await asyncio.sleep(24 * 3600)  # günde bir


def main() -> None:
    parser = argparse.ArgumentParser(description="Funding-harvest ileri doğrulama takipçisi")
    parser.add_argument("--loop", action="store_true", help="Günlük döngü (watchdog ile)")
    parser.add_argument("--once", action="store_true", help="Tek koşu")
    args = parser.parse_args()
    setup_logging("logs", "INFO")
    if args.loop:
        asyncio.run(daily_loop())
    else:
        asyncio.run(run_once())


if __name__ == "__main__":
    main()
