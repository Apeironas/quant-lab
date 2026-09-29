"""
CANLI MİKROYAPI DASHBOARD'U - toplayıcının izleme ekranı.

Çalıştırma (toplayıcıdan AYRI bir terminalde):
    python -m src.tools.live_dashboard
    python -m src.tools.live_dashboard --minutes 60 --refresh 2
    python -m src.tools.live_dashboard --once          # tek kare bas ve çık

Tasarım ilkesi: websocket'e DOKUNMAZ - yalnız collector'ın yazdığı CSV'nin
kuyruğunu okur. Veri hattına sıfır risk: izleyici çökse de toplama sürer.

Gösterilenler:
    - Fiyat (mid) ve OBI@10 sparkline'ları (son N dakika)
    - Anlık micro-basis, spread, bid/ask hacimleri
    - VERİ KALİTESİ: tazelik (son satır kaç sn önce?), boşluk sayısı,
      bugünkü satır sayısı - 2 haftalık birikimin asıl sigortası bu panel.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

OB_DIR = Path("data/orderbook")
BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 64) -> str:
    """Sayı dizisini blok karakterli mini grafiğe çevir."""
    vals = [v for v in values if v == v]  # NaN ayıkla
    if len(vals) < 2:
        return "(veri bekleniyor)"
    if len(vals) > width:  # genişliğe indirge (örnekle)
        step = len(vals) / width
        vals = [vals[int(i * step)] for i in range(width)]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1e-12
    return "".join(BLOCKS[int((v - lo) / span * (len(BLOCKS) - 1))] for v in vals)


def load_tail(symbol_safe: str, minutes: int) -> pd.DataFrame:
    """Son `minutes` dakikayı yükle (gerekirse dünün dosyasıyla birleştirir)."""
    files = sorted(OB_DIR.glob(f"{symbol_safe}_metrics_*.csv"))[-2:]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cutoff = time.time() * 1000 - minutes * 60_000
    return df[df["ts"] >= cutoff].reset_index(drop=True)


def count_gaps(ts: pd.Series, threshold_ms: int = 5_000) -> int:
    """5 sn'den uzun aralık = veri boşluğu (1 Hz akışta olmamalı)."""
    return int((ts.diff() > threshold_ms).sum()) if len(ts) > 1 else 0


def build_frame(symbol_safe: str, minutes: int) -> Panel:
    df = load_tail(symbol_safe, minutes)
    if df.empty:
        return Panel("Veri bulunamadı - toplayıcı çalışıyor mu?\n"
                     "Beklenen dosya: data/orderbook/*_metrics_*.csv",
                     title="Mikroyapı Dashboard", border_style="red")

    last = df.iloc[-1]
    lag_s = time.time() - last["ts"] / 1000
    gaps = count_gaps(df["ts"])
    obi = float(last["obi_10"])
    basis = float(last["micro_basis_bps"])

    # --- Anlık durum tablosu ---
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="bold")
    t.add_column()
    obi_style = "green" if obi > 0 else "red"
    t.add_row("Fiyat (mid):", f"{last['mid']:,.2f} USDT")
    t.add_row("OBI@10:", Text(f"{obi:+.3f}  {'ALICI baskın' if obi > 0 else 'SATICI baskın'}",
                              style=obi_style))
    t.add_row("Micro-basis:", Text(f"{basis:+.3f} bps", style="green" if basis > 0 else "red"))
    t.add_row("Spread:", f"{last['spread_bps']:.3f} bps")
    t.add_row("Hacim (10 kademe):", f"bid {last['bid_vol_10']:.2f} / ask {last['ask_vol_10']:.2f} BTC")

    # --- Sparkline'lar ---
    charts = Table.grid(padding=(0, 1))
    charts.add_column(justify="right", style="dim")
    charts.add_column()
    mid_lo, mid_hi = df["mid"].min(), df["mid"].max()
    charts.add_row("Fiyat", Text(sparkline(df["mid"]), style="cyan"))
    charts.add_row("", Text(f"min {mid_lo:,.0f}  max {mid_hi:,.0f}  (son {minutes} dk)", style="dim"))
    charts.add_row("OBI@10", Text(sparkline(df["obi_10"]), style="magenta"))
    charts.add_row("", Text("(-1 satıcı duvarı ... +1 alıcı duvarı)", style="dim"))
    charts.add_row("μ-basis", Text(sparkline(df["micro_basis_bps"]), style="yellow"))

    # --- Veri kalitesi paneli ---
    # Not: collector ~30 sn'de bir flush eder; 45 sn'ye kadar bayatlık NORMALDİR.
    fresh_style = "green" if lag_s < 45 else ("yellow" if lag_s < 120 else "bold red")
    fresh_note = ("  (buffer, normal)" if lag_s < 45
                  else ("  <- gecikiyor..." if lag_s < 120
                        else "  <- TOPLAYICIYI KONTROL ET!"))
    q = Table.grid(padding=(0, 2))
    q.add_column(justify="right", style="bold")
    q.add_column()
    q.add_row("Tazelik:", Text(f"son satır {lag_s:.1f} sn önce{fresh_note}",
                               style=fresh_style))
    q.add_row("Pencredeki satır:", f"{len(df):,} (beklenen ~{minutes * 60:,})")
    q.add_row("Boşluk (>5 sn):", Text(str(gaps), style="green" if gaps == 0 else "yellow"))

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    return Panel(
        Group(t, Text(""), charts, Text(""),
              Panel(q, title="Veri Kalitesi", border_style="dim")),
        title=f"📡 Mikroyapı Dashboard | {symbol_safe} | {now}",
        border_style="blue",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canlı OBI/fiyat izleme ekranı")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--minutes", type=int, default=30, help="Grafik penceresi (dk)")
    parser.add_argument("--refresh", type=float, default=1.0, help="Yenileme aralığı (sn)")
    parser.add_argument("--once", action="store_true", help="Tek kare bas ve çık (test)")
    args = parser.parse_args()

    symbol = args.symbol if ":" in args.symbol else f"{args.symbol}:USDT"
    safe = symbol.replace("/", "").replace(":", "")

    # Windows konsolunda blok karakterler/emoji için UTF-8 zorunlu
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    console = Console(legacy_windows=False)
    if args.once:
        console.print(build_frame(safe, args.minutes))
        return
    with Live(build_frame(safe, args.minutes), console=console,
              refresh_per_second=4, screen=False) as live:
        try:
            while True:
                time.sleep(args.refresh)
                live.update(build_frame(safe, args.minutes))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
