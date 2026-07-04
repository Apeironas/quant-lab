"""
Merkezi loglama kurulumu.

İki hedefe yazar:
  1. Konsol - anlık takip için
  2. logs/bot_YYYY-MM-DD.log - kalıcı kayıt (her işlem, hata, gecikme buraya düşer)

Neden önemli: canlıda bir şey ters gittiğinde "dün gece 03:14'te ne oldu?"
sorusunun tek cevabı bu dosyalardır.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

FMT = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    # Windows konsolu varsayılan cp1252 ile Türkçe karakterlerde çöker;
    # stdout/stderr'i UTF-8'e zorla (hem loglar hem print için)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / f"bot_{date.today().isoformat()}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(FMT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # ccxt'nin kendi debug gürültüsünü kıs
    logging.getLogger("ccxt").setLevel(logging.WARNING)
