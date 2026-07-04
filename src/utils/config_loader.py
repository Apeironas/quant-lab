"""
Yapılandırma yükleyici: config.yaml + .env dosyasını birleştirir.

API anahtarları .env'de tutulur (git'e girmez), davranış ayarları
config.yaml'dadır (git'e girebilir). İkisini karıştırma.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(path: str = "config/config.yaml") -> dict:
    load_dotenv()  # .env dosyasını ortam değişkenlerine yükler

    with open(Path(path), encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)

    # API anahtarlarını config sözlüğüne enjekte et
    config["exchange"]["api_key"] = os.getenv("EXCHANGE_API_KEY", "")
    config["exchange"]["api_secret"] = os.getenv("EXCHANGE_API_SECRET", "")

    _validate(config)
    return config


def _validate(config: dict) -> None:
    """Tehlikeli/yanlış yapılandırmayı bot başlamadan yakala."""
    risk = config["risk"]
    if risk["risk_per_trade_pct"] > 5:
        raise ValueError("risk_per_trade_pct > %5: bu ayar hesabı hızla eritir. Bilerek yapıyorsan bu kontrolü kaldır.")
    if risk["stop_loss_pct"] <= 0 or risk["take_profit_pct"] <= 0:
        raise ValueError("stop_loss_pct ve take_profit_pct pozitif olmak zorunda - SL/TP'siz işlem yok.")
    if config["mode"] == "live" and not config["exchange"].get("testnet", True):
        # Gerçek para modu: anahtar yoksa hiç başlama
        if not config["exchange"]["api_key"]:
            raise ValueError("Canlı mod için .env dosyasında API anahtarı gerekli.")
