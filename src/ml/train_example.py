"""
ÖRNEK MODEL EĞİTİMİ - MLStrategy'ye takılacak modeli üretir.

Çalıştırma (proje kökünden):
    python -m src.ml.train_example --days 365 --timeframe 1h
    python -m src.ml.train_example --days 365 --timeframe 1h --horizon 12

Ne yapar:
    1. Geçmiş veriyi indirir (backtest ile aynı önbelleği kullanır)
    2. features.build_dataset ile (X, y) üretir - strateji ile AYNI özellikler
    3. ZAMAN SIRALI böler: ilk %70 train, aradan `horizon` bar EMBARGO, son %30 test
       (shuffle YOK - zaman serisinde karıştırmalı split sızıntının ta kendisidir;
        embargo, train'in son etiketlerinin test dönemine taşan geleceği
        görmesini engeller)
    4. Holdout metriklerini basar (AUC ~0.50 = modelin hiçbir öngörüsü yok)
    5. Modeli TÜM veriyle yeniden eğitip models/ml_model.joblib'e kaydeder
       (değerlendirme dürüst holdout'ta yapıldı; dağıtıma giden model ise
        eldeki tüm bilgiyi kullanır - standart pratik)

Kendi modelini kullanmak istersen: sadece `make_model()` fonksiyonunu değiştir
(LightGBM/XGBoost sklearn-wrapper'ları dahil predict_proba'sı olan her şey olur)
veya tamamen kendi pipeline'ınla eğitip joblib.dump ile kaydet.

NOT: Bu script tek bir train/test bölmesidir - hızlı fikir elemek içindir.
Nihai karar için sıradaki modül olan Walk-Forward analizi kullanılacak
(çok pencereli, kayan eğitim). Buradaki build_dataset zaten pencere dilimi
kabul ettiği için o modül bu altyapının üstüne doğrudan oturacak.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ..backtest.backtest_runner import download_ohlcv
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging
from .features import build_dataset

logger = logging.getLogger("train")

MODELS_DIR = Path("models")


def make_model():
    """
    >>> KENDİ MODELİNİ BURADA TANIMLA <<<
    Örnekler:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.03)

        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=4)
    """
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        early_stopping=False, random_state=42,
    )


async def train(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], "INFO")

    symbol = args.symbol or config["market_data"]["symbols"][0]
    df = await download_ohlcv(config["exchange"]["id"], symbol, args.timeframe, args.days)

    X, y = build_dataset(df, horizon=args.horizon, threshold=args.threshold)
    logger.info("Veri seti: %d örnek, %d özellik | pozitif oran=%.3f",
                len(X), X.shape[1], y.mean())

    # --- Zaman sıralı bölme + embargo ---
    split = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split + args.horizon:], y.iloc[split + args.horizon:]

    model = make_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, proba >= 0.5)

    print()
    print("=" * 58)
    print(f"  EĞİTİM RAPORU  |  {symbol} {args.timeframe}  |  horizon={args.horizon} bar")
    print("=" * 58)
    print(f"  Train örnek         : {len(X_train):,}")
    print(f"  Test örnek (holdout): {len(X_test):,}  (embargo={args.horizon} bar)")
    print(f"  Holdout AUC         : {auc:.4f}")
    print(f"  Holdout accuracy    : {acc:.4f}")
    print("-" * 58)
    if auc < 0.52:
        print("  UYARI: AUC ~0.50 -> model yazı-tura'dan farksız. Bu modeli")
        print("  canlıda kullanmak anlamsız; özellik/horizon/hedef değiştir.")
    elif auc < 0.55:
        print("  Not: Zayıf ama sıfır olmayan sinyal. Komisyonları yenip")
        print("  yenmediğini SADECE backtest gösterir - AUC tek başına yetmez.")
    else:
        print("  Not: Umut verici görünüyor - ama önce farklı dönemlerde")
        print("  backtest + walk-forward ile doğrula (overfitting şüphesi hep var).")
    print("=" * 58)

    # --- Dağıtım modeli: tüm veriyle yeniden eğit, kaydet ---
    final_model = make_model()
    final_model.fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / "ml_model.joblib"
    joblib.dump(final_model, out_path)
    print(f"\n  Model kaydedildi: {out_path}")
    print(f"  Backtest için    : python -m src.backtest.backtest_runner "
          f"--strategy ml --days {args.days} --timeframe {args.timeframe}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MLStrategy için örnek model eğitimi")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--horizon", type=int, default=6,
                        help="Kaç bar sonrasının getirisi tahmin edilecek (varsayılan: 6)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Pozitif etiket için asgari getiri (varsayılan: 0 = yön tahmini)")
    args = parser.parse_args()
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
