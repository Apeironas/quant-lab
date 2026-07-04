"""
ÖZELLİK ÖNEMİ ANALİZİ - modelin beyninin içine bakan bağımsız araç.

Çalıştırma (proje kökünden):
    python -m src.ml.analyze_importance                       # dürüst mod (önerilen)
    python -m src.ml.analyze_importance --mode saved          # kayıtlı modeli incele
    python -m src.ml.analyze_importance --horizon 24 --days 540

İki mod:
    honest (varsayılan): Model, verinin ilk %70'iyle SIFIRDAN eğitilir;
        önem, hiç görmediği son %30 (embargo'lu) üzerinde ölçülür.
        "Hangi özellik GERÇEKTEN öngörüye katkı veriyor?" sorusunun cevabı.
    saved: models/ml_model.joblib (veya --model yolu) yüklenir ve aynı
        değerlendirme verisiyle incelenir. "Dağıtılmış model neye bakıyor?"
        sorusunun cevabı. DİKKAT: kayıtlı model tüm veriyle eğitildiyse
        değerlendirme kısmen in-sample'dır - rapor bunu belirtir.

İki önem ölçüsü:
    1. Native (feature_importances_): LightGBM/XGBoost/RandomForest sunar;
       sklearn HistGradientBoosting SUNMAZ. Varsa basılır.
    2. Permutation importance (her zaman): bir özelliğin sütunu rastgele
       karıştırılır, holdout AUC'deki düşüş ölçülür (n_repeats kez).
       Model-agnostiktir ve ağaç içi önemlerin split-sayısı yanlılığını
       taşımaz. Önemi <= 0 çıkan özellik: model onsuz da aynı - GÜRÜLTÜ.

Bağımsız analiz aracıdır: mevcut hiçbir modüle dokunmaz, yalnız okur.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from ..backtest.backtest_runner import download_ohlcv
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging
from .features import build_dataset
from .train_example import make_model

logger = logging.getLogger("importance")


def _bar(value: float, max_value: float, width: int = 34) -> str:
    """Basit ASCII bar: negatifler için boş, pozitifler oranlı."""
    if max_value <= 0 or value <= 0:
        return ""
    return "█" * max(1, round(width * value / max_value))


def print_table(title: str, names: list[str], values: np.ndarray,
                errors: np.ndarray | None = None) -> None:
    order = np.argsort(values)[::-1]
    total_pos = values[values > 0].sum()
    vmax = values.max() if len(values) else 0.0

    w = 78
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)
    print(f"  {'Özellik':<22} {'Önem':>9} {'Pay':>7}  Grafik")
    print("-" * w)
    for i in order:
        pct = 100 * values[i] / total_pos if total_pos > 0 and values[i] > 0 else 0.0
        err = f" ±{errors[i]:.4f}" if errors is not None else ""
        flag = "  <- gürültü (önem <= 0)" if values[i] <= 0 else ""
        print(f"  {names[i]:<22} {values[i]:>+9.4f}{err} {pct:>5.1f}%  "
              f"{_bar(values[i], vmax)}{flag}")
    print("=" * w)


async def analyze(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], "WARNING")
    logger.setLevel(logging.INFO)

    symbol = args.symbol or config["market_data"]["symbols"][0]
    df = await download_ohlcv(config["exchange"]["id"], symbol, args.timeframe, args.days)
    X, y = build_dataset(df, horizon=args.horizon, threshold=args.threshold)

    # Zaman sıralı bölme + embargo (train_example ile aynı disiplin)
    split = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_eval, y_eval = X.iloc[split + args.horizon:], y.iloc[split + args.horizon:]

    if args.mode == "saved":
        model_path = Path(args.model)
        if not model_path.exists():
            raise SystemExit(f"Model dosyası yok: {model_path} - önce src.ml.train_example çalıştır.")
        model = joblib.load(model_path)
        origin = f"kayıtlı model ({model_path})"
        caveat = ("NOT: Kayıtlı model tüm veriyle eğitildiyse bu değerlendirme kısmen\n"
                  "  IN-SAMPLE'dır; 'model neye bakıyor' sorusuna güvenilir, 'hangi özellik\n"
                  "  OOS katkı verir' sorusu için varsayılan (honest) modu kullan.")
    else:
        model = make_model()
        model.fit(X_train, y_train)
        origin = "dürüst mod: ilk %70 ile sıfırdan eğitildi"
        caveat = None

    auc = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
    print(f"\n  Model    : {type(model).__name__} | {origin}")
    print(f"  Veri     : {symbol} {args.timeframe}, {args.days}g, horizon={args.horizon} "
          f"| eğitim {len(X_train):,} / değerlendirme {len(X_eval):,} satır")
    print(f"  Değerlendirme AUC: {auc:.4f}")
    if caveat:
        print(f"  {caveat}")

    # --- 1) Native önem (varsa: LightGBM/XGBoost/RF) ---
    if hasattr(model, "feature_importances_"):
        print_table(f"NATIVE FEATURE IMPORTANCE ({type(model).__name__})",
                    list(X.columns), np.asarray(model.feature_importances_, dtype=float))
    else:
        print(f"\n  Not: {type(model).__name__} native feature_importances_ sunmaz "
              f"(sklearn HistGB'nin bilinen kısıtı) - permutation importance esas alınır.")

    # --- 2) Permutation importance (her zaman; holdout üzerinde) ---
    result = permutation_importance(
        model, X_eval, y_eval, scoring="roc_auc",
        n_repeats=args.repeats, random_state=42, n_jobs=-1)
    print_table(f"PERMUTATION IMPORTANCE (holdout AUC düşüşü, {args.repeats} tekrar)",
                list(X.columns), result.importances_mean, result.importances_std)

    noise = [c for c, v in zip(X.columns, result.importances_mean) if v <= 0]
    if noise:
        print(f"  Gürültü adayları ({len(noise)}): {', '.join(noise)}")
        print("  Bunlar karıştırılınca AUC düşmüyor -> model onlarsız da aynı.")
        print("  Elemek istersen: build_features'tan çıkar, modeli yeniden eğit,")
        print("  kararı grid-search OOS raporuyla doğrula (tek doğruluk kaynağı ilkesi).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Özellik önemi analizi (bağımsız araç)")
    parser.add_argument("--mode", choices=["honest", "saved"], default="honest",
                        help="honest: %%70 ile eğit, %%30 holdout'ta ölç (varsayılan). "
                             "saved: kayıtlı modeli incele.")
    parser.add_argument("--model", default="models/ml_model.joblib",
                        help="saved modunda yüklenecek model yolu")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=10,
                        help="Permutation tekrar sayısı (varsayılan: 10)")
    args = parser.parse_args()
    asyncio.run(analyze(args))


if __name__ == "__main__":
    main()
