"""
META-MODEL EĞİTİMİ (Meta-Labeling, López de Prado bölüm 3.6)

Çalıştırma (proje kökünden):
    python -m src.ml.train_meta --days 720 --timeframe 4h --horizon 24
    python -m src.ml.train_meta --days 720 --timeframe 4h --horizon 24 --funding

Fikir: birincil model YÖN/GİRİŞ kararı verir ("şimdi long mantıklı");
meta-model ise o kararın BAŞARI OLASILIĞINI puanlar. Meta-model yalnız
birincil modelin sinyal verdiği anlarda eğitilir - görevi piyasayı tahmin
etmek değil, "birincil modelin hangi sinyalleri tutuyor?"u öğrenmektir.

Akış (zaman sıralı, embargo'lu):
    [--- birincil eğitim %60 ---][emb][-- meta eğitim %20 --][emb][-- meta test %20 --]
    1. Birincil model ilk dilimde eğitilir (triple-barrier etiketleri)
    2. Meta dilimlerinde birincil olasılıklar üretilir; olasılık >= eşik olan
       barlar "SİNYAL OLAYI" sayılır
    3. Her olayın meta-etiketi = o barın triple-barrier SONUCU (TP mi SL mi?)
    4. Meta-model, [özellikler + primary_proba] üzerinde bu olaylarla eğitilir
       (MLStrategy'deki çıkarım kancasıyla birebir aynı girdi formatı)
    5. Meta-test olaylarında rapor: meta onayının kazanma oranını ne kadar
       yükselttiği (precision uplift) + AUC
    6. Model models/meta_model.joblib'e kaydedilir -> config'te
       strategy.ml.meta_model_path ile aktive edilir

DÜRÜST NOT: meta-model, birincil modelde OLMAYAN alpha'yı yaratamaz; zayıf
bir edge'in kesinliğini artırır (az ama isabetli işlem). Bölüm 1'in sonucu
"birincilde edge yok" idi - bu script şimdilik altyapı ısındırmasıdır ve
OBI özellikleri geldiğinde gerçek görevine başlayacaktır.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..backtest.backtest_runner import download_ohlcv
from ..utils.config_loader import load_config
from ..utils.logger import setup_logging
from .features import build_dataset
from .train_example import make_model

logger = logging.getLogger("train_meta")

MODELS_DIR = Path("models")


def make_meta_model():
    """Meta-model mimarisi (sklearn API yeterli)."""
    return make_model()  # varsayılan: birincil ile aynı aile (HistGB)


async def train_meta(args: argparse.Namespace) -> None:
    config = load_config()
    setup_logging(config["logging"]["dir"], "INFO")

    symbol = args.symbol or config["market_data"]["symbols"][0]
    df = await download_ohlcv(config["exchange"]["id"], symbol, args.timeframe, args.days)
    if args.funding:
        from ..data.derivatives import download_funding_history, merge_derivatives
        funding = await download_funding_history(config["exchange"]["id"], symbol, args.days)
        df = merge_derivatives(df, funding_df=funding)

    # Triple-barrier etiketleri: HER bar için "long açılsaydı akıbeti ne olurdu?"
    X, y = build_dataset(df, horizon=args.horizon)
    emb = args.horizon

    # --- Zaman sıralı üçlü bölme ---
    i1 = int(len(X) * 0.6)
    i2 = int(len(X) * 0.8)
    X_pri, y_pri = X.iloc[:i1], y.iloc[:i1]
    X_mtr, y_mtr = X.iloc[i1 + emb:i2], y.iloc[i1 + emb:i2]
    X_mte, y_mte = X.iloc[i2 + emb:], y.iloc[i2 + emb:]

    # --- 1) Birincil model ---
    primary = make_model()
    primary.fit(X_pri, y_pri)

    def make_events(X_seg: pd.DataFrame, y_seg: pd.Series):
        """Birincil sinyal olayları: proba >= eşik olan barlar + meta girdisi."""
        proba = primary.predict_proba(X_seg)[:, 1]
        mask = proba >= args.primary_threshold
        meta_X = X_seg[mask].copy()
        meta_X["primary_proba"] = proba[mask]  # MLStrategy kancasıyla aynı format
        return meta_X, y_seg[mask]

    ev_train_X, ev_train_y = make_events(X_mtr, y_mtr)
    ev_test_X, ev_test_y = make_events(X_mte, y_mte)
    logger.info("Sinyal olayları: meta-eğitim %d, meta-test %d (eşik=%.2f)",
                len(ev_train_X), len(ev_test_X), args.primary_threshold)
    if len(ev_train_X) < 50 or len(ev_test_X) < 20:
        raise SystemExit("Yetersiz sinyal olayı - --primary-threshold'u düşür "
                         "veya --days'i artır.")

    # --- 2) Meta-model: "bu sinyal tutar mı?" ---
    meta = make_meta_model()
    meta.fit(ev_train_X, ev_train_y)

    # --- 3) Dürüst değerlendirme (meta-test olayları) ---
    p_success = meta.predict_proba(ev_test_X)[:, 1]
    base_wr = ev_test_y.mean()                       # tüm birincil sinyallerin isabeti
    approved = p_success >= args.meta_threshold
    appr_wr = ev_test_y[approved].mean() if approved.sum() > 0 else float("nan")
    auc = roc_auc_score(ev_test_y, p_success) if ev_test_y.nunique() > 1 else float("nan")

    w = 62
    print()
    print("=" * w)
    print(f"  META-LABELING RAPORU  |  {symbol} {args.timeframe}  h={args.horizon}")
    print("=" * w)
    print(f"  Birincil eşik        : proba >= {args.primary_threshold}")
    print(f"  Meta-test olayı      : {len(ev_test_X)} sinyal")
    print(f"  Taban isabet (meta'sız): %{100 * base_wr:.1f}")
    print(f"  Meta onaylı isabet   : %{100 * appr_wr:.1f}  "
          f"({int(approved.sum())}/{len(ev_test_X)} sinyal onaylandı)")
    print(f"  Meta AUC             : {auc:.4f}")
    print("-" * w)
    uplift = 100 * (appr_wr - base_wr) if appr_wr == appr_wr else float("nan")
    if uplift == uplift and uplift > 2:
        print(f"  Meta filtre isabeti +{uplift:.1f} puan artırıyor - umut verici;")
        print("  yine de son söz walk-forward'ındır.")
    else:
        print("  Meta filtre anlamlı seçicilik katmıyor (beklenen: birincilde")
        print("  edge yokken meta da mucize yaratamaz). OBI özellikleri gelince")
        print("  bu script yeniden koşulacak.")
    print("=" * w)

    # --- 4) Kaydet ---
    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / "meta_model.joblib"
    joblib.dump(meta, out)
    print(f"\n  Meta-model kaydedildi: {out}")
    print("  Aktive etmek için config.yaml -> strategy.ml altında:")
    print("    meta_model_path: models/meta_model.joblib")
    print(f"    meta_threshold: {args.meta_threshold}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-labeling model eğitimi")
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--horizon", type=int, default=24,
                        help="Triple-barrier dikey bariyeri (bar)")
    parser.add_argument("--primary-threshold", type=float, default=0.45,
                        help="Birincil sinyal eşiği (TB taban oranı ~0.35'e göre)")
    parser.add_argument("--meta-threshold", type=float, default=0.5,
                        help="Meta onay eşiği")
    parser.add_argument("--funding", action="store_true")
    args = parser.parse_args()
    asyncio.run(train_meta(args))


if __name__ == "__main__":
    main()
