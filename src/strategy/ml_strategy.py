"""
ML STRATEJİ ŞABLONU - "tak-çalıştır" model entegrasyonu

Sadece ÇIKARIM (inference) yapar; eğitim tamamen dışarıdadır:
    1. Modeli eğit:  python -m src.ml.train_example --days 365 --timeframe 1h
       (veya kendi pipeline'ınla eğitip joblib.dump ile kaydet)
    2. Backtest:     python -m src.backtest.backtest_runner --strategy ml ...
    3. Canlı/paper:  config.yaml -> strategy.name: ml

Model gereksinimleri:
    - sklearn API'sine uyan predict_proba(X) -> [:, 1] = p(yükseliş)
      (sklearn, LightGBM sklearn-wrapper, XGBClassifier... hepsi uyar)
    - features.build_features'ın ürettiği sütunlarla eğitilmiş olmalı.
      Aynı fonksiyonu kullandığımız sürece bu garanti (train/serve skew yok).

Walk-forward hazırlığı: __init__ 'model' parametresiyle hazır model nesnesi
enjekte edilmesine izin verir. İleride walk-forward modülü her pencere için
yeni model eğitip MLStrategy(config, bus, model=pencere_modeli) diye
oluşturacak - bu dosyaya dokunmak gerekmeyecek.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import ta

from ..core.events import SignalDirection, SignalEvent
from ..ml.features import (MAX_LOOKBACK, TB_ATR_WINDOW, TB_SL_ATR, TB_TP_ATR,
                           build_features)
from .base_strategy import BaseStrategy

logger = logging.getLogger("ml_strategy")


class MLStrategy(BaseStrategy):
    NAME = "ml"

    def __init__(self, config: dict, bus, model=None, meta_model=None) -> None:
        super().__init__(config, bus)
        self.long_threshold: float = self.params.get("long_threshold", 0.60)
        self.exit_threshold: float = self.params.get("exit_threshold", 0.45)
        self.min_bars: int = self.params.get("min_bars", MAX_LOOKBACK + 10)
        # SL/TP bariyerleri, triple-barrier ETİKETLERİYLE AYNI ATR katlarını
        # kullanır: model neyi öğrendiyse yürütme de onu yaşar (tutarlılık).
        self.tp_atr: float = self.params.get("tp_atr", TB_TP_ATR)
        self.sl_atr: float = self.params.get("sl_atr", TB_SL_ATR)
        # v2.1 BARİYER-SADIK İŞLEM YÖNETİMİ:
        #   barrier_exit_only=True -> olasılık-bazlı erken çıkış (proba-exit)
        #   DEVRE DIŞI; pozisyon yalnız üçlü bariyerle kapanır (SL / TP /
        #   dikey zaman bariyeri). Etiket neyi ölçüyorsa yürütme onu yaşar.
        #   time_stop_bars = dikey bariyer (etiketin horizon'u ile AYNI olmalı;
        #   grid-search bunu pencere başına otomatik enjekte eder).
        self.barrier_exit_only: bool = self.params.get("barrier_exit_only", True)
        self.time_stop_bars: int = self.params.get("time_stop_bars", 24)
        #: Sembol başına pozisyon durumu - FILL geri beslemesiyle güncellenir
        #: (sinyal yayınlamak değil, emrin GERÇEKLEŞMESİ pozisyon açar)
        self._in_long: dict[str, bool] = {}

        if model is not None:
            # Walk-forward / test enjeksiyonu: hazır model nesnesi
            self.model = model
        else:
            model_path = Path(self.params.get("model_path", "models/ml_model.joblib"))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model dosyası yok: {model_path}\n"
                    f"Önce eğit: python -m src.ml.train_example --days 365 --timeframe 1h\n"
                    f"veya joblib.dump(model, '{model_path}') ile bir model kaydedin."
                )
            self.model = joblib.load(model_path)
            logger.info("Model yüklendi: %s (%s)", model_path, type(self.model).__name__)

        # ------------------------------------------------------------------
        # META-LABELING HAZIRLIĞI (López de Prado, bölüm 3.6)
        # Birincil model YÖNÜ söyler (long girilsin mi?); meta-model o
        # işlemin BAŞARI OLASILIĞINI puanlar. Meta-model:
        #   - girdi olarak birincil özellikleri + birincil olasılığı alır,
        #   - p_basari < meta_threshold ise sinyal VETO edilir,
        #   - p_basari, SignalEvent.strength'e yazılır (ileride pozisyon
        #     boyutunu olasılıkla ölçeklemek için hazır kanca).
        # meta_model None ise sistem bugünkü gibi tek modelle çalışır.
        # Eğitimi: birincil modelin sinyalleri üzerinde triple-barrier
        # sonuçlarıyla (başarılı/başarısız) ayrı bir script'te yapılacak.
        # ------------------------------------------------------------------
        self.meta_model = meta_model
        meta_path = self.params.get("meta_model_path")
        if self.meta_model is None and meta_path and Path(meta_path).exists():
            self.meta_model = joblib.load(meta_path)
            logger.info("Meta-model yüklendi: %s", meta_path)
        self.meta_threshold: float = self.params.get("meta_threshold", 0.5)

    async def on_fill(self, event) -> None:
        """
        DURUM FARKINDALIĞI (v2.1): pozisyon durumu FILL olaylarından öğrenilir.
        SL/TP/dikey bariyer pozisyonu kapattığında strateji bunu anında görür
        ve koşullar oluşursa YENİDEN pozisyon alabilir. (Eski histerezis
        yaklaşımında strateji SL sonrası kör kalıyordu.)
        """
        from ..core.events import OrderSide
        self._in_long[event.symbol] = (event.side == OrderSide.BUY)
        logger.debug("%s: fill geri beslemesi -> in_long=%s (%s)",
                     event.symbol, self._in_long[event.symbol], event.reason)

    def generate_signal(self, symbol: str, candles) -> SignalEvent | None:
        if len(candles) < self.min_bars:
            return None

        # Özellikler eğitimdekiyle AYNI fonksiyondan gelir; sadece son
        # (en yeni kapanmış) mumun satırı çıkarım için kullanılır.
        features = build_features(candles)
        x_last = features.iloc[[-1]]
        if x_last.isna().any(axis=None):
            return None  # pencereler henüz dolmamış

        proba_up = float(self.model.predict_proba(x_last)[0, 1])
        price = float(candles["close"].iloc[-1])
        in_long = self._in_long.get(symbol, False)

        # Her mum kapanışında kararı logla (canlıda 4h'de bir satır - izlenebilirlik).
        # Backtest'te log seviyesi WARNING olduğundan bu satır raporu kirletmez.
        logger.info("%s değerlendirme: p(başarı)=%.3f | eşik=%.2f | pozisyon=%s -> %s",
                    symbol, proba_up, self.long_threshold,
                    "AÇIK" if in_long else "yok",
                    "sinyal üretilecek" if (proba_up >= self.long_threshold and not in_long)
                    else "PAS")

        # Histerezis: giriş eşiği çıkış eşiğinden yüksek tutulur ki
        # olasılık eşik etrafında titreştiğinde gir-çık yapılmasın.
        if proba_up >= self.long_threshold and not in_long:
            strength = proba_up

            # --- META-LABELING KAPISI (meta-model varsa) ---
            if self.meta_model is not None:
                meta_x = x_last.copy()
                meta_x["primary_proba"] = proba_up  # birincil güven de bir özellik
                p_success = float(self.meta_model.predict_proba(meta_x)[0, 1])
                if p_success < self.meta_threshold:
                    logger.debug("%s: meta-model vetosu (p_basari=%.2f)", symbol, p_success)
                    return None
                strength = p_success  # boyutlandırma kancası: başarı olasılığı

            # SL/TP: triple-barrier etiketleriyle AYNI ATR katları -> modelin
            # öğrendiği senaryo ile yürütmenin yaşadığı senaryo birebir aynı.
            atr = float(ta.volatility.average_true_range(
                candles["high"], candles["low"], candles["close"],
                window=TB_ATR_WINDOW).iloc[-1])
            if atr <= 0:
                return None  # ATR'siz bariyer kurulamaz
            # NOT: _in_long burada DEĞİL, on_fill'de güncellenir - pozisyonu
            # sinyal değil, gerçekleşen emir açar (durum farkındalığı).
            return SignalEvent(
                symbol=symbol, direction=SignalDirection.LONG,
                price=price, strength=strength,
                stop_loss=price - self.sl_atr * atr,
                take_profit=price + self.tp_atr * atr,
                time_stop_bars=self.time_stop_bars,  # dikey bariyer = etiket horizon'u
            )

        # --- Olasılık-bazlı erken çıkış: v2.1'de varsayılan olarak KAPALI ---
        # Etiket "bariyere kadar tut"u ölçer; erken çıkış kazananları TP'den
        # önce kesip tasarlanan 1:2 RR'yi bozuyordu (gerçekleşen ~0.9'a düşmüştü).
        if (not self.barrier_exit_only
                and proba_up <= self.exit_threshold and in_long):
            return SignalEvent(
                symbol=symbol, direction=SignalDirection.EXIT,
                price=price, strength=1.0 - proba_up,
            )
        return None
