"""
ÖZELLİK ÇIKARIMI (Feature Engineering) - TEK DOĞRULUK KAYNAĞI

Bu modül bilinçli olarak stratejiden, backtest'ten ve eğitimden AYRI durur.
Aynı fonksiyonlar üç yerde birden kullanılır:
    1. Eğitim   : src/ml/train_example.py -> build_dataset()
    2. Çıkarım  : src/strategy/ml_strategy.py -> build_features() (son satır)
    3. Walk-forward / grid-search: her pencere için build_dataset(df.iloc[a:b])

Eğitim ile çıkarım aynı kodu çağırdığı için "train/serve skew" (eğitimde
başka, canlıda başka özellik) sınıfı hatalar yapısal olarak imkansızdır.

=====================================================================
VERİ SIZINTISI (DATA LEAKAGE) KURALLARI
=====================================================================
1. t satırındaki özellik, YALNIZCA t ve öncesi mumlardan hesaplanabilir.
   rolling(), pct_change(), shift(+n), ewm() -> güvenli (geriye bakar)
   shift(-n), merkezli pencere (center=True), tüm seriye fit edilen
   scaler/normalizasyon -> SIZINTI! Kullanma.
2. t mumunun kendi OHLCV'sini kullanmak sızıntı DEĞİLDİR: sistem sinyali
   mum KAPANDIKTAN sonra üretir (bkz. data_fetcher - sadece kapanan mum
   yayınlanır). Yani t kapanışı, karar anında bilinen bir veridir.
3. Geleceğe bakmak SADECE build_labels()'a aittir (eğitim hedefi) ve
   asla özellik olarak kullanılamaz.
4. Ölçekleme gerekiyorsa (NN vb.) scaler'ı SADECE train penceresine fit
   et, modelle birlikte kaydet. Ağaç tabanlı modellerde gerek yok.

=====================================================================
ÇOKLU-ZAMAN DİLİMİ (MTF) SIZINTI KİLİDİ - resample + shift(1) + ffill
=====================================================================
Üst zaman dilimi (4h/1d) barının değeri ancak o bar KAPANDIĞINDA bilinir.
Saat 13:00 satırı, 16:00'da kapanacak 4h barın hiçbir bilgisini göremez.
Üç adımlı garanti:

    a) resample(label="left", closed="left"): 12:00 indeksli 4h bar,
       [12:00, 16:00) aralığını kapsar - yani 16:00'da kapanır.
    b) shift(1): HTF indikatör serisi BİR HTF BAR ileri kaydırılır.
       Böylece 16:00 indeksine, 16:00'da kapanmış olan [12:00-16:00)
       barının değeri yazılır; 12:00 indeksine bir önceki barınki.
    c) reindex(base_index, method="ffill"): 1h satırları, kendi zamanına
       eşit/önceki son HTF indeks değerini alır. Örn. 13:00, 14:00, 15:00
       satırları hâlâ [08:00-12:00) barını görür; [12:00-16:00) barını
       İLK görebilen satır 16:00'dır (o anda bar kapanmıştır).

Sonuç: her 1h satırı, o an itibarıyla TAMAMLANMIŞ son HTF barı görür -
gerektiğinde bir bar muhafazakâr, asla ileriye bakmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

#: build_features'ın en uzun geriye bakışı, TABAN BAR cinsinden.
#: 1h taban: 1d SMA10 + shift(1) ~ 12 gün = ~288 bar.
#: 4h taban: 1d SMA40 + shift(1) ~ 42 gün = ~252 bar.
#: İkisini de 300 karşılar. Walk-forward/grid-search ısınma payını ve
#: strateji min_bars kontrolünü bu sabit besler. DİKKAT: 1h'den KÜÇÜK
#: taban (örn. 15m) gün bazlı özellikler için daha çok bar ister;
#: MAX_LOOKBACK'i ve BaseStrategy.MAX_BUFFER'ı ona göre büyüt.
MAX_LOOKBACK = 300

_HTF_AGG = {"open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}


def _htf_features(df_dt: pd.DataFrame, rule: str, prefix: str,
                  sma_window: int, with_close_vs_sma: bool = True) -> pd.DataFrame:
    """
    Üst zaman dilimi (4h/1d) trend özellikleri - sızıntı kilitli.
    df_dt: DatetimeIndex'li taban OHLCV. Çıktı: taban indekse hizalanmış frame.

    Budama notu (permutation importance, h=6 ve h=24'te çift doğrulama):
    HTF RSI'lar (h4_rsi14, d1_rsi7) her iki ufukta da gürültü/aktif zararlı
    çıktı ve elendi; h4_close_vs_sma20 de öyle. Trend EĞİMİ (slope) ise
    özellikle uzun ufukta önemin lideri - o çekirdek olarak kalıyor.
    """
    htf = df_dt.resample(rule, label="left", closed="left").agg(_HTF_AGG).dropna()

    feats = pd.DataFrame(index=htf.index)
    sma = htf["close"].rolling(sma_window).mean()
    feats[f"{prefix}_sma{sma_window}_slope"] = sma.pct_change()        # trend eğimi (yön+şiddet)
    if with_close_vs_sma:
        feats[f"{prefix}_close_vs_sma{sma_window}"] = htf["close"] / sma - 1

    # SIZINTI KİLİDİ (b): değerler ancak bar kapanınca bilinir -> 1 HTF bar kaydır
    feats = feats.shift(1)
    # SIZINTI KİLİDİ (c): taban satırlara "o ana kadar bilinen son değer" ile yay
    return feats.reindex(df_dt.index, method="ffill")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame'inden model özellikleri üretir.

    Girdi : kronolojik sıralı df (sütunlar: ts, open, high, low, close, volume)
    Çıktı : df ile aynı (pozisyonel) index'te özellik DataFrame'i.
            İlk ~MAX_LOOKBACK satır NaN içerir (pencereler dolana kadar) -
            eğitimde build_dataset bunları atar, çıkarımda strateji kontrol eder.

    Yeni özellikler yukarıdaki sızıntı kurallarına uyduğu sürece eklenebilir.
    """
    close, high, low = df["close"], df["high"], df["low"]
    open_, volume = df["open"], df["volume"]
    out = pd.DataFrame(index=df.index)

    # ================= TABAN TIMEFRAME ÖZELLİKLERİ =================

    # --- Momentum / getiri (farklı ufuklarda yüzde değişim) ---
    # ret_24 elendi: her iki ufukta da gürültü + close_vs_sma50 ile r=0.90 korele
    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = close.pct_change(n)

    # --- Volatilite (1-bar getirilerin kayan std'si) ---
    r1 = close.pct_change()
    for n in (12, 48):
        out[f"volat_{n}"] = r1.rolling(n).std()

    # --- Normalize ATR: mutlak volatiliteyi fiyattan bağımsızlaştırır ---
    atr14 = ta.volatility.average_true_range(high, low, close, window=14)
    out["natr_14"] = atr14 / close

    # bb_pctb_20 elendi: permutation importance h=6 ve h=24'te gürültü
    # (bilgisi zaten volat/close_vs_sma kombinasyonunda mevcut)

    # --- Trend konumu (fiyatın kayan ortalamaya göre yeri) ---
    # close_vs_sma50 elendi: her iki ufukta gürültü + ret_24 ile korele
    out["close_vs_sma20"] = close / close.rolling(20).mean() - 1

    # --- Osilatör ---
    out["rsi_14"] = ta.momentum.rsi(close, window=14) / 100.0

    # --- Hacim ivmesi: mevcut hacim / son 24 barın ortalaması (breakout tespiti) ---
    out["volume_ratio_24"] = volume / volume.rolling(24).mean() - 1
    # Kısa vadeli hacim patlaması (son 6 bara göre ani sıçrama)
    out["volume_ratio_6"] = volume / volume.rolling(6).mean() - 1

    # --- Mum yapısı ---
    candle_range = (high - low).replace(0, np.nan)
    out["candle_pos"] = (close - low) / candle_range   # kapanış mumun neresinde (0=dip, 1=tepe)
    out["range_pct"] = (high - low) / close             # bar genişliği (mikro volatilite)
    # Not: "body" = (close-open)/open özelliği ELENDİ: kripto 7/24 kesintisiz
    # işlem gördüğünden open == önceki close olur ve body, ret_1 ile r=1.000
    # birebir aynıdır (check_correlation yakaladı). Gap'li piyasalarda (BIST
    # vb.) anlamlı olabilir; gerekirse geri ekle.

    # ================= ÇOKLU-ZAMAN DİLİMİ (MTF) ÖZELLİKLERİ =================
    # Sızıntı kilidi ayrıntısı dosya başındaki blokta; kısaca:
    # resample(label=left) -> indikatör -> shift(1) -> taban indekse ffill.
    # Taban timeframe ts aralığından otomatik algılanır; MTF katmanları
    # tabana göre ölçeklenir (1h taban -> 4h+1d, 4h taban -> 1d kısa+uzun).
    bar_ms = int(df["ts"].diff().median()) if len(df) > 1 else 3_600_000
    df_dt = df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))[
        ["open", "high", "low", "close", "volume"]]

    if bar_ms < 3_600_000:
        # DAKİKA ÖLÇEĞİ taban (5m/15m - mikroyapı araştırma profili):
        # Günlük/4h MTF bilinçli olarak YOK - ısınmaları MAX_BUFFER'a sığmaz
        # ve kısa veri dönemini yer. Trend bağlamı 1h katmanından gelir.
        # NOT: 1m taban desteklenmez (1h SMA12 bile 780 bar ister); asgari 5m.
        mtf_blocks = [
            _htf_features(df_dt, "1h", "h1", sma_window=12, with_close_vs_sma=True),
        ]
    elif bar_ms == 3_600_000:
        # 1h taban: MTF = 4h eğim + günlük trend
        mtf_blocks = [
            _htf_features(df_dt, "4h", "h4", sma_window=20, with_close_vs_sma=False),
            _htf_features(df_dt, "1D", "d1", sma_window=10, with_close_vs_sma=True),
        ]
    else:
        # 4h taban (trend-following profili): günlük KISA trend (10g, eğim+konum)
        # + günlük UZUN trend (40g, yalnız eğim). Haftalık resample yerine uzun
        # günlük SMA bilinçli tercih: MAX_BUFFER (500 bar) sınırına sığar.
        mtf_blocks = [
            _htf_features(df_dt, "1D", "d1", sma_window=10, with_close_vs_sma=True),
            _htf_features(df_dt, "1D", "d1", sma_window=40, with_close_vs_sma=False),
        ]

    # ================= TÜREV PİYASA ÖZELLİKLERİ (funding / OI) =================
    # Bu kolonlar src/data/derivatives.py -> merge_derivatives ile OHLCV'ye
    # SIZINTISIZ yapıştırılır (as-of backward join, mum açılışına <= son kayıt).
    # Kolon mevcutsa özellikler aktifleşir; yoksa sessizce atlanır.
    # DİKKAT: modelin eğitildiği kolon seti ile çıkarımdaki set AYNI olmalı -
    # funding'li eğitilen model, funding'li veri akışı ister (train/serve).
    # Pencereler DUVAR SAATİ cinsinden sabit, bar sayısı tabana göre ölçeklenir
    # (1h tabanda 7 gün=168 bar, 4h tabanda 7 gün=42 bar; isimler değişmez).
    bars_per_day = max(1, 86_400_000 // bar_ms)
    w7d, w3d = 7 * bars_per_day, 3 * bars_per_day
    w8h = max(1, 8 * 3_600_000 // bar_ms)  # bir funding periyodu

    if "funding_rate" in df.columns:
        f = df["funding_rate"]
        # Ham seviye: pozitif = long'lar ödüyor (kalabalık long), negatif = tersi
        out["funding_rate"] = f
        # Aşırılık: 7 günlük pencereye göre z-score
        f_std = f.rolling(w7d).std().replace(0, np.nan)
        out["funding_z_7d"] = ((f - f.rolling(w7d).mean()) / f_std).replace(
            [np.inf, -np.inf], 0.0)
        # Kümülatif eğilim: ~3 günlük ortalama funding (kalıcı basınç ölçüsü)
        out["funding_cum_3d"] = f.rolling(w3d).mean()
        # Momentum: bir funding periyodundaki (8 saat) değişim
        out["funding_diff_8h"] = f.diff(w8h)

    # ================= MİKROYAPI (ORDER BOOK) ÖZELLİKLERİ =================
    # Kolonlar src/data/orderbook_store.py -> merge_orderbook ile gelir
    # (bar-içi istatistik, mum konvansiyonuyla hizalı - sızıntısız).
    # Toplama dönemi dışındaki satırlar NaN'dır ve eğitimden düşer.
    if "obi10_mean" in df.columns:
        obi = df["obi10_mean"]
        out["obi10_mean"] = obi                      # barın ortalama baskı dengesi
        out["obi10_last"] = df["obi10_last"]         # bar kapanışındaki denge
        out["obi10_std"] = df["obi10_std"]           # bar içi baskı oynaklığı
        out["obi_persist_6"] = obi.rolling(6).mean() # kalıcılık (~30 dk @5m)
        out["obi_mom_3"] = obi.diff(3)               # baskı momentumu
        out["micro_basis_mean"] = df["micro_basis_mean"]
        out["spread_mean"] = df["spread_bps_mean"]

    if "open_interest" in df.columns:
        # NOT: Binance OI geçmişini ~30 gün tutar; bu blok uzun backtest'te
        # değil, canlı/paper ve ileriye dönük biriken veriyle devreye girer.
        oi = df["open_interest"]
        out["oi_change_1d"] = oi.pct_change(bars_per_day)  # günlük OI momentumu
        oi_std = oi.rolling(w7d).std().replace(0, np.nan)
        out["oi_z_7d"] = ((oi - oi.rolling(w7d).mean()) / oi_std).replace(
            [np.inf, -np.inf], 0.0)                        # OI patlaması (breakout)
    # DatetimeIndex -> pozisyonel hizalama (satır sırası birebir aynı)
    for block in mtf_blocks:
        for col in block.columns:
            out[col] = block[col].to_numpy()

    return out


# ===================================================================== #
#  KORELASYON KONTROLÜ (çoklu bağlantı / gürültü teşhisi)
# ===================================================================== #
def check_correlation(X: pd.DataFrame, threshold: float = 0.90) -> list[tuple[str, str, float]]:
    """
    |korelasyon| >= threshold olan özellik çiftlerini döndürür (teşhis aracı).

    Kullanım (eğitim/inceleme sırasında):
        X, y = build_dataset(df)
        for a, b, c in check_correlation(X):
            print(f"UYARI: {a} ~ {b} (r={c:.2f}) - birini elemeyi düşün")

    Eleme burada değil, build_features'ta yapılır: sütunları çalışma
    zamanında dinamik düşürmek train/serve tutarlılığını bozar (model,
    eğitimde gördüğü sütun setini çıkarımda da ister). Bir özellik
    elenecekse build_features içindeki satırı kaldırıp model yeniden
    eğitilir. Ağaç tabanlı modeller (LightGBM/XGBoost)
    çoklu bağlantıdan tahmin gücü olarak az etkilenir; buradaki amaç
    gürültüyü ve özellik-önemi yanılsamalarını azaltmaktır.
    """
    corr = X.corr().abs()
    cols = corr.columns
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and value >= threshold:
                pairs.append((cols[i], cols[j], float(value)))
    return sorted(pairs, key=lambda p: -p[2])


# ===================================================================== #
#  ETİKETLER (yalnız eğitim)
# ===================================================================== #
def build_labels(df: pd.DataFrame, horizon: int = 6, threshold: float = 0.0) -> pd.Series:
    """
    ESKİ yöntem (sabit ufuk): "t'den horizon bar sonra getiri threshold'u
    aşıyor mu?" (1/0). Kıyas/deney için korunuyor; varsayılan artık
    build_labels_triple_barrier (bkz. build_dataset).
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    labels = (future_return > threshold).astype(float)
    return labels.where(future_return.notna())  # kuyruktaki bilinmeyenler NaN kalsın


#: Triple-barrier varsayılan bariyer genişlikleri (ATR katı).
#: 1:2 risk/ödül - config'deki SL/TP oranı ve trend-following karakteriyle uyumlu.
TB_TP_ATR = 2.0
TB_SL_ATR = 1.0
TB_ATR_WINDOW = 14


def build_labels_triple_barrier(df: pd.DataFrame, horizon: int = 24,
                                tp_atr: float = TB_TP_ATR, sl_atr: float = TB_SL_ATR,
                                atr_window: int = TB_ATR_WINDOW) -> pd.Series:
    """
    TRIPLE-BARRIER etiketleme (López de Prado) - endüstri standardı.

    t anında hayali bir LONG açılır; üç bariyerden hangisi ÖNCE vurulur?
        Üst bariyer  : entry + tp_atr * ATR(t)   -> etiket 1 (kâr al)
        Alt bariyer  : entry - sl_atr * ATR(t)   -> etiket 0 (zarar kes)
        Dikey bariyer: horizon bar dolarsa       -> kapanış > entry ise 1, değilse 0

    Neden sabit ufuktan iyi: etiket, stratejinin GERÇEKTE yaşadığı şeyi
    (SL/TP'li bir işlemin akıbetini) öğretir ve bariyerler ATR'ye bağlı
    olduğundan volatilite rejimine otomatik uyum sağlar.

    Sızıntı notları:
      - Bariyerler t anındaki bilgiyle (close_t, ATR_t) kurulur.
      - Geleceğe bakış YALNIZ etiket sonucunda vardır (izinli tek yer).
      - Aynı barda iki bariyer birden değerse MUHAFAZAKÂR varsayım: SL önce
        (ExecutionEngine'deki simülasyon kuralıyla birebir aynı).
      - Kuyruktaki, akıbeti veri içinde belirlenememiş satırlar NaN kalır
        ve build_dataset tarafından atılır.
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=atr_window).to_numpy(dtype=float)

    n = len(df)
    labels = np.full(n, np.nan)
    for t in range(n - 1):
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue  # ATR ısınması bitmeden etiket üretme
        upper = close[t] + tp_atr * atr[t]
        lower = close[t] - sl_atr * atr[t]
        end = min(t + horizon, n - 1)

        label = None
        for j in range(t + 1, end + 1):
            if low[j] <= lower:          # SL önce kontrol edilir (muhafazakâr)
                label = 0.0
                break
            if high[j] >= upper:
                label = 1.0
                break
        if label is None:
            if t + horizon <= n - 1:      # dikey bariyer veri içinde
                label = 1.0 if close[t + horizon] > close[t] else 0.0
            # değilse: akıbet henüz gelecekte -> NaN kalır
        labels[t] = label if label is not None else np.nan

    return pd.Series(labels, index=df.index)


def build_dataset(df: pd.DataFrame, horizon: int = 24, threshold: float = 0.0,
                  method: str = "triple_barrier") -> tuple[pd.DataFrame, pd.Series]:
    """
    Eğitime hazır (X, y) çifti. Walk-forward/grid-search giriş noktası:
    her pencere için build_dataset(df.iloc[start:end]) çağırmak yeterlidir.

    method="triple_barrier" (varsayılan): horizon = dikey bariyer (bar).
    method="fixed_horizon": eski sabit-ufuk etiketi (threshold kullanılır).
    """
    X = build_features(df)
    if method == "triple_barrier":
        y = build_labels_triple_barrier(df, horizon=horizon)
    else:
        y = build_labels(df, horizon=horizon, threshold=threshold)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid].astype(int)
