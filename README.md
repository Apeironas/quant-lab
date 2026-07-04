# Kripto Algoritmik Al-Sat Araştırma Laboratuvarı (Quant Lab)

> **EN summary:** An event-driven crypto trading research lab in Python: async
> data pipeline (OHLCV + perpetual funding), leakage-proof feature engineering,
> triple-barrier labeling, realistic cost simulation, walk-forward analysis and
> nested grid-search. Nine rigorous experiments concluded in an **honest
> negative result**: no exploitable post-cost alpha was found for gradient
> boosting on BTC 1h/4h price+funding features — and the infrastructure that
> proved it is the deliverable.

Bu depo bir "para basma botu" değildir. **Bir hipotezi dürüstçe test edip
reddedebilen, uçtan uca bir nicel araştırma altyapısıdır.** Dokuz deneylik
sistematik süreç sonunda ulaşılan bilimsel sonuç: bu veri/model kombinasyonunda
maliyet sonrası sürdürülebilir alpha **yoktur** — ve bunu tek kuruş gerçek
para riske etmeden, tekrarlanabilir deneylerle kanıtlamak sistemin başarısıdır.

---

## Mimari: Olay Güdümlü Çekirdek

Tüm modüller `asyncio.Queue` tabanlı bir EventBus üzerinden tipli olaylarla
konuşur; hiçbir modül diğerinin içini bilmez. Aynı strateji kodu, tek satır
değişmeden hem canlı akışta hem backtest'te çalışır.

```
DataFetcher ──MarketEvent──> Strategy ──SignalEvent──> RiskManager
 (ccxt.pro websocket,        (ML/kural               (boyutlandırma,
  yalnız KAPANAN mumlar)      tabanlı)                devre kesici)
                                                          │
Portfolio <───FillEvent─── ExecutionEngine <──OrderEvent──┘
(muhasebe, PnL)            (komisyon+slippage simülasyonu,
      │                     SL/TP/dikey bariyer izleme)
      └────── FillEvent ──> Strategy (durum farkındalığı)
```

| Modül | Sorumluluk |
|---|---|
| `src/core/` | Olay tanımları + EventBus (`drain()` ile deterministik backtest) |
| `src/data/` | Websocket/REST piyasa verisi; funding & open-interest hattı |
| `src/strategy/` | İzole strateji katmanı; ML şablonu + meta-labeling kancası |
| `src/risk/` | Sabit oranlı risk boyutlandırma, zorunlu SL/TP, günlük devre kesici |
| `src/execution/` | Paper/live yürütme; üçlü bariyer takibi; çift-çıkış koruması |
| `src/portfolio/` | Tek doğruluk kaynağı muhasebe (komisyonlar dahil) |
| `src/ml/` | Özellik mühendisliği, etiketleme, eğitim, önem analizi |
| `src/backtest/` | Backtest, walk-forward ve nested grid-search sürücüleri |

## Veri Hattı

- **OHLCV**: sayfalı indirme + yerel önbellek (`data/cache/`); canlıda ccxt.pro
  websocket, yalnız **kapanmış** mumlar yayınlanır (repaint imkânsız).
- **Funding rate** (perpetual): yıllarca geriye giden geçmiş, `merge_asof
  (backward)` ile mum açılış zamanına sızıntısız hizalanır.
- **Open interest**: altyapı hazır; Binance geçmişi ~30 günle sınırladığı için
  uzun backtest'te kullanılmaz (dürüst envanter — bkz. `src/data/derivatives.py`).

## Sızıntı (Look-Ahead) Savunması

Veri sızıntısı, backtest'leri geçersiz kılan bir numaralı hatadır; burada üç
katmanla savunulur:

1. **Kural**: t satırının özellikleri yalnız t ve öncesinden hesaplanır;
   geleceğe bakmak sadece etiket fonksiyonlarına aittir (`features.py` başındaki
   sözleşme).
2. **Yapı**: üst zaman dilimi (MTF) özellikleri `resample → shift(1) → ffill`
   kilidiyle iner: bir 4h/1d barının değeri, ancak o bar **kapandıktan** sonra
   taban satırlara düşer. Eğitim ve çıkarım aynı `build_features()`'ı çağırır —
   train/serve kayması yapısal olarak imkânsız.
3. **Test**: `tests/test_leakage.py` veriyi rastgele noktalardan keser ve kalan
   satırların özelliklerinin bit-bit aynı kaldığını doğrular ("gelecek silinince
   geçmiş değişmemeli").

## Triple-Barrier Etiketleme + Bariyer-Sadık Yürütme

Etiketler sabit-ufuk yerine **üçlü bariyer** (López de Prado) ile üretilir:
TP = giriş + 2×ATR, SL = giriş − 1×ATR, dikey bariyer = `horizon` bar; aynı
barda ikisi değerse muhafazakâr varsayımla SL sayılır. Kritik tamamlayıcı:
**yürütme motoru da birebir aynı kuralları uygular** (ATR-bazlı SL/TP, bar
sayaçlı zaman durdurucu, olasılık-bazlı erken çıkış kapalı). Model eğitimde
neyi öğrendiyse, canlıda kuruşu kuruşuna onu yaşar. Meta-labeling için mimari
kanca hazırdır (`MLStrategy(meta_model=...)`: başarı olasılığıyla veto +
boyutlandırma).

## Değerlendirme Metodolojisi

- **Gerçekçi maliyetler**: her simüle işlemde maker/taker komisyonu + aleyhte
  slippage (spot ve futures profilleri `config.yaml`'da).
- **Walk-forward**: 180 gün eğit → embargo → 60 gün görülmemiş veride test →
  kaydır; sermaye pencereler arası devreder, OOS eğrileri uç uca eklenir.
- **Nested grid-search**: hiperparametreler her pencerenin YALNIZ iç
  train/validation kesitinde seçilir; OOS seçime asla katılmaz (nested CV).
  Skor, komisyon sonrası Net Sharpe'tır — brüt kâr değil.
- **Permutation importance**: holdout üzerinde model-agnostik özellik önemi;
  gürültü özellikleri sistematik budanır.

## Bilimsel Süreç ve Hipotez Reddi

Dokuz deneylik yolculuğun tamamı (hepsi out-of-sample, komisyon+slippage dahil,
BTC/USDT, 10.000 USDT başlangıç):

| # | Deney | OOS NET | Anahtar bulgu |
|---|---|---|---|
| 0 | Kontrol: aynı dönemde eğit+test | *+%24.0* | Holdout AUC 0.52'yken +%24 "kâr": **in-sample yanılsamasının kanıtı** — bu satır geçersizdir ve o yüzden buradadır |
| 1 | Walk-forward, sabit parametreler (1h, 14 özellik) | -%24.9 (0/11) | Dürüst zemin kuruldu |
| 2 | Nested grid-search (14 öz.) | -%12.8 (1/11) | Optimizasyon maliyeti yönetir, alpha yaratamaz |
| 3 | + MTF özellikleri (22 öz.) | -%6.3 (4/11) | MTF trend eğimleri özellik öneminin %61'i |
| 4 | Budama + uzun ufuklar (16 öz.) | -%6.8 (3/11) | **Val-OOS ters korelasyonu**: 30 günlük pencereler istatistiksel olarak yetersiz |
| 5 | + Funding rate, 60g pencereler | -%6.7 (2/8) | Ters korelasyon kırıldı; plato sürdü |
| 6 | Maliyet A/B: futures komisyonları (%47 ucuz) | -%5.5 | **Brüt PnL yine eksi → sorun maliyet değil, sinyal** |
| 7 | 4h taban + Triple-Barrier | -%6.8 (3/8) | Gerçekleşen RR 0.9 ≠ tasarlanan 2.0: erken çıkış kazananları kesiyor |
| 8 | v2.1 bariyer-sadık yürütme | -%7.7 (1/8) | RR 1.63'e çıktı ama isabet %33.2'ye indi — **1:2 bariyerin sıfır-edge denge noktası (⅓)** |

**Sonuç ve yorum:** İşlem yönetimi tarafsızlaştırıldığında isabet oranının tam
olarak şans seviyesine (⅓) yakınsaması, giriş sinyalinin zamanlama becerisi
taşımadığının en temiz kanıtıdır. Gradient boosting sınıfı modeller + 1h/4h
OHLCV/MTF/funding özellikleri + yön/bariyer etiketleri kombinasyonu, BTC'de
maliyet sonrası sömürülebilir alpha üretmemektedir. Bu **negatif ama sağlam**
bir sonuçtur: sistem her iterasyonda kendi hatalarını da yakaladı (in-sample
yanılsaması, çift-çıkış muhasebe hatası, val-OOS örneklem yetersizliği,
gerçekleşen-RR uyumsuzluğu) ve tek kuruş gerçek sermaye riske edilmeden
öğrenildi. *Verimli piyasalar hipotezi, en azından bu silahlarla, bu cephede
ayakta.*

## Kurulum ve Kullanım

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env    # Binance TESTNET anahtarları (testnet.binance.vision, ücretsiz)

# Canlı paper-trading botu (testnet, gerçek para yok)
python -m src.main

# Model eğitimi + backtest
python -m src.ml.train_example --days 720 --timeframe 4h
python -m src.backtest.backtest_runner --strategy ml --days 720 --timeframe 4h

# Walk-forward ve nested grid-search (asıl değerlendirme araçları)
python -m src.backtest.walk_forward_runner --days 720 --timeframe 4h --train-days 180 --test-days 60 --funding
python -m src.backtest.grid_search --days 720 --timeframe 4h --train-days 180 --test-days 60 --horizons 12,24,42 --thresholds 0.45:0.35,0.50:0.40,0.55:0.45 --funding

# Özellik önemi (permutation, holdout üzerinde)
python -m src.ml.analyze_importance --days 720 --timeframe 4h --horizon 24

# Testler (sızıntı + triple-barrier + yürütme entegrasyonu)
python -m pytest tests/ -v
```

## Bölüm 2: Gelecek Vizyonu (Future Work)

Bu altyapının üzerine inşa edilecek, her biri **yeni bilgi kaynağı** getiren
araştırma hatları:

- **Order Book / Mikro-yapı**: L2 derinlik dengesizliği (order flow imbalance),
  spread dinamikleri, agresör hacim oranları — websocket altyapısı hazır,
  `watch_order_book` config'te bir bayrak.
- **On-Chain Veriler**: borsa giriş/çıkış akışları, aktif adresler, MVRV/SOPR
  tipi değerleme metrikleri; `merge_derivatives` deseni her yeni zaman serisini
  sızıntısız yapıştırmak için genelleşir.
- **Open Interest birikimi**: OI fetcher'ı zamanlanmış görevle çalıştırıp kendi
  tarihsel arşivini biriktirmek (borsanın 30 günlük penceresini aşmanın tek yolu).
- **Cross-Sectional Momentum**: tek varlıkta zaman serisi tahmini yerine, çok
  varlıklı evrende göreli güç sıralaması — tek-varlık gürültüsüne karşı en
  belgelenmiş savunma.
- **Market Making**: maker iadeleriyle spread'den kazanç + envanter riski
  yönetimi; "yön tahmin etme, likidite sat" paradigma değişimi.
- **Meta-labeling'in tamamlanması**: birincil sinyaller üzerinde başarı
  olasılığı modeli eğitip `MLStrategy`'deki hazır kancaya takmak.

## Uyarı

Bu yazılım araştırma ve eğitim amaçlıdır. Kripto piyasaları yüksek risklidir;
geçmiş performans (özellikle de negatifse!) gelecek sonuçların göstergesi
değildir. Canlı kullanımdan doğan tüm sorumluluk kullanıcıya aittir.
