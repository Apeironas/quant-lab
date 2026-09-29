# Quant Lab — Kripto Algoritmik Al-Sat Araştırma Altyapısı

Kripto piyasasında sistematik al-sat stratejileri geliştirmek ve dürüstçe test
etmek için yazdığım araştırma altyapısı. Olay güdümlü bir çekirdek üzerine
kurulu: veri toplama, özellik üretimi, etiketleme, backtest, walk-forward
doğrulama ve canlı paper-trading aynı kod yolunu paylaşıyor.

Çıkış sorusu basitti: makine öğrenmesiyle fiyat yönü tahmin edip maliyetler
sonrası para kazanmak mümkün mü? Cevabı aramak için on bir kontrollü deney
yaptım. Sonuçların çoğu negatif çıktı ve raporda olduğu gibi duruyorlar, çünkü
asıl ürün strateji değil: bir hipotezi eleyebilen test düzeneği.

> **EN:** Event-driven crypto trading research lab in Python. Async data pipeline
> (OHLCV, perpetual funding, L2 order book), leakage-proof feature engineering,
> triple-barrier labeling, realistic cost simulation, walk-forward analysis and
> nested grid-search. Eleven controlled experiments, most of them negative and
> reported as such. The infrastructure that proves them is the deliverable.

## Ne yaptım

1. Olay güdümlü bot iskeleti: veri, strateji, risk, yürütme ve portföy katmanları
   tipli olaylarla haberleşiyor; aynı strateji kodu hem canlıda hem backtest'te
   değişmeden çalışıyor.
2. Look-ahead bias'a karşı üç katmanlı savunma kurdum ve testle kanıtladım.
3. ML tarafı: özellik üretimi, triple-barrier etiketleme, walk-forward ve
   nested grid-search ile maliyet sonrası değerlendirme.
4. Dokuz deneyde yön tahmini hattını eledim. Kalıcı bir kenar bulunamadı.
5. Yeni veri kaynaklarına geçtim: emir defteri mikroyapısı (arşivi sıfırdan
   biriktirerek) ve perpetual funding.
6. Mikroyapı sinyalinin gerçek olduğunu ama taker maliyetiyle işlenemeyecek
   kadar küçük kaldığını ölçtüm; maker yürütme modeli yazıp A/B testi yaptım.
7. Funding carry hattına geçtim. Cross-section backtest olumlu, canlı ileriye
   dönük doğrulama sürüyor.

---

## Mimari

Tüm modüller `asyncio.Queue` tabanlı bir EventBus üzerinden tipli olaylarla
konuşuyor, hiçbiri diğerinin içini bilmiyor. Backtest ve canlı akış aynı yolu
kullandığı için strateji kodunda ikinci bir sürüm tutmak gerekmiyor.

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
| `src/core/` | Olay tanımları ve EventBus (`drain()` ile deterministik backtest) |
| `src/data/` | Websocket/REST piyasa verisi; funding, open interest, L2 emir defteri |
| `src/strategy/` | İzole strateji katmanı; ML şablonu ve meta-labeling kancası |
| `src/risk/` | Sabit oranlı risk boyutlandırma, zorunlu SL/TP, günlük devre kesici |
| `src/execution/` | Paper/live yürütme, üçlü bariyer takibi, maker limit modeli |
| `src/portfolio/` | Tek doğruluk kaynağı muhasebe (komisyonlar dahil) |
| `src/ml/` | Özellik mühendisliği, etiketleme, eğitim, önem analizi |
| `src/backtest/` | Backtest, walk-forward ve nested grid-search sürücüleri |
| `src/tools/` | Canlı izleme dashboard'u ve funding harvest takipçisi |

## Veri Hattı

- **OHLCV**: sayfalı indirme ve yerel önbellek (`data/cache/`). Canlıda ccxt.pro
  websocket kullanılıyor ve yalnız kapanmış mumlar yayınlanıyor, böylece repaint
  mümkün değil.
- **Funding rate** (perpetual): yıllarca geriye giden geçmiş, `merge_asof`
  (backward) ile mum açılış zamanına hizalanıyor.
- **Open interest**: altyapı hazır. Binance geçmişi yaklaşık 30 günle
  sınırladığı için uzun backtest'te kullanılmıyor.
- **L2 emir defteri**: satın alınamadığı için biriktiriliyor. Collector 7/24
  çalışıp tahtanın ilk 20 kademesini 1 Hz örnekliyor.

## Sızıntı (Look-Ahead) Savunması

Veri sızıntısı backtest'leri sessizce geçersiz kılar, o yüzden üç katman koydum:

1. **Kural**: t satırının özellikleri yalnız t ve öncesinden hesaplanır.
   Geleceğe bakma izni sadece etiket fonksiyonlarına ait.
2. **Yapı**: üst zaman dilimi özellikleri `resample → shift(1) → ffill`
   kilidiyle iniyor. Bir 4h/1d barının değeri ancak o bar kapandıktan sonra
   taban satırlara düşüyor. Eğitim ve çıkarım aynı `build_features()` çağrısını
   kullandığı için train/serve kayması yapısal olarak mümkün değil.
3. **Test**: `tests/test_leakage.py` veriyi rastgele noktalardan kesiyor ve
   kalan satırların özelliklerinin bit düzeyinde aynı kaldığını doğruluyor.
   Gelecek silindiğinde geçmiş değişmiyorsa sızıntı yok demektir.

## Triple-Barrier Etiketleme ve Bariyer-Sadık Yürütme

Etiketler sabit ufuk yerine üçlü bariyerle üretiliyor (López de Prado):
TP = giriş + 2×ATR, SL = giriş − 1×ATR, dikey bariyer = `horizon` bar. Aynı
barda iki bariyer birden değerse muhafazakâr varsayımla SL sayılıyor.

Bunun kritik tamamlayıcısı, yürütme motorunun da birebir aynı kuralları
uygulaması: ATR bazlı SL/TP, bar sayaçlı zaman durdurucu, olasılık bazlı erken
çıkış kapalı. Model eğitimde hangi senaryoyu öğrendiyse canlıda da onu yaşıyor.
Meta-labeling için mimari kanca hazır (`MLStrategy(meta_model=...)`).

## Değerlendirme Metodolojisi

- **Gerçekçi maliyetler**: her simüle işlemde maker/taker komisyonu ve aleyhte
  slippage. Spot ve futures profilleri `config.yaml`'da.
- **Walk-forward**: 180 gün eğit, embargo bırak, 60 gün görülmemiş veride test
  et, pencereyi kaydır. Sermaye pencereler arası devrediyor, OOS eğrileri uç
  uca ekleniyor.
- **Nested grid-search**: hiperparametreler her pencerenin yalnız iç
  train/validation kesitinde seçiliyor, OOS seçime hiç katılmıyor. Skor
  komisyon sonrası Net Sharpe.
- **Permutation importance**: holdout üzerinde model-agnostik özellik önemi;
  gürültü özellikleri budanıyor.

## Deney Kayıtları ve Hipotez Reddi

Dokuz deneyin tamamı out-of-sample, komisyon ve slippage dahil, BTC/USDT
üzerinde 10.000 USDT başlangıçla koşuldu.

| # | Deney | OOS NET | Anahtar bulgu |
|---|---|---|---|
| 0 | Kontrol: aynı dönemde eğit+test | *+%24.0* | Holdout AUC 0.52 iken +%24 "kâr". In-sample yanılsamasının kanıtı, o yüzden tabloda duruyor |
| 1 | Walk-forward, sabit parametreler (1h, 14 özellik) | -%24.9 (0/11) | Dürüst zemin kuruldu |
| 2 | Nested grid-search (14 öz.) | -%12.8 (1/11) | Optimizasyon maliyeti yönetiyor, kenar yaratmıyor |
| 3 | + MTF özellikleri (22 öz.) | -%6.3 (4/11) | MTF trend eğimleri özellik öneminin %61'i |
| 4 | Budama + uzun ufuklar (16 öz.) | -%6.8 (3/11) | Val-OOS ters korelasyonu: 30 günlük pencereler istatistiksel olarak yetersiz |
| 5 | + Funding rate, 60g pencereler | -%6.7 (2/8) | Ters korelasyon kırıldı, plato sürdü |
| 6 | Maliyet A/B: futures komisyonları (%47 ucuz) | -%5.5 | Brüt PnL yine eksi, yani sorun maliyet değil sinyal |
| 7 | 4h taban + triple-barrier | -%6.8 (3/8) | Gerçekleşen RR 0.9, tasarlanan 2.0. Erken çıkış kazananları kesiyor |
| 8 | Bariyer-sadık yürütme | -%7.7 (1/8) | RR 1.63'e çıktı ama isabet %33.2'ye indi: 1:2 bariyerin sıfır-kenar denge noktası (⅓) |

İşlem yönetimini tarafsızlaştırdığımda isabet oranının tam olarak şans
seviyesine (⅓) yakınsaması, giriş sinyalinin zamanlama becerisi taşımadığını
gösterdi. Gradient boosting sınıfı modeller, 1h/4h OHLCV/MTF/funding özellikleri
ve yön etiketleri kombinasyonu BTC'de maliyet sonrası sömürülebilir bir kenar
üretmiyor.

Bu negatif ama sağlam bir sonuç. Süreç boyunca sistem kendi hatalarını da
yakaladı: in-sample yanılsaması, çift çıkış muhasebe hatası, val-OOS örneklem
yetersizliği, gerçekleşen risk/ödül uyumsuzluğu. Hiçbiri gerçek sermayeye mal
olmadı.

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

# Testler (sızıntı, triple-barrier, mikroyapı, maker, harvest)
python -m pytest tests/ -v
```

Bölüm 2 araçları (canlı veri toplama ve doğrulama; her biri kendi watchdog
`.bat` dosyasıyla 7/24 bırakılabilir):

```powershell
# L2 emir defteri arşivi biriktir (API anahtarı gerekmez)
python -m src.data.orderbook_collector --symbol BTC/USDT --depth 20

# Canlı mikroyapı dashboard'u (CSV'yi okur, veri hattına dokunmaz)
python -m src.tools.live_dashboard --minutes 30

# Mikroyapı özellikleriyle grid-search (OBI + maker yürütme)
python -m src.backtest.grid_search --days 14 --timeframe 5m --train-days 7 --test-days 2 --orderbook --maker

# Funding evreni indir + harvest ileriye dönük doğrulama defteri
python -m src.data.funding_universe --top 80 --days 400
python -m src.tools.harvest_tracker --once     # tek koşu
python -m src.tools.harvest_tracker --loop     # günlük döngü
```

## Bölüm 2: Mikroyapı, Maker Yürütme ve Funding Harvest

Yön tahmini hattı kapandıktan sonra araştırmayı iki yöne kaydırdım: yeni bilgi
kaynakları ve yürütme ekonomisi. Bu bölüm canlı veriyle yürütüldü; emir defteri
arşivi sıfırdan biriktirildi, stratejiler ileriye dönük doğrulamaya alındı.

### Emir Defteri Mikroyapısı (L2)

L2 geçmişi satın alınamıyor, biriktirmek gerekiyor. `orderbook_collector.py`
7/24 çalışarak Binance USDT-perp tahtasının ilk 20 kademesini 1 Hz örnekliyor
(15 günde yaklaşık 1.2M satır). `microstructure.py` bundan OBI, micro-price
(VWMP), spread ve türevlerini üretiyor; `orderbook_store.py` bar istatistiklerine
çevirip mumlara sızıntısız yapıştırıyor.

| Ölçüm | Sonuç |
|---|---|
| OBI → ileri getiri IC (10 sn) | +0.148 (444K gözlem; ilk 188K örneklemde +0.150) |
| IC bozunumu | 30 sn: +0.094, 1 dk: +0.072, 15 dk: +0.020 |
| Ablasyon (aynı bölme, TB h=12) | OHLCV-only AUC 0.506, +OBI 0.565 (katkı +0.059) |
| Koşullu kenar (\|OBI\|>0.95 + kalıcılık) | 1.2–1.4 bps |

Sinyal gerçek ve tekrarlanabilir, projenin en güçlü öngörü ilişkisi. Ama
büyüklüğü 1 bps civarında tavan yapıyor. Uçlaştırma yalnız 1.5 kat kazandırıyor,
oysa taker maliyetini (16 bps) aşmak için 6 kat gerekiyordu. Üst zaman dilimine
çıkmak da işe yaramıyor, çünkü sinyal 15 dakikada sönüyor. Pratik sonuç: OBI bir
market-maker sinyali, spread ödeyerek değil kazanarak işlenir.

### Maker Yürütme Modeli

Sinyalin doğal yeri maker tarafı olduğu için `ExecutionEngine`'e limit emir
modeli ekledim. Giriş limitte bekliyor ve fiyat seviyenin içinden geçerse
doluyor; "değdi, belki dolmuştur" varsayımı yok. `fill_window_bars` içinde
dolmazsa iptal ediliyor. TP çıkışı maker, SL ve zaman çıkışı taker.

| Aynı grid, tek değişken yürütme | Taker | Maker |
|---|---|---|
| OOS NET (6 gün) | -%3.30 | -%2.12 |
| Profit factor | 0.13 | 0.31 |
| Komisyon | 194 USDT | 122 USDT |
| Kazanma oranı | %26.3 | %29.0 |

Maliyet %37 düştü ve kayıp üçte bir azaldı. Ama iki isabet oranı da etiket taban
oranının (%36.5) altında kaldı, yani ters seçilim ölçülmüş oldu: limit emirler
fiyat aleyhe gelirken doluyor. Bu noktada sorun maliyet değil, seçilen işlemlerin
kalitesi.

### Funding Harvest

Yön tahmini yerine taşıma (carry): perp short ve spot long ile delta-nötr durup
funding tahsil etmek. BTC'de carry mütevazı kalıyor (iki yılda yıllık ~%4.8
nominal, periyotların %80'i pozitif) ve rejime bağlı. Asıl fırsat alt-perp
evreninde: taramada yıllık %10 üzeri ödeyen 82 perp vardı.

Look-ahead'siz cross-section backtest (79 perp, 400 gün, seçim yalnız t-1
verisiyle, maliyet bacak başına %0.3):

| Sepet | Rebalance | Brüt/yıl | NET/yıl | Max düşüş |
|---|---|---|---|---|
| Top-5 | günlük | +%46.7 | +%3.3 | %4.10 |
| Top-10 | 7 gün | +%30.0 | +%19.4 | %1.75 |
| Top-20 | 7 gün | +%18.5 | +%11.2 | %1.06 |

Rebalance hızı belirleyici: günlük yenileme kendini komisyona yediriyor.
Yarı dönem kontrolünde sonuç tutarlı (+%18.2 ve +%19.5).

Canlı doğrulama, backtest'in göremediği iki sorunu ortaya çıkardı ve ikisini de
kurala çevirdim:

1. **Likidite tabanı ve ekstrem tavan.** İlk canlı sepetin 6/10 ismi beş günde
   likit evrenden düştü, biri funding'i yıllık -%395'e çakıldı. Artık 24 saatlik
   ciro tabanı (25M USDT) ve yıllık %150 üzeri tuzak elemesi uygulanıyor.
2. **Negatif-çıkış kuralı.** Funding'i negatife dönen isim 7 günü beklemeden
   atılıyor. Tarihsel doğrulama: +%17.2 / MaxDD %1.78 yerine +%19.0 / MaxDD
   %0.94. Kör sıklaştırma ise +%6.1'e düşüyor, yani kazandıran şey sadece
   bozulanı hedef almak.

Canlı durum: `harvest_tracker.py` ileriye dönük paper defteri tutuyor, emir
göndermiyor. Kümülatif defter şu an ekside, ancak bunun baskın sebebi strateji
değil kesintili çalıştırma. Uzun boşluklarda tam rebalance maliyeti yazılırken
gelir yalnız kısa pencereden sayılıyordu; bu muhasebe hatasını sonradan
düzelttim. Kuralların tam işlediği temiz günlerde ölçüm, filtre öncesi yıllık
-%15 iken filtre ve negatif-çıkış sonrası yıllık +%31 nominal. Örneklem yalnız
3 gün, yani hüküm vermek için yetersiz. Doğrulama sürüyor ve gerçek sermaye
kullanılmadı.

### Canlı Altyapı

Üç bağımsız süreç çalışıyor, her biri kendi watchdog'uyla çökerse yeniden
başlıyor:

| Süreç | Görev |
|---|---|
| `orderbook_collector` | L2 tahta arşivi, 1 Hz, günlük CSV rotasyonu |
| `main` (paper bot) | 4h triple-barrier ML stratejisi, sanal para |
| `harvest_tracker` | Funding harvest ileriye dönük doğrulama defteri |

Canlı prova sadece araştırma sonucu değil mühendislik hatası da yakaladı: paper
bot günlerce çalışıyor görünüp tek bir mum kapanışı işlememişti, çünkü ccxt'nin
delta modunda kapanış tespiti sessizce başarısız oluyordu. Bu sınıf hatalar
ancak canlı akışta ortaya çıkıyor.

## Sıradaki Hatlar

- **On-chain veriler**: borsa giriş/çıkış akışları, aktif adresler, MVRV/SOPR.
  `merge_derivatives` deseni her yeni zaman serisi için genelleşiyor.
- **Open interest arşivi**: OI fetcher'ı zamanlanmış görevle çalıştırıp
  borsanın 30 günlük penceresini aşan kendi tarihçemi biriktirmek.
- **Market making**: maker iadeleriyle spread geliri ve envanter riski yönetimi.
  Mikroyapı bölümündeki sonucun işaret ettiği yön.
- **Meta-labeling**: `train_meta.py` hattı kurulu, birincil sinyalde gerçek bir
  kenar bulunduğunda devreye girecek.
- **Kesintisiz işletim**: VPS'e taşıma, ölçümün kesintiden zarar görmemesi için.

## Uyarı

Bu yazılım araştırma ve eğitim amaçlıdır. Kripto piyasaları yüksek risklidir ve
geçmiş performans gelecek sonuçların göstergesi değildir. Canlı kullanımdan
doğan sorumluluk kullanıcıya aittir.
