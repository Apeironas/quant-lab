"""
YÜRÜTME MOTORU

Görevleri:
  1. OrderEvent'leri gerçekleştirmek
       - paper modda: komisyon + slippage ile LOKAL simülasyon (borsaya gitmez)
       - live  modda: ccxt üzerinden gerçek emir (testnet=true ise sanal para)
  2. Açık pozisyonların SL/TP seviyelerini HER mumda kontrol etmek ve
     tetiklenince çıkış emri üretmek
  3. API istek sınırları: ccxt `enableRateLimit` ile otomatik; ayrıca tüm emir
     gönderimleri tek kuyruktan (bus) sıralı aktığı için burst oluşmaz.

Güvenlik kuralı: SL veya TP'si olmayan GİRİŞ emri kabul edilmez.
"""
from __future__ import annotations

import logging
import time

import ccxt.async_support as accxt

from ..core.event_bus import EventBus
from ..core.events import (FillEvent, MarketEvent, OrderEvent, OrderSide,
                           OrderType)
from ..portfolio.portfolio import Portfolio

logger = logging.getLogger("execution")


class ExecutionEngine:
    def __init__(self, config: dict, bus: EventBus, portfolio: Portfolio) -> None:
        self.bus = bus
        self.portfolio = portfolio
        self.mode: str = config["mode"]

        self.maker_fee = config["fees"]["maker_pct"] / 100.0
        self.taker_fee = config["fees"]["taker_pct"] / 100.0
        self.slippage = config["slippage"]["bps"] / 10_000.0

        # MAKER YÜRÜTME MODELİ (yalnız paper): girişler limit emir olarak
        # bekletilir; muhafazakâr dolum kuralı = fiyat seviyenin İÇİNDEN
        # geçmeli (low < limit). TP çıkışı maker, SL/zaman çıkışı taker.
        exec_cfg = config.get("execution", {})
        self.entry_mode: str = exec_cfg.get("entry_mode", "market")
        self.fill_window_bars: int = exec_cfg.get("fill_window_bars", 3)

        #: SL/TP izleme defteri: symbol -> {"stop_loss", "take_profit", "amount"}
        #: Portföyden ayrı tutulur: portföy muhasebe yapar, bu defter tetik izler.
        self._watchlist: dict[str, dict] = {}
        #: Bekleyen limit girişler (maker modu): symbol -> emir + kalan bar
        self._pending: dict[str, dict] = {}

        self._exchange: accxt.Exchange | None = None
        self._config = config

    async def connect(self) -> None:
        """Live modda borsa bağlantısını kur. Paper modda gerek yok."""
        if self.mode != "live":
            logger.info("Paper mod: emirler lokal simüle edilecek (komisyon=%%%.2f, slippage=%d bps)",
                        self.taker_fee * 100, round(self.slippage * 10_000))
            return
        ex_cfg = self._config["exchange"]
        exchange_class = getattr(accxt, ex_cfg["id"])
        self._exchange = exchange_class({
            "apiKey": ex_cfg["api_key"],
            "secret": ex_cfg["api_secret"],
            "enableRateLimit": True,  # ccxt, borsanın istek sınırına göre kendini frenler
        })
        if ex_cfg.get("testnet", True):
            self._exchange.set_sandbox_mode(True)
        logger.info("Live mod: %s bağlantısı hazır (testnet=%s)", ex_cfg["id"], ex_cfg.get("testnet"))

    # ------------------------------------------------------------------ #
    #  Emir işleme
    # ------------------------------------------------------------------ #
    async def on_order(self, order: OrderEvent) -> None:
        # Güvenlik kapısı: girişler SL/TP olmadan işleme alınmaz
        if order.side == OrderSide.BUY and (order.stop_loss <= 0 or order.take_profit <= 0):
            logger.error("EMİR REDDEDİLDİ: SL/TP eksik - %s", order)
            return

        # Çift çıkış koruması: aynı mumda hem SL/TP hem strateji EXIT'i
        # tetiklenirse ikinci SELL, pozisyon çoktan kapandığı için yok sayılır.
        # (Bu olmadan simülasyon "olmayan coini satıp" yoktan para basar.)
        if order.side == OrderSide.SELL and order.symbol not in self.portfolio.positions:
            logger.warning("SELL yok sayıldı: %s için açık pozisyon yok (%s)",
                           order.symbol, order.reason)
            return

        # --- MAKER GİRİŞ (paper): anında dolum yok, limit emir bekletilir ---
        if (self.mode == "paper" and self.entry_mode == "maker"
                and order.side == OrderSide.BUY):
            if order.symbol in self._pending:
                logger.debug("%s: zaten bekleyen limit emir var, yenisi yok sayıldı",
                             order.symbol)
                return
            ref_price = order.price or self.portfolio.last_prices.get(order.symbol, 0.0)
            if ref_price <= 0:
                return
            self._pending[order.symbol] = {
                "order": order, "limit_price": ref_price,
                "bars_left": self.fill_window_bars,
            }
            logger.info("%s: LIMIT giriş bekletildi @ %.2f (pencere=%d bar)",
                        order.symbol, ref_price, self.fill_window_bars)
            return

        t0 = time.perf_counter()
        if self.mode == "paper":
            fill = self._simulate_fill(order)
        else:
            fill = await self._live_fill(order)
        if fill is None:
            return
        fill.latency_ms = (time.perf_counter() - t0) * 1000

        # SL/TP/dikey bariyer izleme defterini güncelle
        if order.side == OrderSide.BUY:
            self._watchlist[order.symbol] = {
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "amount": fill.amount,
                # Dikey bariyer sayacı: her yeni mumda azalır, 0'da zorla çıkış.
                # None = zaman durdurucu yok (time_stop_bars=0 gönderilmişse).
                "bars_left": order.time_stop_bars if order.time_stop_bars > 0 else None,
            }
        else:
            self._watchlist.pop(order.symbol, None)

        await self.bus.publish(fill)

    def _simulate_fill(self, order: OrderEvent) -> FillEvent:
        """
        GERÇEKÇİ paper-trading gerçekleşmesi:
          - Piyasa emri: son fiyata ALEYHTE slippage eklenir
            (alırken daha pahalı, satarken daha ucuz)
          - Komisyon: piyasa emri = taker, limit = maker
        Bu maliyetleri atlamak, backtest'te kârlı görünen stratejilerin canlıda
        zarar etmesinin 1 numaralı sebebidir - o yüzden burada zorunlular.
        """
        ref_price = order.price or self.portfolio.last_prices.get(order.symbol, 0.0)
        if ref_price <= 0:
            logger.error("Simülasyon için referans fiyat yok: %s", order.symbol)
            return None

        if order.order_type == OrderType.MARKET:
            direction = 1 if order.side == OrderSide.BUY else -1
            fill_price = ref_price * (1 + direction * self.slippage)
            fee_rate = self.taker_fee
        else:
            fill_price = ref_price  # limit emir kendi fiyatından dolar (basitleştirme)
            fee_rate = self.maker_fee

        fee = order.amount * fill_price * fee_rate
        return FillEvent(
            symbol=order.symbol, side=order.side, amount=order.amount,
            fill_price=fill_price, fee_quote=fee,
            order_id=f"paper-{int(time.time()*1000)}", reason=order.reason,
        )

    async def _live_fill(self, order: OrderEvent) -> FillEvent | None:
        """Gerçek emir gönderimi (testnet dahil). Hata olursa loglar, botu düşürmez."""
        try:
            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.value,
                side=order.side.value,
                amount=order.amount,
                # Piyasa emrine fiyat gönderilmez (bazı borsalar reddeder);
                # order.price piyasa emirlerinde sadece simülasyon referansıdır
                price=order.price if order.order_type == OrderType.LIMIT else None,
            )
            # Not: borsa tarafında OCO (SL+TP) emri kurmak borsaya özgüdür.
            # Bu iskelet SL/TP'yi kendi izleyip piyasa emriyle çıkar; canlıda
            # bot çökerse koruma kalmasın istemiyorsan borsa tarafı OCO ekle.
            return FillEvent(
                symbol=order.symbol, side=order.side,
                amount=float(result.get("filled") or order.amount),
                fill_price=float(result.get("average") or result.get("price") or 0),
                fee_quote=float((result.get("fee") or {}).get("cost") or 0),
                order_id=str(result.get("id", "")), reason=order.reason,
            )
        except Exception:
            logger.exception("Emir gönderimi BAŞARISIZ: %s", order)
            return None

    # ------------------------------------------------------------------ #
    #  SL/TP izleme - her yeni mumda çağrılır
    # ------------------------------------------------------------------ #
    async def on_market_event(self, event: MarketEvent) -> None:
        # --- Bekleyen LIMIT girişler (maker modu) ---
        pending = self._pending.get(event.symbol)
        if pending is not None:
            order = pending["order"]
            if event.candle["low"] < pending["limit_price"]:
                # MUHAFAZAKÂR dolum: fiyat seviyenin İÇİNDEN geçti (low < limit).
                # "Değdi ama geçmedi" iyimserliği yok - kuyruk pozisyonu bilinemez.
                fill_price = pending["limit_price"]
                fee = order.amount * fill_price * self.maker_fee  # maker ücreti, slippage YOK
                del self._pending[event.symbol]
                self._watchlist[event.symbol] = {
                    "stop_loss": order.stop_loss,
                    "take_profit": order.take_profit,
                    "amount": order.amount,
                    "bars_left": order.time_stop_bars if order.time_stop_bars > 0 else None,
                }
                logger.info("%s: LIMIT giriş DOLDU @ %.2f (maker, ücret=%.4f)",
                            event.symbol, fill_price, fee)
                await self.bus.publish(FillEvent(
                    symbol=event.symbol, side=OrderSide.BUY, amount=order.amount,
                    fill_price=fill_price, fee_quote=fee,
                    order_id=f"paper-maker-{int(time.time() * 1000)}",
                    reason=order.reason + " (maker dolum)",
                ))
                return  # dolum barında bariyer kontrolü yapılmaz (muhafazakâr)
            pending["bars_left"] -= 1
            if pending["bars_left"] <= 0:
                del self._pending[event.symbol]
                logger.info("%s: LIMIT giriş DOLMADI, iptal (fiyat %.2f seviyesine inmedi)",
                            event.symbol, pending["limit_price"])

        pos = self._watchlist.get(event.symbol)
        if pos is None:
            return

        low, high = event.candle["low"], event.candle["high"]
        exit_reason: str | None = None
        # Muhafazakâr varsayım: aynı mumda ikisi de değdiyse ÖNCE SL sayılır.
        # Maker modunda TP için katı kural: seviye İÇİNDEN geçilmeli (high > tp).
        tp_hit = (high > pos["take_profit"]) if self.entry_mode == "maker" \
            else (high >= pos["take_profit"])
        if low <= pos["stop_loss"]:
            exit_reason = "STOP_LOSS tetiklendi"
        elif tp_hit:
            exit_reason = "TAKE_PROFIT tetiklendi"
        elif pos.get("bars_left") is not None:
            # DİKEY BARİYER (v2.1): bar sayacı işler; süre dolunca mum
            # kapanışından zorla çıkış - triple-barrier etiketinin üçüncü
            # bariyeriyle birebir aynı kural (t+horizon kapanışında çık).
            pos["bars_left"] -= 1
            if pos["bars_left"] <= 0:
                exit_reason = "TIME_STOP (dikey bariyer)"
        if exit_reason is None:
            return

        # Çift tetiklenmeyi önle: emri üretmeden önce izlemeden çıkar
        details = self._watchlist.pop(event.symbol)
        logger.info("%s: %s (SL=%.2f TP=%.2f low=%.2f high=%.2f)",
                    event.symbol, exit_reason, details["stop_loss"],
                    details["take_profit"], low, high)
        # Gerçekçilik: SL/TP çıkışı mum kapanışından değil TETİK SEVİYESİNDEN
        # simüle edilir; dikey bariyer (time stop) ise tanımı gereği mum
        # KAPANIŞINDAN çıkar. Slippage her durumda aleyhte eklenir.
        if exit_reason.startswith("STOP_LOSS"):
            trigger_price = details["stop_loss"]
        elif exit_reason.startswith("TAKE_PROFIT"):
            trigger_price = details["take_profit"]
        else:  # TIME_STOP
            trigger_price = event.candle["close"]
        # TP çıkışı maker modunda LIMIT'tir (kendi fiyatından, maker ücreti,
        # slippage yok); SL ve zaman çıkışı her zaman taker (panik emri).
        exit_type = (OrderType.LIMIT
                     if exit_reason.startswith("TAKE_PROFIT") and self.entry_mode == "maker"
                     else OrderType.MARKET)
        await self.bus.publish(OrderEvent(
            symbol=event.symbol, side=OrderSide.SELL,
            order_type=exit_type, amount=details["amount"],
            price=trigger_price,
            stop_loss=details["stop_loss"], take_profit=details["take_profit"],
            reason=exit_reason,
        ))

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
