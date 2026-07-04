"""
RİSK YÖNETİCİSİ - strateji ile emir arasındaki zorunlu kapı.

Hiçbir sinyal doğrudan emre dönüşmez; önce buradan geçer:
  1. Devre kesici: günlük zarar limiti aşıldı mı? -> işlem yok
  2. Açık pozisyon sayısı sınırda mı? -> işlem yok
  3. Pozisyon boyutlandırma: "her işlemde portföyün en fazla %X'i riske atılır"
  4. Zorunlu SL/TP fiyatlarını hesaplar ve emre mühürler

Boyutlandırma mantığı (sabit oranlı risk / fixed-fractional):
    riske_edilen = portföy * risk_per_trade_pct
    miktar       = riske_edilen / (giriş_fiyatı - stop_fiyatı)
Yani SL ne kadar yakınsa pozisyon o kadar büyük olabilir; SL uzaksa küçülür.
Her durumda kayıp senaryosu portföyün ~%X'i ile sınırlı kalır.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from ..core.events import (OrderEvent, OrderSide, OrderType, SignalDirection,
                           SignalEvent)
from ..portfolio.portfolio import Portfolio

logger = logging.getLogger("risk")


class RiskManager:
    def __init__(self, config: dict, portfolio: Portfolio) -> None:
        r = config["risk"]
        self.risk_per_trade = r["risk_per_trade_pct"] / 100.0
        self.max_position = r["max_position_pct"] / 100.0
        self.max_open_positions = r["max_open_positions"]
        self.sl_pct = r["stop_loss_pct"] / 100.0
        self.tp_pct = r["take_profit_pct"] / 100.0
        self.max_daily_loss = r["max_daily_loss_pct"] / 100.0
        self.portfolio = portfolio
        # "Gün" duvar saatinden değil, işlenen olayın zaman damgasından türetilir.
        # Böylece aynı kod canlıda bugünü, backtest'te tarihsel günü kullanır.
        self._day: date | None = None
        self._day_start_equity = portfolio.equity()

    def _daily_circuit_breaker_tripped(self, event_day: date) -> bool:
        """Günlük zarar limiti kontrolü. Gün değiştiyse sayaç sıfırlanır."""
        if event_day != self._day:
            self._day = event_day
            self._day_start_equity = self.portfolio.equity()
        loss = 1.0 - self.portfolio.equity() / self._day_start_equity
        if loss >= self.max_daily_loss:
            logger.warning("DEVRE KESİCİ: günlük zarar %%%.1f limiti aştı, bugün yeni işlem yok",
                           loss * 100)
            return True
        return False

    def evaluate(self, signal: SignalEvent) -> OrderEvent | None:
        """Sinyali değerlendir: ya boyutlandırılmış OrderEvent döner ya da None (ret)."""

        # --- ÇIKIŞ sinyali: pozisyon varsa her koşulda kapatılır (risk azaltıcı) ---
        if signal.direction == SignalDirection.EXIT:
            pos = self.portfolio.positions.get(signal.symbol)
            if pos is None:
                return None  # kapatılacak pozisyon yok
            return OrderEvent(
                symbol=signal.symbol, side=OrderSide.SELL,
                order_type=OrderType.MARKET, amount=pos["amount"],
                reason="strateji EXIT sinyali",  # çıkışlarda SL/TP gerekmez
            )

        # --- GİRİŞ sinyali: tüm kontrollerden geçmeli ---
        event_day = datetime.fromtimestamp(signal.timestamp_ms / 1000, tz=timezone.utc).date()
        if self._daily_circuit_breaker_tripped(event_day):
            return None
        if len(self.portfolio.positions) >= self.max_open_positions:
            logger.info("Ret: azami açık pozisyon sayısına (%d) ulaşıldı", self.max_open_positions)
            return None
        if signal.symbol in self.portfolio.positions:
            logger.info("Ret: %s pozisyonu zaten açık (piramitleme kapalı)", signal.symbol)
            return None
        if signal.direction == SignalDirection.SHORT:
            logger.info("Ret: spot iskelette SHORT desteklenmiyor (margin/futures eklenince açılır)")
            return None

        entry = signal.price
        # Strateji özel SL/TP önerdiyse onu kullan, yoksa config yüzdeleri
        stop_loss = signal.stop_loss or entry * (1 - self.sl_pct)
        take_profit = signal.take_profit or entry * (1 + self.tp_pct)

        # --- Pozisyon boyutlandırma ---
        equity = self.portfolio.equity()
        risk_amount_quote = equity * self.risk_per_trade      # riske edilen USDT
        per_unit_risk = entry - stop_loss                     # birim başına olası kayıp
        if per_unit_risk <= 0:
            logger.error("Geçersiz SL: stop (%f) girişin (%f) üstünde", stop_loss, entry)
            return None
        amount = risk_amount_quote / per_unit_risk

        # Üst sınır: pozisyon değeri portföyün %max_position'ını aşamaz
        max_amount = (equity * self.max_position) / entry
        amount = min(amount, max_amount)

        # Nakit yeterliliği (spot: alım için USDT gerekli)
        cost = amount * entry
        if cost > self.portfolio.cash:
            amount = self.portfolio.cash / entry * 0.99  # ufak pay: komisyon için
            if amount * entry < 10:  # Binance asgari emir tutarı ~10 USDT
                logger.info("Ret: yetersiz bakiye (%.2f USDT)", self.portfolio.cash)
                return None

        logger.info("ONAY: %s LONG %.6f adet (risk=%.2f USDT, SL=%.2f, TP=%.2f, T=%d bar)",
                    signal.symbol, amount, risk_amount_quote, stop_loss, take_profit,
                    signal.time_stop_bars)
        return OrderEvent(
            symbol=signal.symbol, side=OrderSide.BUY,
            order_type=OrderType.MARKET, amount=amount,
            stop_loss=stop_loss, take_profit=take_profit,
            time_stop_bars=signal.time_stop_bars,  # dikey bariyer emre mühürlenir
            reason=f"sinyal güç={signal.strength:.2f}",
        )
