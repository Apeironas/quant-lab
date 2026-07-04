"""
PORTFÖY TAKİBİ

Tek doğruluk kaynağı: nakit, açık pozisyonlar, gerçekleşen kâr/zarar.
FillEvent'leri dinler ve muhasebeyi günceller - komisyonlar DAHİL.

Not: Paper modda bu defter simülasyonun kendisidir. Live modda ise borsadaki
gerçek bakiyenin yerel bir aynasıdır; periyodik olarak fetch_balance ile
mutabakat (reconciliation) eklemek canlıya geçmeden önce şarttır.
"""
from __future__ import annotations

import logging

from ..core.events import FillEvent, OrderSide

logger = logging.getLogger("portfolio")


class Portfolio:
    def __init__(self, config: dict) -> None:
        self.cash: float = config["portfolio"]["initial_cash"]
        #: symbol -> {"amount", "entry_price", "entry_fee"}
        self.positions: dict[str, dict] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.trade_count: int = 0
        #: Kapanan her işlemin kaydı - backtest raporu ve performans analizi için
        self.trades: list[dict] = []
        #: Piyasa fiyatları (equity hesabı için, MarketEvent'lerden güncellenir)
        self.last_prices: dict[str, float] = {}

    def equity(self) -> float:
        """Toplam portföy değeri = nakit + açık pozisyonların piyasa değeri."""
        value = self.cash
        for symbol, pos in self.positions.items():
            price = self.last_prices.get(symbol, pos["entry_price"])
            value += pos["amount"] * price
        return value

    async def on_market_event(self, event) -> None:
        """Fiyat aynasını güncelle (equity hesabının güncel kalması için)."""
        self.last_prices[event.symbol] = event.candle["close"]

    async def on_fill(self, event: FillEvent) -> None:
        """Her gerçekleşen işlemde muhasebe. ExecutionEngine SL/TP'yi zaten mühürledi."""
        self.total_fees += event.fee_quote
        self.trade_count += 1

        if event.side == OrderSide.BUY:
            cost = event.amount * event.fill_price + event.fee_quote
            self.cash -= cost
            self.positions[event.symbol] = {
                "amount": event.amount,
                "entry_price": event.fill_price,
                "entry_fee": event.fee_quote,  # net PnL hesabında düşülecek
            }
            logger.info("ALIŞ: %s %.6f @ %.2f | komisyon=%.4f | nakit=%.2f",
                        event.symbol, event.amount, event.fill_price,
                        event.fee_quote, self.cash)
        else:  # SELL - pozisyon kapanışı
            pos = self.positions.pop(event.symbol, None)
            if pos is None:
                # Muhasebe bütünlüğü: pozisyonsuz satış nakde İŞLENMEZ.
                # Buraya düşülüyorsa üst katmanda çift-emir sorunu var demektir.
                logger.warning("Pozisyonsuz SELL fill yok sayıldı: %s (%s)",
                               event.symbol, event.reason)
                self.total_fees -= event.fee_quote  # yukarıda eklenen komisyonu geri al
                self.trade_count -= 1
                return
            proceeds = event.amount * event.fill_price - event.fee_quote
            self.cash += proceeds
            if pos:
                # Net PnL: fiyat farkı - çıkış komisyonu - giriş komisyonu
                pnl = ((event.fill_price - pos["entry_price"]) * event.amount
                       - event.fee_quote - pos["entry_fee"])
                self.realized_pnl += pnl
                self.trades.append({
                    "symbol": event.symbol,
                    "amount": event.amount,
                    "entry_price": pos["entry_price"],
                    "exit_price": event.fill_price,
                    "pnl": pnl,
                    "fees": pos["entry_fee"] + event.fee_quote,
                    "reason": event.reason,
                })
                logger.info("SATIŞ: %s %.6f @ %.2f | işlem PnL=%+.2f | toplam PnL=%+.2f | equity=%.2f",
                            event.symbol, event.amount, event.fill_price,
                            pnl, self.realized_pnl, self.equity())

    def summary(self) -> str:
        return (f"Equity={self.equity():.2f} USDT | Nakit={self.cash:.2f} | "
                f"Gerçekleşen PnL={self.realized_pnl:+.2f} | "
                f"Toplam komisyon={self.total_fees:.2f} | İşlem sayısı={self.trade_count}")
