"""Harvest takipçisinin saf kurallarının birim testleri."""
from src.tools.harvest_tracker import (apply_negative_exit, rebalance_cost,
                                       select_basket)


def test_select_basket_threshold_and_ranking():
    trailing = {"A": 0.50, "B": 0.05, "C": 0.30, "D": 0.12, "E": -0.20}
    basket = select_basket(trailing, n=3, min_ann=0.10)
    assert basket == ["A", "C", "D"]          # eşik altı (B) ve negatif (E) elendi


def test_select_basket_fewer_than_n():
    basket = select_basket({"A": 0.50, "B": 0.02}, n=5, min_ann=0.10)
    assert basket == ["A"]                    # yeterli aday yoksa az isimle devam


def test_select_basket_extreme_cap():
    """trailing > max_ann olan tuzak isimler elenmeli (ortalamaya sert dönüş)."""
    trailing = {"TRAP": 2.00, "GOOD": 0.40, "OK": 0.20}
    basket = select_basket(trailing, n=5, min_ann=0.10, max_ann=1.50)
    assert basket == ["GOOD", "OK"]           # TRAP (%200) tavana takıldı


def test_select_basket_liquidity_floor():
    """Ciro tabanının altındaki isimler, funding yüksek olsa da seçilmez."""
    trailing = {"ILLIQ": 0.80, "LIQUID": 0.30}
    volumes = {"ILLIQ": 5_000_000, "LIQUID": 50_000_000}
    basket = select_basket(trailing, volumes=volumes, n=5, min_volume=25_000_000)
    assert basket == ["LIQUID"]               # ILLIQ yüksek funding'e rağmen elendi


def test_select_basket_no_volumes_skips_liquidity_filter():
    """volumes=None ise likidite filtresi uygulanmaz (geriye uyumluluk)."""
    basket = select_basket({"A": 0.50, "B": 0.30}, volumes=None, n=5)
    assert set(basket) == {"A", "B"}


def test_rebalance_cost_symmetric():
    # 2 çıkan + 2 giren = 4 değişim; N=10, %0.3/bacak -> 0.4 * 0.003
    cost = rebalance_cost({"A", "B", "C"}, {"A", "D", "E"}, n=10, per_side=0.003)
    assert abs(cost - 4 / 10 * 0.003) < 1e-12


def test_rebalance_cost_no_change_is_free():
    assert rebalance_cost({"A", "B"}, {"A", "B"}, n=10) == 0.0


def test_negative_exit_drops_flipped_and_backfills():
    """Funding'i negatife dönen isim atılır, boşalan slot taze isimle dolar."""
    holdings = ["A", "B", "C"]
    trailing = {"A": 0.30, "B": -0.05, "C": 0.20, "FRESH": 0.40}
    out = apply_negative_exit(holdings, trailing, n=3)
    assert "B" not in out                     # negatife döndü, atıldı
    assert "FRESH" in out                     # boşalan slot en yüksek taze ile doldu
    assert set(out) == {"A", "C", "FRESH"}


def test_negative_exit_drops_name_absent_from_universe():
    """Likit evrenden düşen (trailing'de olmayan) isim atılır."""
    holdings = ["A", "GONE", "C"]
    trailing = {"A": 0.30, "C": 0.20, "FRESH": 0.15}  # GONE yok = evrenden düştü
    out = apply_negative_exit(holdings, trailing, n=3)
    assert "GONE" not in out and "FRESH" in out


def test_negative_exit_all_healthy_unchanged():
    """Hepsi pozitif ve evrende ise sepet DEĞİŞMEZ (gereksiz komisyon yok)."""
    holdings = ["A", "B", "C"]
    trailing = {"A": 0.30, "B": 0.10, "C": 0.20, "OTHER": 0.50}
    out = apply_negative_exit(holdings, trailing, n=3)
    assert set(out) == {"A", "B", "C"}        # dolu ve sağlıklı -> dokunulmaz
