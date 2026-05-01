import pytest
from app.collector import fetch_exchange_rates, calculate_change_percent

def test_fetch_returns_dict():
    """환율 데이터가 딕셔너리로 반환"""
    result = fetch_exchange_rates()
    assert isinstance(result, dict)

def test_fetch_contains_target_currencies():
    """주요 통화가 포함되어 있는지 확인"""
    result = fetch_exchange_rates()
    for currency in ["KRW", "EUR", "JPY"]:
        assert currency in result

def test_fetch_rates_are_positive():
    """환율 값이 양수인지 확인"""
    result = fetch_exchange_rates()
    for currency, rate in result.items():
        assert rate > 0

def test_calculate_change_percent_increase():
    """상승률 계산"""
    result = calculate_change_percent(1360.0, 1350.0)
    assert result == pytest.approx(0.7407, rel=1e-3)

def test_calculate_change_percent_decrease():
    """하락률 계산"""
    result = calculate_change_percent(1340.0, 1350.0)
    assert result == pytest.approx(-0.7407, rel=1e-3)

def test_calculate_change_percent_zero_previous():
    """이전 데이터 없을 때 0 반환"""
    result = calculate_change_percent(1350.0, 0)
    assert result == 0.0

def test_calculate_change_percent_no_change():
    """변화 없을 때 0 반환"""
    result = calculate_change_percent(1350.0, 1350.0)
    assert result == 0.0