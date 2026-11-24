from forecast.models import MODELS, backtest, make_series


def test_shapes():
    y = make_series(600)
    for f in MODELS.values():
        assert len(f(y, 28)) == 28


def test_gbm_beats_baselines():
    res = backtest(make_series())
    assert res.index[0] == "gbm_trend"
    assert res.loc["gbm_trend", "mae"] < res.loc["naive", "mae"] * 0.7
