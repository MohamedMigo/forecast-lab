"""Demand forecasting with honest evaluation: rolling-origin backtests against naive baselines."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def make_series(n_days: int = 900, seed: int = 5) -> pd.Series:
    r = np.random.default_rng(seed)
    t = np.arange(n_days)
    dow = np.array([0, 0, 0, 1, 2, 8, 6])[t % 7] * 3
    y = 100 + 0.08 * t + 12 * np.sin(2 * np.pi * t / 365.25) + dow + r.normal(0, 4, n_days)
    return pd.Series(y, index=pd.date_range("2024-01-01", periods=n_days, freq="D"), name="demand")


def naive(train: pd.Series, h: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def seasonal_naive(train: pd.Series, h: int, m: int = 7) -> np.ndarray:
    last = train.iloc[-m:].to_numpy()
    return np.tile(last, h // m + 1)[:h]


def _features(idx: pd.DatetimeIndex, start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"dow": idx.dayofweek, "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25),
                         "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25),
                         "t": (idx - start).days}, index=idx)


def gbm(train: pd.Series, h: int) -> np.ndarray:
    """Gradient boosting on calendar features + linear trend (fitted on detrended data so it extrapolates)."""
    start = train.index[0]
    t = np.arange(len(train))
    slope, icpt = np.polyfit(t, train.to_numpy(), 1)
    resid = train.to_numpy() - (icpt + slope * t)
    m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
    m.fit(_features(train.index, start), resid)
    fidx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    ft = np.arange(len(train), len(train) + h)
    return icpt + slope * ft + m.predict(_features(fidx, start))


MODELS = {"naive": naive, "seasonal_naive": seasonal_naive, "gbm_trend": gbm}


def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def smape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(2 * np.abs(a - b) / (np.abs(a) + np.abs(b))) * 100)


def backtest(y: pd.Series, h: int = 28, folds: int = 6, min_train: int = 500) -> pd.DataFrame:
    rows = []
    cuts = np.linspace(min_train, len(y) - h, folds).astype(int)
    for c in cuts:
        tr, te = y.iloc[:c], y.iloc[c:c + h]
        for name, f in MODELS.items():
            p = f(tr, h)
            rows.append({"model": name, "cut": c, "mae": mae(te, p), "smape": smape(te, p)})
    df = pd.DataFrame(rows)
    return df.groupby("model")[["mae", "smape"]].mean().round(3).sort_values("mae")


if __name__ == "__main__":
    print(backtest(make_series()))
