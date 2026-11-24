# Forecast Lab

Daily demand forecasting with **honest evaluation**: every model is scored with rolling-origin backtesting (6 folds, 28-day horizon) against naive baselines, so a fancy model has to earn its place.

## Models
- `naive`: repeat the last value
- `seasonal_naive`: repeat the last week
- `gbm_trend`: linear trend + gradient boosting on calendar features (day of week, yearly seasonality) fitted on the detrended series, so it extrapolates the trend instead of flat-lining

## Backtest results (synthetic series: trend + yearly cycle + weekly pattern + noise)
| model | MAE | sMAPE % |
|---|---|---|
| gbm_trend | 3.34 | 2.06 |
| seasonal_naive | 4.89 | 2.97 |
| naive | 10.70 | 6.44 |

## Run
```bash
pip install -r requirements.txt
python -m forecast.models
pytest -q
```
To use your own data, pass any `pd.Series` with a daily `DatetimeIndex` to `backtest(y)`.

The synthetic data is generated in `make_series()` so the repo works offline; the tests assert the boosted model beats the naive baseline by at least 30%.
