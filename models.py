import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from pydantic import BaseModel

from config import WEEKS_IN_YEAR, PREDICTION_ONE_YEAR


class EtsForecastModel(BaseModel):
    """Class defining an ETS (exponential smoothing) model.

        Args:
            season: define the seasonality length.
            horizon: define the forecast horizon of the Ets model.
    """

    season: int = WEEKS_IN_YEAR
    horizon: int = PREDICTION_ONE_YEAR

    def fit_predict_single_ts(self, ts_signal: pd.Series) -> pd.Series:
        """ Fit ETS models and return its prediction.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single_forecast: A pd.Series gathering the ETS prediction.
        """

        model = ExponentialSmoothing(
            ts_signal, seasonal_periods=self.season, seasonal="add", trend="add"
        )
        fitted_model = model.fit()
        single_forecast = fitted_model.forecast(self.horizon)
        single_forecast.name = ts_signal.name
        return single_forecast


class SnaiveForecastModel(BaseModel):
    """Class defining a seasonal naive model that repeats the values of the past year as its prediction.

        Args:
            horizon: define the forecast horizon of the Naive model.
    """
    season: int = WEEKS_IN_YEAR
    horizon: int = PREDICTION_ONE_YEAR

    def predict(self, ts_signal: pd.Series) -> pd.Series:
        """ Compute prediction of a Naive model.

            Args:
                ts_signal: A pd.Series gathering a single time series signal.

            Returns:
                single_forecast: A pd.Series gathering the seasonal Naive prediction.

        """

        historical_data_past_year = ts_signal.tail(52)
        nb_repeat, part_repeat = divmod(52, 52)
        naive_forecast = [historical_data_past_year] * nb_repeat + [
            historical_data_past_year[:part_repeat]
        ]
        single_forecast = pd.concat(naive_forecast, axis=0, ignore_index=True)

        forecast_index = pd.date_range(
            start=ts_signal.index[-1], periods=52 + 1, freq=ts_signal.index.freq
        )[1:]
        single_forecast.index = forecast_index
        return single_forecast