import pandas as pd

from models import EtsForecastModel, SnaiveForecastModel
from utils import plot_predictions, load_time_series_csv, eval_predictions

fashion_ts = load_time_series_csv("chunky_sneakers.csv")
fashion_ts_train = fashion_ts.loc[:"2023-01-01"]

ets_model = EtsForecastModel()
ets_prediction = pd.DataFrame(ets_model.fit_predict_single_ts(fashion_ts_train["chunky_sneakers"]))
snaive_model = SnaiveForecastModel()
snaive_prediction = pd.DataFrame(snaive_model.predict(fashion_ts_train["chunky_sneakers"]))

plot_predictions(
    ts_signal=fashion_ts,
    predictions={
        "Exp. Smooth. prediction": ets_prediction,
        "Snaive prediction": snaive_prediction
    },
    output_path="chunky_sneakers.png"
)

eval_predictions(
    ts_signal=fashion_ts,
    predictions={
        "Exp. Smooth.": ets_prediction,
        "Snaive": snaive_prediction
    },
    output_path="error_metrics.csv"
)
