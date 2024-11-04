from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from config import WEEKS_IN_YEAR
from metrics import mase, mape, smape


def load_time_series_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = df.index.inferred_freq
    return df


def plot_predictions(ts_signal: pd.DataFrame, predictions: Dict[str,pd.DataFrame], output_path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(15, 6))
    lns = []
    lns1 = ax1.plot(ts_signal.loc[:"2023-01-01"], label="historical data", linewidth=3, color="black")
    lns += [lns1[0]]
    lns1 = ax1.plot(ts_signal.loc["2023-01-01":"2024-01-01"], label="ground truth", linewidth=3, color="grey")
    lns += [lns1[0]]
    for model_name, model_prediction in predictions.items():
        lns1 = ax1.plot(model_prediction, label=model_name, linewidth=3)
        lns += [lns1[0]]
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize="large")
    ax1.tick_params(axis="x", which="major", labelsize=15)
    ax1.set_xlabel("Date", size=15)
    ax1.set_ylabel("share of category", size=15)
    fig.tight_layout()
    fig.savefig(output_path, format="png")

def eval_predictions(ts_signal: pd.DataFrame, predictions: Dict[str,pd.DataFrame], output_path: str) -> None:
    error_metrics = pd.DataFrame(columns=["mase", "mape", "smape"])
    for model_name, model_prediction in predictions.items():
        error_metrics.loc[model_name] = [
            mase(ts_signal, model_prediction, seasonality=WEEKS_IN_YEAR)[0].round(4),
            mape(ts_signal, model_prediction)[0].round(4),
            smape(ts_signal, model_prediction)[0].round(4)
        ]
    error_metrics.sort_values(by="mase", ascending=False).to_csv(output_path)