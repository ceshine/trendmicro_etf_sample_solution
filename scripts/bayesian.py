"""Model training, evaluation, and prediction."""
from datetime import datetime

import pandas as pd
import numpy as np
import pystan

from prepare_features import collect_chunk

WEIGHTS = {
    0: 0.1,
    1: 0.15,
    2: 0.2,
    3: 0.25,
    4: 0.3
}
STAN_FILE = "scripts/bayesian_linear_model.stan"
MODEL = pystan.StanModel(file=STAN_FILE)


def fit_model(df, y):
    mask = ~y.isnull()
    fit = MODEL.sampling(
        data={
            "x_1": df[mask]["change_m1"],
            "x_2": df[mask]["change_m2"],
            "x_3": df[mask]["change_m3"],
            "x_4": df[mask]["change_m4"],
            "x_5": df[mask]["change_m5"],
            "t": y[mask],
            "N": df[mask].shape[0],
            "df": 2.6
        },
        iter=2000, chains=8)
    result = fit.extract(permuted=True)
    bias = result["w_1_0"][np.newaxis, :]
    weights = np.array([
        result["w_1_1"],
        result["w_1_2"],
        result["w_1_3"],
        result["w_1_4"],
        result["w_1_5"]
    ])
    return bias, weights


def get_x_matrix(df):
    return np.array([
        df["change_m1"].values,
        df["change_m2"].values,
        df["change_m3"].values,
        df["change_m4"].values,
        df["change_m5"].values,
    ]).transpose(1, 0)


def fill_predictions(df_local, preds_local, weight, bias, i):
    samples = get_x_matrix(df_local) @ weight + bias
    pred_changes = np.mean(samples, axis=1)
    print(pd.Series(pred_changes).describe())
    preds_local[f"target2_{i}"] = (
        df_local["latest"] *
        (1 + pred_changes / 100)
    )
    preds_local[f"target1_{i}"] = 0
    preds_local.loc[pred_changes < -0.01, f"target1_{i}"] = -1
    preds_local.loc[pred_changes > 0.01, f"target1_{i}"] = 1


def predict(df, df_test=None):
    train_preds = pd.DataFrame()
    test_preds = pd.DataFrame()
    for i in range(5):
        bias, weight = fit_model(df, df[f"target2_{i}_change"])
        print("Weight means:", np.mean(weight, axis=1))
        print("Weight medians:", np.median(weight, axis=1))
        fill_predictions(df, train_preds, weight, bias, i)
        if df_test is not None:
            fill_predictions(df_test, test_preds, weight, bias, i)
    return train_preds, test_preds


def evaluate():
    df = pd.read_feather("cache/train.feather")
    preds, _ = predict(df)
    # By day
    scores = pd.DataFrame()
    for i in range(5):
        scores[f"t1_{i}"] = (
            df[f"target1_{i}"] == preds[f"target1_{i}"]
        ) * 0.5 * WEIGHTS[i]
        scores[f"t2_{i}"] = ((
            df[f"target2_{i}"] -
            np.abs(df[f"target2_{i}"] - preds[f"target2_{i}"])
        ) / df[f"target2_{i}"] * 0.5) * WEIGHTS[i]
    print(scores.describe())
    # By week
    scores["t1"] = np.nansum(
        [scores[f"t1_{i}"].values for i in range(5)], axis=0
    )
    scores["t2"] = np.nansum(
        [scores[f"t2_{i}"].values for i in range(5)], axis=0
    )
    scores["symbol"] = df["symbol"]
    scores["total"] = scores["t1"] + scores["t2"]
    print("=" * 20)
    print("Valid entries:", scores.dropna().shape[0])
    print("=" * 20)
    symbol_scores = scores.groupby("symbol")[["t1", "t2", "total"]].mean()
    print(symbol_scores)
    print(symbol_scores["total"].sum())
    print(scores.symbol.nunique())
    # Overall
    print(scores[["t1", "t2", "total"]].mean())


def make_submission():
    target_date = datetime(2018, 6, 18)
    prices = pd.read_feather("cache/raw_prices.feather").dropna()
    print(prices.tail())
    df_test = collect_chunk(prices, target_date, test=True).reset_index()
    print(df_test[["symbol", "latest"]])
    df_train = pd.read_feather("cache/train.feather")
    print(df_test.columns)
    _, preds = predict(df_train, df_test)
    preds["symbol"] = df_test["symbol"]
    name_mapping = {
        "symbol": "ETFid",
        "target1_0": "Mon_ud",
        "target2_0": "Mon_cprice",
        "target1_1": "Tue_ud",
        "target2_1": "Tue_cprice",
        "target1_2": "Wed_ud",
        "target2_2": "Wed_cprice",
        "target1_3": "Thu_ud",
        "target2_3": "Thu_cprice",
        "target1_4": "Fri_ud",
        "target2_4": "Fri_cprice",
    }
    preds.rename(columns=name_mapping, inplace=True)
    preds = preds[[
        "ETFid", "Mon_ud", "Mon_cprice", "Tue_ud", "Tue_cprice",
        "Wed_ud", "Wed_cprice", "Thu_ud", "Thu_cprice",
        "Fri_ud", "Fri_cprice"]]
    print(preds.head())
    preds.to_csv("cache/baseline.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    # evaluate()
    make_submission()
