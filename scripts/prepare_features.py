"""Feature engineering utility functions."""
from datetime import datetime, timedelta

import pandas as pd


def collect_labels(df, start):
    res = None
    for i in range(5):
        target_date = start + timedelta(days=i)
        df_local = df[df.date == target_date]
        if df_local.shape[0] == 0:
            print("Skipping", target_date)
            continue
        tmp = df[
            df.date == (start + timedelta(days=i))
        ][["symbol", "close", "change"]].rename(
            columns={"close": f"target2_{i}", "change": f"target2_{i}_change"}
        ).set_index(["symbol"])
        if res is None:
            res = tmp
        else:
            res = res.join(tmp)
        res = res.join(df[
            df.date == (start + timedelta(days=i))
        ][["symbol", "target1"]].rename(
            columns={"target1": f"target1_{i}"}
        ).set_index(["symbol"]))
    not_null = (~res.isnull()).sum().sum()
    print(f"# of non-NA values: {not_null}")
    if res is None:
        return pd.DataFrame()
    return res


def collect_target1_stats(df, lookback_weeks, end_date):
    start_date = end_date - timedelta(days=7*lookback_weeks)
    df = df[["symbol", "date", "weekday", "target1"]][
        (df.date >= start_date) &
        (df.date < end_date)
    ].drop(["date"], axis=1)

    def rename_columns(df_counts, suffix=""):
        mapping = {}
        for col in df_counts.columns:
            mapping[col] = f"t1_{col}_{lookback_weeks}{suffix}"
        return df_counts.rename(columns=mapping)

    global_counts = rename_columns(
        df.groupby(["symbol", "target1"]).size().astype("uint16").unstack(-1)
    ).fillna(0)
    total_counts = df.groupby(["symbol"]).size()
    global_counts = global_counts.apply(lambda x: x / total_counts.values)
    return global_counts


def collect_close_stats(df, lookback_weeks, end_date):
    start_date = end_date - timedelta(days=7*lookback_weeks)
    df = df[["symbol", "date", "weekday", "change"]][
        (df.date >= start_date) & (df.date < end_date)
    ]
    df_res = df.groupby("symbol")["change"].agg({
        "min", "max", "mean", "median", "std"
    })
    df_res.columns = [f"change_{x}_{lookback_weeks}" for x in df_res.columns]
    return df_res


def collect_last_5_days(df, start):
    res = []
    df = df[df.date < start].groupby(
        "symbol")[["symbol", "change", "close"]].tail(5)
    df_res = df.groupby("symbol").apply(
        lambda x: pd.Series({
            "change_m1": x["change"].iloc[-1],
            "change_m2": x["change"].iloc[-2],
            "change_m3": x["change"].iloc[-3],
            "change_m4": x["change"].iloc[-4],
            "change_m5": x["change"].iloc[-5],
            "latest": x["close"].iloc[-1]
        } if x.shape[0] == 5 else pd.Series({
            "change_m1": None,
            "change_m2": None,
            "change_m3": None,
            "change_m4": None,
            "change_m5": None,
            "latest": None
        }))
    )
    return df_res.dropna()


def collect_chunk(df, start, test=False):
    df_features = collect_target1_stats(df, 14, start)
    df_features = df_features.join(collect_close_stats(
        df, 7, start))
    df_features = df_features.join(collect_last_5_days(
        df, start))
    if test is False:
        df_features = df_features.join(
            collect_labels(df, start)
        )
    return df_features


def main(steps=50):
    prices = pd.read_feather("cache/raw_prices.feather")
    start = datetime(2018, 4, 23)
    res = []
    for i in range(steps):
        date = start - timedelta(days=7*i)
        print("Collecting", date)
        tmp = collect_chunk(prices, date, test=False)
        tmp["start"] = date
        res.append(tmp.reset_index())
    df_final = pd.concat(res, axis=0)
    df_final = df_final[~df_final.latest.isnull()].reset_index(drop=True)
    print(df_final.sample(10))
    print("Total rows:", df_final.shape[0])
    df_final.to_feather("cache/train.feather")


if __name__ == "__main__":
    main()
