"""Conver the csv file into a data frame.

Dump the dataframe to the disk as a feather file.
"""
import pandas as pd

WEIGHTS = {
    0: 0.1,
    1: 0.15,
    2: 0.2,
    3: 0.25,
    4: 0.3,
    5: 0,
    6: 0
}


def main():
    # prices = pd.read_csv(
    #     "data/tetfp.csv", parse_dates=["日期"], encoding="big5")
    prices = pd.read_csv(
        "data/tetfp_fixed.csv", parse_dates=["日期"],
        dtype={"代碼": object}
    )
    prices.rename(columns={
        "日期": "date",
        "代碼": "symbol",
        "開盤價(元)": "open",
        "最高價(元)": "high",
        "最低價(元)": "low",
        "收盤價(元)": "close",
        "成交張數(張)": "volume"
    }, inplace=True)
    prices["symbol"] = prices.symbol.str.strip()
    prices["weekday"] = prices.date.dt.weekday
    print("Duplicates:\n", prices[prices.duplicated(
        ["symbol", "date"], keep=False)])
    prices = prices.drop_duplicates(["symbol", "date"])
    prices_last_day = prices.groupby(["symbol"]).close.shift(+1)
    prices["change"] = (
        prices.close - prices_last_day
    ) / prices_last_day * 100
    prices.dropna(inplace=True)
    prices["target1"] = 0
    prices.loc[prices.change > 0, "target1"] = 1
    prices.loc[prices.change < 0, "target1"] = -1
    # Print zero change stats
    prices["target1_zero"] = prices["target1"] == 0
    print("Stats of Chagne == Zero")
    print(prices.groupby("symbol")[
          "target1_zero"].agg(["count", "sum", "mean"]))
    # Assign weights
    prices["weight"] = prices.weekday.map(WEIGHTS)
    prices.drop(["中文簡稱", "target1_zero"], axis=1, inplace=True)
    print("Samples")
    print(prices.sample(10))
    prices.reset_index(drop=True).to_feather("cache/raw_prices.feather")


if __name__ == "__main__":
    main()
