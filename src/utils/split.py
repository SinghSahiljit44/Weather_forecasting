import pandas as pd

class Split:
    TRAIN_END = 2017  # train: 1990-2017
    VAL_END = 2021    # validation: 2018-2021, test: 2022+

    df: pd.DataFrame
    y: pd.Series
    train: pd.Series
    val: pd.Series
    test: pd.Series

    def __init__(self, df, target):
        # daily frequency, so shift(-h) is always h days ahead (missing days become NaN rows)
        self.df = df.sort_index().asfreq("D")
        self.y = self.df[target]
        year = pd.Series(self.df.index.year, index=self.df.index)

        self.train = year <= self.TRAIN_END
        self.val = (year > self.TRAIN_END) & (year <= self.VAL_END)
        self.test = year > self.VAL_END

    # forecast origins t whose target t+h falls in the same period as t
    def masks(self, h):
        target_year = pd.Series((self.df.index + pd.Timedelta(days=h)).year, index=self.df.index)
        fit = self.train & (target_year <= self.TRAIN_END)
        evaluate = self.val & (target_year <= self.VAL_END)
        return fit, evaluate, self.test

    # (X, y) pairs for train/val/test: features at t, target at t+h, rows with NaN dropped
    def get(self, features, h):
        X = self.df[features]
        y = self.y.shift(-h)
        valid = X.notna().all(axis=1) & y.notna()
        return tuple((X[mask & valid], y[mask & valid]) for mask in self.masks(h))
