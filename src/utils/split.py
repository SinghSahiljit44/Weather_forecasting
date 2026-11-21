import pandas as pd

class Split:
    TRAIN_END = 2017  # train: 1990-2017
    VAL_END = 2021    # validation: 2018-2021, test: 2022+
    HORIZON = 30

    def __init__(self, df, horizon=HORIZON):
        # daily frequency, so shift(-h) is always h days ahead (missing days become NaN rows)
        self.df = df.sort_index().asfreq("D")
        self.horizon = horizon
        self.horizons = list(range(1, horizon + 1))

        year = pd.Series(self.df.index.year, index=self.df.index)
        self.train = year <= self.TRAIN_END
        self.val = (year > self.TRAIN_END) & (year <= self.VAL_END)
        self.test = year > self.VAL_END

    # Y[h] = target at t+h, for h = 1..horizon
    def define_target(self, target):
        self.y = self.df[target]
        self.Y = pd.DataFrame({h: self.y.shift(-h) for h in self.horizons})

    # forecast origins t whose whole window t+1..t+horizon falls in the same period as t
    def masks(self):
        last_year = pd.Series((self.df.index + pd.Timedelta(days=self.horizon)).year, index=self.df.index)
        fit = self.train & (last_year <= self.TRAIN_END)
        evaluate = self.val & (last_year <= self.VAL_END)
        return fit, evaluate, self.test

    # (X, Y) for train/val/test: features at t, one target column per lead time.
    # A row is kept only if all horizons exist, so every h is scored on the same days.
    def get(self, features, h=None):
        X = self.df[features]
        valid = X.notna().all(axis=1) & self.Y.notna().all(axis=1)
        Y = self.Y if h is None else self.Y[h]
        return tuple((X[m & valid], Y[m & valid]) for m in self.masks())