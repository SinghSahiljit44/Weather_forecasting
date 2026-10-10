import pandas as pd

class Split:
    TARGET = "t2m_mean"
    TRAIN_END = 2017  # train: 1990-2017
    VAL_END = 2021    # validation: 2018-2021

    df: pd.DataFrame
    y: pd.Series
    train: pd.Series
    val: pd.Series
    test: pd.Series

    def __init__(self, df, target=TARGET):
        self.df = df.asfreq("D")
        self.y = self.df[target]
        year = pd.Series(self.y.index.year, index=self.y.index)
        
        self.train = year <= self.TRAIN_END
        self.val = (year > self.TRAIN_END) & (year <= self.VAL_END)
        self.test = year > self.VAL_END

    # origini il cui target t+h cade nello stesso periodo di train/val/test
    def masks(self, h):
        target_year = pd.Series((self.y.index + pd.Timedelta(days=h)).year, index=self.y.index)
        fit = self.train & (target_year <= self.TRAIN_END)
        evaluate = self.val & (target_year > self.TRAIN_END) & (target_year <= self.VAL_END)
        return fit, evaluate
