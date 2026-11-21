import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def print_metrics_value(what_am_i_printing, y_val, y_val_pred):
    print("\n" + what_am_i_printing)
    y_val_pred = pd.DataFrame(y_val_pred, index=y_val.index, columns=y_val.columns)
    for h in y_val.columns:
        print(f"\n{h} giorni previsti in avanti:")
        print("MAE:", mean_absolute_error(y_val[h], y_val_pred[h]))
        print("RMSE:", mean_squared_error(y_val[h], y_val_pred[h]) ** 0.5)
        print("R²:", r2_score(y_val[h], y_val_pred[h]))


def get_mean_absolte_error(y_val, y_val_pred):
    return mean_absolute_error(y_val, y_val_pred)

def get_rmse(y_val, y_val_pred):
    return mean_squared_error(y_val, y_val_pred) ** 0.5

def get_r2_score(y_val, y_val_pred):
    return r2_score(y_val, y_val_pred)