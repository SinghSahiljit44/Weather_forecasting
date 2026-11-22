import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_accuracy(y, y_pred, tolerance=2.0):
    return np.mean(np.abs(np.asarray(y) - np.asarray(y_pred)) <= tolerance)

def get_mean_absolute_error(y, y_pred):
    return mean_absolute_error(y, y_pred)

def get_rmse(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5

def get_r2_score(y, y_pred):
    return r2_score(y, y_pred)

def plot_rmse(y_true, y_pred, title, unit):
    pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    rmse = pd.Series({h: get_rmse(y_true[h], pred[h]) for h in y_true.columns})

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(rmse.index, rmse.values, color="#2a78d6", linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel(f"RMSE ({unit})")
    ax.set_title(title, loc="left")
    ax.set_xticks(rmse.index[::2])
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.show()

def metrics_table(splits, tolerance=None, leads=None):
    scores = {"MAE": get_mean_absolute_error, "RMSE": get_rmse, "R²": get_r2_score}
    if tolerance is not None:
        scores[f"Acc ±{tolerance}"] = lambda y, p: get_accuracy(y, p, tolerance)

    preds = {name: (y, pd.DataFrame(p, index=y.index, columns=y.columns))
             for name, (y, p) in splits.items()}

    table = pd.DataFrame({
        (metric, name): {h: fn(y[h], p[h]) for h in y.columns}
        for metric, fn in scores.items()
        for name, (y, p) in preds.items()
    })
    table.index.name = "Lead time (days)"
    return table if leads is None else table.loc[leads]

def plot_training_history(history, metric="mae", test_scores=None, title=None):
    hist = getattr(history, "history", history)
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", metric]):
        ax.plot(epochs, hist[key], label="train")
        if f"val_{key}" in hist:
            ax.plot(epochs, hist[f"val_{key}"], label="validation")
        if test_scores is not None and key in test_scores:
            ax.axhline(test_scores[key], color="gray", linestyle="--", label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(key)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig

def keras_accuracy(tolerance=2.0):
    import tensorflow as tf

    def tol_accuracy(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= tolerance, tf.float32))

    return tol_accuracy