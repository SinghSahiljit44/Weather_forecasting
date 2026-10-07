"""Task 6 - Rimozione di trend e stagionalità, e nuovi test di stazionarietà."""

import matplotlib.pyplot as plt
import pandas as pd

from src.task_analisistazionarieta import t2_stl
from src.task_analisistazionarieta.common import BLUE, SERIES, UNITS, save_figure, save_table, setup_style, stl_component
from src.task_analisistazionarieta.t4_stationarity import report, test_series

TITLE = "Task 6 - Rimozione di trend e stagionalità"


def transforms(stl, variable):
    observed = stl_component(stl, variable, "observed")
    return {
        "originale": observed,
        "senza trend": observed - stl_component(stl, variable, "trend"),
        "destagionalizzata": observed - stl_component(stl, variable, "seasonal"),
        "residuo": stl_component(stl, variable, "resid"),
    }


def plot(stl, variable):
    series = transforms(stl, variable)
    fig, axes = plt.subplots(len(series), 1, figsize=(11, 7.5), sharex=True)
    for ax, (label, values) in zip(axes, series.items()):
        ax.plot(values.index, values, color=BLUE, linewidth=0.35)
        ax.set_title(label.capitalize(), loc="left")
        ax.set_ylabel(UNITS[variable])
    fig.suptitle(f"{SERIES[variable]} - serie trasformate", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"t6_trasformazioni_{variable}")


def run(verbose=True):
    stl = t2_stl.get_components()
    rows = [
        test_series(values, variable, label)
        for variable in SERIES
        for label, values in transforms(stl, variable).items()
    ]
    table = pd.DataFrame(rows)
    save_table(table, "stationarity_transformed")
    for variable in SERIES:
        plot(stl, variable)

    if verbose:
        print(report(table))
    return table


if __name__ == "__main__":
    setup_style()
    run()
