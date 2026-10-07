"""Esegue le task 2-7 in ordine: python -m src.task_analisistazionarieta

La task 1 (analisi esplorativa) sta in src/eda.py, con le tabelle in data/eda/.
"""

from src.task_analisistazionarieta import (
    t2_stl,
    t3_mann_kendall,
    t4_stationarity,
    t5_autocorrelation,
    t6_detrending,
    t7_change_points,
)
from src.task_analisistazionarieta.common import RESULTS_DIR, setup_style

TASKS = [t2_stl, t3_mann_kendall, t4_stationarity, t5_autocorrelation, t6_detrending, t7_change_points]


def main():
    setup_style()
    for task in TASKS:
        print(f"\n=== {task.TITLE} ===")
        task.run()
    print(f"\nRisultati in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
