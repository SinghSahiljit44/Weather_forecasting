from src.data.dataset import load_dataset, select_cell, to_hourly_frame, save_hourly
from src.data.preprocessing import hourly_to_daily, save_daily

ds = load_dataset()
cell = select_cell(ds)
hourly = to_hourly_frame(cell)
save_hourly(hourly)

daily = hourly_to_daily(hourly)
save_daily(daily)

