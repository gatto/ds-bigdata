import pandas as pd
from rich import print

cols_to_drop = ["instant"]
hourly = pd.read_csv("data/hour.csv")
hourly = hourly.drop(cols_to_drop, axis=1)

print(hourly.info())
print(hourly.head())

