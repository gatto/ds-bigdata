from extract import hourly
import pandas as pd

uni = {}

for col in hourly:
	x = hourly[col].unique()
	uni[col] = (len(x))

unique_values = pd.Series(uni)

print(unique_values)