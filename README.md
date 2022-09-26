# ds-bigdata - package `extractbda`
An interface for easy access to data and final ML-predictor for our Big Data Analytics project. By installing and importing this package, you can very easily create an object that holds the data and our best predictor (a RandomForestRegressor). You can then run your own analysis and compare your results to ours.


## Usage
```python
!pip install -U git+https://github.com/gatto/ds-bigdata.git
from extractbda import Bikes
bik = Bikes(geo_k=21)
```



### Parameters for `Bikes()`
#### `geo_k = 21|11|6` (default 11)
How many zones to divide the dataset in. Although the default is 11, we did most of our analysis using **21**.

#### `val = True|False` (default False)
If to provide a holdout **validation** set besides the standard training and test sets. Use `False` if cross-validating.

### Objects found in `bik`
#### predictor-related
- `bik.model["RF"]` (scikit-learn.RandomForestRegressor object)
- `bik.model["y_pred"]` (prediction over x_test)
- `bik.model["r2"]` (r-squared score for RF)
- `bik.model["mse"]` (mse metric for RF)

#### train/test datasets
- `bik.d[“x_train”]`
- `bik.d[“x_test”]`
- `bik.d[“x_val”]` (only if creating `Bikes(val=True)`)
- `bik.d[“y_train”]`
- `bik.d[“y_test”]`
- `bik.d[“y_val”]` (only if creating `Bikes(val=True)`)

#### whole datasets
- `bik.geo_df_SD` (dataset with zones, seasons and weathersit dummies. Used in the train/test datasets above)
- `bik.geo_df` (dataset with no dummies)

## Notes on the model
We choose as target `cnt`: the total count of how many bikes were taken out over a granularity of one day and one zone. We have different aggregations of zones: either 6, 11 or 21 zones. The model was trained on 21 zones.

<p align="center">
    <img src="https://user-images.githubusercontent.com/63819344/192303823-239b22c4-57f4-45c3-b2f2-e4bf03b61b43.png" width="50%"/>
    <br>
    <em>Fig - 11 zones partitioning</em>
</p>
<p>
<p align="center">
    <img  align="center" src="https://user-images.githubusercontent.com/63819344/192303868-a578a26a-251c-40c8-bc35-61766adfd21a.png" width="50%"/>
    <br>
    <em>Fig - 21 zones partitioning</em>
</p>



No trend features were inserted and the data was not treated as time series because we don't think there are causality links between the `cnt` of one day and the `cnt` of the next or previous day.

## Notes on some attributes
- dteday: date
- season: season (1:winter, 2:spring, 3:summer, 4:fall)
- yr: year (0: 2011, 1:2012)
- mnth: month (1 to 12)
- hr: hour (0 to 23)
- holiday: day is holiday or not
- weekday: day of the week
- workingday: if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit:
    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - **(DELETED)** 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- temp: Normalized temperature in Celsius. The values are derived via (t-tmin)/(tmax-tmin), tmin=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-tmin)/(tmax-tmin), tmin=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered
