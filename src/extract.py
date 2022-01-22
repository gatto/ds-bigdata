from pathlib import Path

import attr
import pandas as pd
import pkg_resources
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    DATA_PATH = pkg_resources.resource_filename("extractbda", "data/")
except ModuleNotFoundError:
    DATA_PATH = "data/"


@attr.s
class Bikes:
    df_raw = attr.ib()
    df_date = attr.ib()
    df_full = attr.ib()
    df = attr.ib()
    geo_k = attr.ib(default=11)
    geo_df = attr.ib()
    geo_df_SD = attr.ib()
    val = attr.ib(default=False)
    d = attr.ib()
    model = attr.ib()

    def preprocessing(self, url: str) -> pd.DataFrame:
        df = pd.read_csv(url)
        my_len = len(df)

        to_drop = ["casual", "registered"]
        df = df.drop(columns=to_drop)
        df["dteday"] = pd.to_datetime(df["dteday"])

        df = df.append(self.geo_df, ignore_index=True)
        df = self._preproc(df)
        df = df.iloc[0:my_len]

        to_drop = ["winter", "w_sunny", "z_Zone 1", "dteday", "season"]
        df = df.drop(columns=to_drop)

        return df.drop(columns="cnt"), df["cnt"]

    def _preproc(self, df: pd.DataFrame) -> pd.DataFrame:
        seasons = df["season"]
        my_df = pd.get_dummies(data=df, columns=["season"])
        my_df = my_df.rename(
            columns={
                "season_1": "winter",
                "season_2": "spring",
                "season_3": "summer",
                "season_4": "autumn",
            }
        )
        # add zone dummies
        my_df = my_df.rename(columns={"station zone": "z"})
        my_df = pd.get_dummies(
            data=my_df,
            columns=["z"],
        )
        my_df = pd.get_dummies(data=my_df, columns=["weathersit"])
        my_df = my_df.rename(
            columns={
                "weathersit_1": "w_sunny",
                "weathersit_2": "w_cloudy",
                "weathersit_3": "w_rain",
            }
        )
        # re-add season to allow for stratification split
        my_df["season"] = seasons
        return my_df

    @model.default
    def _model_default(self):
        to_drop = "dteday"
        x_train = self.d["x_train"].drop(columns=to_drop)
        x_test = self.d["x_test"].drop(columns=to_drop)
        if self.geo_k == 11:
            RF_reg = RandomForestRegressor(
                criterion="squared_error",
                n_estimators=250,
                max_depth=20,
                max_features="auto",
                min_samples_leaf=10,
                min_samples_split=30,
                random_state=0,
            )
        elif self.geo_k == 21:
            RF_reg = RandomForestRegressor(
                criterion="squared_error",
                n_estimators=350,
                max_depth=30,
                max_features="auto",
                min_samples_leaf=10,
                min_samples_split=30,
                random_state=0,
            )
        else:
            return {"RF": None, "y_pred": None, "r2": None, "mse": None}
        RF_reg.fit(x_train, self.d["y_train"])
        # y_pred_train = RF_reg.predict(self.d["x_train"])
        y_pred = RF_reg.predict(x_test)
        r2 = r2_score(self.d["y_test"].astype(float), y_pred)
        mse = mean_squared_error(self.d["y_test"].astype(float), y_pred)
        print(f"Fitted a RFRegressor with R^2 {r2:.3f} and MSE {mse:.1f}.")
        return {"RF": RF_reg, "y_pred": y_pred, "r2": r2, "mse": mse}

    @d.default
    def _d_default(self):
        my_d = {}
        df = self.geo_df_SD
        attributes = [c for c in df.columns if c != "cnt"]
        x = df[attributes]
        y = df["cnt"]
        to_drop = "season"

        # split in train e test
        (
            my_d["x_train"],
            my_d["x_test"],
            my_d["y_train"],
            my_d["y_test"],
        ) = train_test_split(
            x, y, test_size=0.15, random_state=420, stratify=x["season"]
        )
        if self.val:
            # split del train in valid
            (
                my_d["x_train"],
                my_d["x_val"],
                my_d["y_train"],
                my_d["y_val"],
            ) = train_test_split(
                my_d["x_train"],
                my_d["y_train"],
                test_size=0.20,
                random_state=420,
                stratify=my_d["x_train"]["season"],
            )
            my_d["y_val"] = my_d["y_val"].sort_index()
            my_d["x_val"] = my_d["x_val"].drop(columns=to_drop).sort_index()
        my_d["x_train"] = my_d["x_train"].drop(columns=to_drop).sort_index()
        my_d["x_test"] = my_d["x_test"].drop(columns=to_drop).sort_index()
        my_d["y_train"] = my_d["y_train"].sort_index()
        my_d["y_test"] = my_d["y_test"].sort_index()
        return my_d

    @geo_k.validator
    def geo_k_validator(self, attribute, value):
        rang = [6, 11, 21]
        if value not in rang:
            raise ValueError(f"{attribute} must be in {rang}")

    @geo_df_SD.default
    def _geo_df_SD_default(self):
        # add season dummies
        my_df = self._preproc(self.geo_df)

        if self.geo_k == 11:
            to_drop = ["winter", "z_Alexandria", "w_sunny"]
        elif self.geo_k == 21:
            to_drop = ["winter", "w_sunny", "z_Zone 1"]
        my_df = my_df.drop(columns=to_drop)
        return my_df

    @geo_df.default
    def _geo_df_default(self):
        k = self.geo_k
        if k == 11:
            df = self._load("geo11")
            to_drop = ["casual", "registered"]
            df = df.drop(columns=to_drop)
            df["dteday"] = pd.to_datetime(df["dteday"])
        elif k == 6:
            df = self._load("geo6")
            to_drop = ["casual", "registered", "Unnamed: 0"]
            df = df.drop(columns=to_drop)
            df["dteday"] = pd.to_datetime(df["dteday"])
        elif k == 21:
            df = self._load("geo21")
            to_drop = ["casual", "registered"]
            df = df.drop(columns=to_drop)
            df["dteday"] = pd.to_datetime(df["dteday"])
        return df

    @df_raw.default
    def _df_raw_default(self):
        # loading
        my_df = self._load("hour")

        # dropping
        cols_to_drop = ["instant"]
        my_df = my_df.drop(cols_to_drop, axis=1)
        return my_df

    @df_date.default
    def _df_date_default(self):
        my_df = self.df_raw
        my_df["dteday"] = pd.to_datetime(my_df["dteday"])
        my_df["datestamp"] = list(
            map(lambda x, y: x.replace(hour=y), my_df["dteday"], my_df["hr"])
        )
        return my_df

    @df_full.default
    def _df_full_default(self):
        return self._fill_timestamps(self.df_date)

    @df.default
    def _df_default(self):
        return self._hour_filling(self.df_full)

    def _load(self, file):
        my_csv = Path(f"{DATA_PATH}{file}.csv")
        my_csv_pkl = Path(f"{DATA_PATH}{file}.pkl")
        try:
            pipi = pd.read_pickle(my_csv_pkl)
        except FileNotFoundError:
            pipi = pd.read_csv(my_csv)
        return pipi

    def _fill_timestamps(self, df):
        """
        input dataframe must have the datestamp
        create pd.Series with all timestamps from 2011-01-01 00:00:00 to 2012-12-31 23:00:00
        """
        idx = pd.Series(
            pd.date_range(
                "2011-01-01 00:00:00",
                "2012-12-31 23:00:00",
                freq="H",
            ),
            name="datestamp_full",
        )
        # join new pd.Series with our dataset and delete old datestamp
        result = pd.merge(
            df, idx, left_on="datestamp", right_on="datestamp_full", how="right"
        ).drop("datestamp", axis=1)
        return result

    def _hour_filling(self, df_tofill):
        # df_missing is a dataframe with only missing values
        df_missing = df_tofill[~df_tofill["cnt"].notnull()].copy()
        date = df_missing["datestamp_full"]
        hour = pd.Timedelta("1h")
        in_block = ((date - date.shift(-1)).abs() == hour) | (date.diff() == hour)
        filt = df_missing.loc[in_block]
        breaks = filt["datestamp_full"].diff() != hour
        # count consecutive missing hours
        groups = breaks.cumsum()
        df_missing["groups"] = groups
        groups_count = df_missing.groupby(["groups"], as_index=False).size()
        df_missing_groups = pd.merge(df_missing, groups_count, on=["groups", "groups"])
        # filter blocks of hours with more than 3 consecutive missing hours
        dates_to_discard = df_missing_groups[df_missing_groups["size"] > 3]
        list_dates_to_discard = list(dates_to_discard["datestamp_full"])
        # delete list of flagged hours
        df_tofill = df_tofill.loc[
            (~df_tofill["datestamp_full"].isin(list_dates_to_discard)), :
        ].copy()

        df_tofill.loc[:, ("casual", "registered", "cnt")].fillna(value=0, inplace=True)
        df_tofill.ffill(inplace=True)
        return df_tofill
