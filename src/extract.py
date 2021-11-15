from pathlib import Path

import attr
import pandas as pd
import pkg_resources
from sklearn.model_selection import train_test_split

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
    val = attr.ib(default=True)
    d = attr.ib()

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
        ) = train_test_split(x, y, test_size=0.15, random_state=420, stratify="season")
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
            print(my_d["x_train"].head())
            my_d["x_val"] = my_d["x_val"].drop(columns=to_drop)
        my_d["x_train"] = my_d["x_train"].drop(columns=to_drop)
        my_d["x_test"] = my_d["x_test"].drop(columns=to_drop)
        print(my_d["x_train"].head())
        return my_d

    @geo_k.validator
    def geo_k_validator(self, attribute, value):
        rang = [6, 11]
        if value not in rang:
            raise ValueError(f"{attribute} must be in {rang}")

    @geo_df_SD.default
    def _geo_df_SD_default(self):
        # add season dummies
        seasons = self.geo_df["season"]
        my_df = pd.get_dummies(data=self.geo_df, columns=["season"])
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
        # drop unneeded
        to_drop = ["winter", "z_Alexandria"]
        my_df = my_df.drop(columns=to_drop)
        # re-add season to allow for stratification split
        my_df["season"] = seasons
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
