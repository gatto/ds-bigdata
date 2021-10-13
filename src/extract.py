import pkg_resources
from pathlib import Path

import attr
import pandas as pd

try:
    DATA_PATH = pkg_resources.resource_filename("extractbda", "data/")
except ModuleNotFoundError:
    DATA_PATH = "data/"


@attr.s
class Bikes:
    df = attr.ib()
    df_date = attr.ib()
    df_full = attr.ib()

    @df.default
    def _df_default(self):
        # loading
        my_df = self._load("hour")

        # dropping
        cols_to_drop = ["instant"]
        my_df = my_df.drop(cols_to_drop, axis=1)
        return my_df

    @df_date.default
    def _df_date_default(self):
        my_df = self.df
        my_df["dteday"] = pd.to_datetime(my_df["dteday"])
        my_df["datestamp"] = list(
            map(lambda x, y: x.replace(hour=y), my_df["dteday"], my_df["hr"])
        )
        return my_df

    @df_full.default
    def _df_full_default(self):
        return self._fill_timestamps(self.df_date)

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
