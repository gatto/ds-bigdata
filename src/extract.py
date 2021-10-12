import pkg_resources
from pathlib import Path

import attr
import pandas as pd

DATA_PATH = pkg_resources.resource_filename("extractbda", "data/")


@attr.s
class Bikes:
    df = attr.ib()
    df_ds = attr.ib()

    @df.default
    def _df_default(self):
        # loading
        my_df = self._load("hour")

        # dropping
        cols_to_drop = ["instant"]
        my_df = my_df.drop(cols_to_drop, axis=1)
        return my_df

    @df_ds.default
    def _df_ds_default(self):
        # options
        my_df = self.df
        # my_df["datestamp"] = list(map(lambda x, y: x.replace(hour=y), my_df["dteday"], my_df["hr"]))
        return my_df

    def _load(self, file):
        my_csv = Path(f"{DATA_PATH}{file}.csv")
        my_csv_pkl = Path(f"{DATA_PATH}{file}.pkl")
        try:
            pipi = pd.read_pickle(my_csv_pkl)
        except FileNotFoundError:
            pipi = pd.read_csv(my_csv)
        return pipi
