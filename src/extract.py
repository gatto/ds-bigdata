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
        return self._hour_filling(result)

    #df_tofill to fill is the full hour dataset with missing values
    def _hour_filling(self, df_tofill):
        #dataframe with only missing values rows
        mask = ~df_tofill['cnt'].notnull()
        df_missing = df_tofill.loc[mask, :].copy()

        #add a new column composed by only the date for the group by to check how many missing hours per day
        df_missing["date_full"] = df_missing["datestamp_full"].dt.date

        #list of dates in which we find more than 2 missing hours
        rows_to_delete = df_missing.groupby(["date_full"]).size()
        list_rows = list(rows_to_delete[rows_to_delete > 2].index)

        #if a date is in the previous list and it has missing cnt value we state it as True
        keep = (df_tofill["datestamp_full"].dt.date.apply(lambda x: x in list_rows)) & (~df_tofill['cnt'].notnull()) & ( (~df_tofill['cnt'].notnull()) | (~df_tofill["cnt"].shift(-1).notnull() ) )

        #we remove all true rows in the previous step
        df_tofill = df_tofill.drop(df_tofill[keep].index)

        #we fill all the missing casual registered and cnt values of the remaning missing rows as 0
        df_tofill[['casual', 'registered', 'cnt']] = df_tofill[['casual', 'registered', 'cnt']].fillna(value=0)

        #we fill all the remaining values with the previous not null row (be carefull that dtday is filled to, should we delete it?)
        df_tofill.ffill(inplace=True)
        return df_tofill
