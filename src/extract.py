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
    df_raw = attr.ib()
    df_date = attr.ib()
    df_full = attr.ib()
    df = attr.ib()
    geo_k = attr.ib(default=None)
    geo_df = attr.ib()
    geo_df_SD = attr.ib()

    @geo_k.validator
    def geo_k_validator(self, attribute, value):
        rang = [None, 4, 5, 6]
        if value not in rang:
            raise ValueError(f"{attribute} must be in {rang}")

    @geo_df_SD.default
    def _geo_df_SD_default(self):
        if self.geo_df is not None:
            my_df = pd.get_dummies(data=self.geo_df, columns=["season"])
            my_df = my_df.rename(
                columns={
                    "season_1": "winter",
                    "season_2": "spring",
                    "season_3": "summer",
                    "season_4": "autumn",
                }
            )
            to_drop = ["winter"]
            my_df = my_df.drop(columns=to_drop)
            return my_df

    @geo_df.default
    def _geo_df_default(self):
        if self.geo_k is not None:
            k = self.geo_k
            if k == 4:
                pass
            elif k == 5:
                pass
            elif k == 6:
                df = self._load("geo6")
                df = pd.get_dummies(data=df, columns=["station zone"])
                df = df.rename(
                    columns={
                        "station zone_Alexandria": "z_Alexandria",
                        "station zone_Arlington": "z_Arlington",
                        "station zone_Washington NE": "z_Wa-NE",
                        "station zone_Washington NW": "z_Wa-NW",
                        "station zone_Washington SE": "z_Wa-SE",
                        "station zone_Washington SW": "z_Wa-SW",
                    }
                )
                to_drop = ["casual", "registered", "Unnamed: 0", "z_Alexandria"]
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
