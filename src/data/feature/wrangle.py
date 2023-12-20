import pandas as pd
from config import COLUMNS
from helpers import ObjectFactory


def set_time(df):
    df[COLUMNS["Time"]] = pd.to_datetime(df[COLUMNS["Time"]]).dt.tz_localize(None)
    df[COLUMNS["Time"]] = df[COLUMNS["Time"]] - pd.Timedelta(hours=1)
    df.set_index(COLUMNS["Time"], inplace=True)
    return df


def solar_wrangler(df, **ignored):
    df = set_time(df)
    df = df.groupby("StationID").apply(lambda group: group.ffill().bfill())
    df = df.reset_index(level="StationID", drop=True)
    df.drop_duplicates()
    # df = df.groupby(["StationID"], group_keys=True).apply(lambda x: x)
    return df


def wind_wrangler(df, **ignored):
    df = set_time(df)
    print(df)
    df = df[df.index.minute == 0]

    # Reset index after filtering
    # df.reset_index(drop=True, inplace=True)
    # df = df.groupby(["StationID"], group_keys=True).apply(lambda x: x)
    df = df.groupby("StationID").apply(lambda group: group.ffill().bfill())
    df = df.reset_index(level="StationID", drop=True)
    df.drop_duplicates()
    return df


def mean_weather_wrangler(df, **ignored):
    df = set_time(df)
    df = df.groupby([COLUMNS["WestDK"], pd.Grouper(freq="H")]).mean()
    # df.reset_index(inplace=True)
    # df.set_index("HourUTC", inplace=True)
    return df


factory = ObjectFactory()
factory.register_builder("SOLAR", solar_wrangler)
factory.register_builder("WIND", wind_wrangler)
factory.register_builder("MEAN", mean_weather_wrangler)
