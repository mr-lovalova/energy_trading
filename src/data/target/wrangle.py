import pandas as pd
from config import COLUMNS
from helpers import ObjectFactory


def set_time(df):
    df[COLUMNS["Time"]] = pd.to_datetime(df[COLUMNS["Time"]]).dt.tz_localize(None)
    df.set_index(COLUMNS["Time"], inplace=True)
    return df


def price_wrangler(df):
    df = set_time(df)
    print(df)
    df[COLUMNS["WestDK"]] = df["PriceArea"].map({"DK1": True, "DK2": False})
    df.reset_index(inplace=True)
    df.set_index("HourUTC", inplace=True)
    df = df.dropna()
    print(df)
    return df


def wind_production_wrangler(df):
    df = set_time(df)
    df = df[COLUMNS["WIND_PRODUCTION"]]
    df["WindProduction"] = df[
        ["OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh", "OnshoreWindMWh"]
    ].sum(axis=1)
    df = df.drop(
        columns=[
            "OffshoreWindLt100MW_MWh",
            "OffshoreWindGe100MW_MWh",
            "OnshoreWindMWh",
        ],
        axis=1,
    )

    # df = df.groupby(["MunicipalityNo"], group_keys=True).apply(lambda x: x)
    return df


def solar_production_wrangler(df):
    df = set_time(df)
    df = df[COLUMNS["SOLAR_PRODUCTION"]]
    # df = df.groupby(["MunicipalityNo"], group_keys=True).apply(lambda x: x)
    return df


factory = ObjectFactory()
factory.register_builder("PRICE", price_wrangler)
factory.register_builder("SOLAR_PRODUCTION", solar_production_wrangler)
factory.register_builder("WIND_PRODUCTION", wind_production_wrangler)
