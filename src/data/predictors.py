import pandas as pd
from config import COLUMNS


def make_df(path):
    df = pd.read_csv(path, delimiter=";")
    df = wrangle(df)
    return df


def wrangle(df):
    df[COLUMNS["Time"]] = pd.to_datetime(df[COLUMNS["Time"]])
    df[COLUMNS["WestDK"]] = df["PriceArea"].map({"DK1": True, "DK2": False})
    df.drop(["PriceArea", "SpotPriceEUR"], axis=1, inplace=True)
    return df
