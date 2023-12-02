import os
import feature
import target
import pandas as pd
from config import COLUMNS


def make_dataset(path="data/", feat="WIND", pred="WIND_PRODUCTION"):
    """Assumes your data folder is structures as given by README
    path: path to data folder"""
    x_df = feature.builder(feat, os.path.join(path, "raw/metops"))
    # y_df = target.builder(pred, os.path.join(path, "raw/prices/Elspotprices.csv"))
    y_df = target.builder(
        pred, os.path.join(path, "raw/production/ProductionMunicipalityHour.csv")
    )
    print(x_df)

    merged = pd.merge(x_df, y_df, how="inner", left_index=True, right_index=True)
    merged = merged.dropna()
    print(merged)
    merged.to_csv(os.path.join(path, f"interim/{pred}.csv"))
    # merged.to_excel(os.path.join(path, f"interim/{pred}.xlsx"))


def merge(x_df, y_df):
    merged = pd.merge(
        x_df,
        y_df,
        left_on=[COLUMNS["Time"]],
        right_on=[COLUMNS["Time"]],
        how="left",
    )
    return merged


if __name__ == "__main__":
    make_dataset()
