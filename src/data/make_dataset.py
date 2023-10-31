import os
import pandas as pd
import features
import predictors
from config import COLUMNS


def make_dataset(path="data/"):
    """Assumes your data folder is structures as given by README
    path: path to data folder"""
    x_df = features.make_df(os.path.join(path, "raw/metops"))
    y_df = predictors.make_df(os.path.join(path, "raw/prices/Elspotprices.csv"))
    merged = pd.merge(
        x_df,
        y_df,
        left_on=[COLUMNS["Time"], COLUMNS["WestDK"]],
        right_on=[COLUMNS["Time"], COLUMNS["WestDK"]],
        how="left",
    )
    merged.to_csv(os.path.join(path, "processed/dataset.csv"))


if __name__ == "__main__":
    make_dataset()
