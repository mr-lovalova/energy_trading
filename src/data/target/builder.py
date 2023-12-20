import pandas as pd
from .wrangle import factory


def builder(predictor, path):
    df = pd.read_csv(path, delimiter=";", decimal=",")
    df = factory.create(predictor, df=df)
    return df
