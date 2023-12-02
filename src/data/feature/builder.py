from pathlib import Path
import pandas as pd
from .wrangle import factory

from .parser import get_feature_data


def builder(type_, path):
    path = Path(path)
    parsed = get_feature_data(path, type_)
    df = pd.DataFrame(parsed)
    df = factory.create(type_, df=df)
    return df
