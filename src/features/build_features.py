"""Scripts to turn interim data into features for modeling"""
import pandas as pd
import os

path = "data/"
path = os.path.join(path, "interim/dataset.csv")
df = pd.read_csv(path, delimiter=";")


print(df)
