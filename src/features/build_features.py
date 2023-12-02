"""Scripts to turn interim data into features for modeling"""
import pandas as pd
import os
import torch
import pandas as pd

path = "data/"
path = os.path.join(path, "interim/WIND_PRODUCTION.csv")
# path = os.path.join(path, "interim/SOLAR_PRODUCTION.csv")
df = pd.read_csv(path)
# df["StationID"] = df["StationID"].astype(str).str.lstrip("0")
# Convert the StationID column back to numeric type
# df["StationID"] = pd.to_numeric(df["StationID"], errors="coerce")
# WindSpeedUTC,StationID,OffshoreWindLt100MW_MWh,OffshoreWindGe100MW_MWh,OnshoreWindMWh,MunicipalityNo
X = df.pivot_table(index="HourUTC", columns="StationID", values="WindSpeedUTC")
y = df.pivot_table(
    index="HourUTC",
    columns="MunicipalityNo",
    values="WindProduction",
)
# X = df.pivot_table(index="HourUTC", columns="StationID", values="RadiationLastHourUTC")
# y = df.pivot_table(index="HourUTC", columns="MunicipalityNo", values="SolarMWh")

# remove ChristiansØ da de ikke har nogen produktion, og korrumperer dataen
y.drop(columns=[411], inplace=True)

# removing rows with missing data entries mostly relevant for X for missing measurements
X = X.dropna(axis=0)
y = y.dropna(axis=0)

common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

# Convert DataFrames to PyTorch tensors
# X_tensor = torch.Tensor(X.values, names=X.columns.to_list())
# y_tensor = torch.Tensor(y.values, names=y.columns.to_list())
# Convert DataFrames to PyTorch tensors
X = torch.Tensor(X.values)
y = torch.Tensor(y.values)

print(X)
print(y)
