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

X = X.dropna(axis=0, how="all")
y = y.dropna(axis=1, how="all")
# Convert DataFrames to PyTorch tensors
X_tensor = torch.Tensor(X.values)
y_tensor = torch.Tensor(y.values)

print(X)
print(y)

print(df)

X.to_excel("X.xlsx")
y.to_excel("Y.xlsx")
