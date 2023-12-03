"""Scripts to turn interim data into features for modeling"""
import pandas as pd
import os
import torch
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

path = "data/"
# path = os.path.join(path, "interim/SOLAR_PRODUCTION.csv")
df = pd.read_csv(os.path.join(path, "interim/WIND_PRODUCTION.csv"))
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

X = torch.Tensor(X.values)
y = torch.Tensor(y.values)
X.to(dtype=torch.float32)

print(X)
print(y)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print(X_scaled)
print(y)
# torch.save(X_scaled, os.path.join(path, "processed/production/X_wind.pt"))
# torch.save(y, os.path.join(path, "processed/production/y_wind.pt"))
