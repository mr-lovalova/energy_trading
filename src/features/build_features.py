"""Scripts to turn interim data into features for modeling"""
import pandas as pd
import os
import torch
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

path = "data/"
# path = os.path.join(path, "interim/SOLAR_PRODUCTION.csv")
df = pd.read_csv(
    os.path.join(path, "interim/WIND_PRODUCTION2022.csv"),
)

# exclude_station_ids = [6096, 6093, 6041, 6073, 6049, 6051, 6132, 6088, 6051]
# df = df[~df["StationID"].isin(exclude_station_ids)]
print(df)
# Drop rows with specified StationID values
# df["StationID"] = df["StationID"].astype(str).str.lstrip("0")
# Convert the StationID column back to numeric type
# df["StationID"] = pd.to_numeric(df["StationID"], errors="coerce")
# WindSpeedUTC,StationID,OffshoreWindLt100MW_MWh,OffshoreWindGe100MW_MWh,OnshoreWindMWh,MunicipalityNo
X = df.pivot_table(index="HourUTC", columns="StationID", values="WindSpeed")  #
y = df.pivot_table(
    index="HourUTC",
    columns="MunicipalityNo",
    values="WindProduction",
)

Xtemp = df.pivot_table(
    index="HourUTC",
    columns="StationID",
    values="Pressure",
)

Xrad = df.pivot_table(
    index="HourUTC",
    columns="StationID",
    values="RadiationLastHour",
)
# X = df.pivot_table(index="HourUTC", columns="StationID", values="RadiationLastHourUTC")
# y = df.pivot_table(index="HourUTC", columns="MunicipalityNo", values="SolarMWh")

# remove ChristiansØ da de ikke har nogen produktion, og korrumperer dataen
y.drop(columns=[411], inplace=True)


# removing rows with missing data entries mostly relevant for X for missing measurements
X = X.dropna(axis=0)
y = y.dropna(axis=0)

dfs = [y, X, Xtemp, Xrad]
common_index = dfs[0].index

# Iterate through the remaining DataFrames and find the intersection of indices
for df in dfs[1:]:
    common_index = common_index.intersection(df.index)
# common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]
Xrad = Xrad.loc[common_index]
Xtemp = Xtemp.loc[common_index]

Xtemp = Xtemp.dropna(axis=1)

Xtime = pd.DataFrame()
Xtime.index = pd.to_datetime(X.index)
Xtime["HOUR"] = Xtime.index.hour
Xtime["Month"] = Xtime.index.month
Xtime["day"] = Xtime.index.day

# X_time = pd.DataFrame(pd.to_datetime(X.index).hour, pd.to_datetime(X.index).month, pd.to_datetime(X.index).year)

print(X)
print(Xrad)
print(Xtemp)
print(Xtime)
print(y)
X = torch.Tensor(X.values)
Xrad = torch.Tensor(Xrad.values)
Xtemp = torch.Tensor(Xtemp.values)
Xtime = torch.Tensor(Xtime.values)
y = torch.Tensor(y.values)

# time_scaler = StandardScaler()
# X_time = torch.Tensor(X_time.values)


# print(X)
# print(y)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print(len(X_scaled[0]))
print(y)
torch.save(X_scaled, os.path.join(path, "processed/production/2022/windspeed.pt"))
torch.save(y, os.path.join(path, "processed/production/2022/production.pt"))


scaler = StandardScaler()
scaler.fit(Xrad)
X_scaled = scaler.transform(Xrad)

torch.save(X_scaled, os.path.join(path, "processed/production/2022/radiation.pt"))


scaler = StandardScaler()
scaler.fit(Xtemp)
X_scaled = scaler.transform(Xtemp)

torch.save(X_scaled, os.path.join(path, "processed/production/2022/pressure.pt"))

scaler = StandardScaler()
scaler.fit(Xtime)
X_scaled = scaler.transform(Xtime)

torch.save(X_scaled, os.path.join(path, "processed/production/2022/hours.pt"))
