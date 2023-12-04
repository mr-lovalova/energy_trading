"""Scripts to turn interim data into features for modeling"""
import pandas as pd
import os
import torch
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

path = "data/"
# path = os.path.join(path, "interim/SOLAR_PRODUCTION.csv")
df = pd.read_csv(os.path.join(path, "interim/WIND_PRODUCTIONbig.csv"))

# exclude_station_ids = [6096, 6093, 6041, 6073, 6049, 6051, 6132, 6088, 6051]
# df = df[~df["StationID"].isin(exclude_station_ids)]

# Drop rows with specified StationID values
# df["StationID"] = df["StationID"].astype(str).str.lstrip("0")
# Convert the StationID column back to numeric type
# df["StationID"] = pd.to_numeric(df["StationID"], errors="coerce")
# WindSpeedUTC,StationID,OffshoreWindLt100MW_MWh,OffshoreWindGe100MW_MWh,OnshoreWindMWh,MunicipalityNo
X = df.pivot_table(
    index="HourUTC",
    columns="StationID",
    values=["WindSpeedUTC", "WindDir", "Temperature", "Humidity", "Precip"],
)
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
# X["Year"] = pd.to_datetime(X.index).hour

# X_time = pd.DataFrame(pd.to_datetime(X.index).hour, pd.to_datetime(X.index).month, pd.to_datetime(X.index).year)

print(X)
# X.to_excel("XXX.xlsx")
X = torch.Tensor(X.values)
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
torch.save(X_scaled, os.path.join(path, "processed/production/X_wind.pt"))
torch.save(y, os.path.join(path, "processed/production/y_wind.pt"))
