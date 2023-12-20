COLUMNS = {
    "Time": "HourUTC",
    "WestDK": "WestDK",
    "MunicipalityNo": "MunicipalityNo",
    "Coordinates": "Coordinates",
    "StationID": "StationID",
    "WIND": ["StationID", "MunicipalityNo"],
    "SOLAR": ["StationID", "MunicipalityNo"],
    "MEAN": ["Coordinates", "WestDK"],
    "SOLAR_PRODUCTION": ["SolarMWh", "MunicipalityNo"],
    "WIND_PRODUCTION": [
        "OffshoreWindLt100MW_MWh",
        "OffshoreWindGe100MW_MWh",
        "OnshoreWindMWh",
        "MunicipalityNo",
    ],
    "PRICE": ["WestDK", "SpotPriceDKK"],
}


FEATURES = {
    "WIND": {
        "wind_speed_past1h": "WindSpeed",
        # "wind_dir_past1h": "WindDir",
        # "temp_mean_past1h": "Temperature",
        "pressure": "Pressure",
        # "humidity_past1h": "Humidity",
        "radia_glob_past1h": "RadiationLastHour",
    },
    "SOLAR": {
        "radia_glob_past1h": "RadiationLastHourUTC",
        "temp_mean_past1h": "Temperature",
        "wind_speed_past1h": "WindSpeed",
        "precip_past1h": "Precip",
    },
    "MEAN": {
        "sun_last1h_glob": "SunMinutesLastHourUTC",
        "wind_speed_past1h": "WindSpeedUTC",
    },
}
