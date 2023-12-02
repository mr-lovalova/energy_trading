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
        "wind_speed_past1h": "WindSpeedUTC",
    },
    "SOLAR": {
        "radia_glob_past1h": "RadiationLastHourUTC",
    },
    "MEAN": {
        "sun_last1h_glob": "SunMinutesLastHourUTC",
        "wind_speed_past1h": "WindSpeedUTC",
    },
}
