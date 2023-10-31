import json
import os
from pathlib import Path
import pandas as pd
from config import FEATURES, COLUMNS


def is_valid(measurement):
    if measurement["properties"]["parameterId"] not in FEATURES:
        return False
    coordinates = measurement["geometry"]["coordinates"]
    if not coordinates:
        return False
    _, lat = coordinates
    if is_greenland(lat):
        return False
    return True


def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_file(path, file, parsed):
    with open(os.path.join(path, file), "r") as f:
        for line in f:
            measurement = json.loads(line)
            if not is_valid(measurement):
                continue
            parsed.append(parse_measurement(measurement))


def parse_measurement(measurement):
    feature = measurement["properties"]["parameterId"]
    lng, _ = measurement["geometry"]["coordinates"]
    time = measurement["properties"]["observed"]
    value = measurement["properties"]["value"]
    dct = {
        COLUMNS["Time"]: time,
        FEATURES[feature]: float(value),
        COLUMNS["WestDK"]: is_west(float(lng)),
    }
    return dct


def is_west(lng):
    # naive guess, take into account earth is round?
    guess = 10.9
    return lng < guess


def is_greenland(lat):
    guess = 58
    return lat > guess


def wrangle(df):
    df[COLUMNS["Time"]] = pd.to_datetime(df[COLUMNS["Time"]]).dt.tz_localize(None)
    df.set_index(COLUMNS["Time"], inplace=True)
    hourly_avg = df.groupby([COLUMNS["WestDK"], pd.Grouper(freq="H")]).mean()
    return hourly_avg


def get_feature_data(path):
    files = get_files(path)
    parsed = []
    for file in files:
        parse_file(path, file, parsed)
        print(f"EXTRACTING DATA FROM FILE: {file}")
    return parsed


def make_df(path):
    path = Path(path)
    parsed = get_feature_data(path)
    df = pd.DataFrame(parsed)
    df = wrangle(df)
    return df
