import json
import os
from config import COLUMNS


def is_west(lng):
    # naive guess, take into account earth is round?
    guess = 10.9
    return lng < guess


def is_valid(measurement):
    coordinates = measurement["geometry"]["coordinates"]
    if not coordinates:
        return False
    _, lat = coordinates
    if is_greenland(lat):
        return False
    return True


def is_greenland(lat):
    guess = 58
    return lat > guess


def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def is_feature(measurement, features):
    if measurement["properties"]["parameterId"] not in features:
        return False
    return True


def parse_file(path, file, features, parsed):
    with open(os.path.join(path, file), "r") as f:
        for line in f:
            measurement = json.loads(line)
            if not is_feature(measurement, features):
                continue
            if not is_valid(measurement):
                continue
            parsed.append(parse_measurement(measurement, features))


def parse_measurement(measurement, features):
    feature = measurement["properties"]["parameterId"]
    time = measurement["properties"]["observed"]
    value = measurement["properties"]["value"]
    lng, _ = measurement["geometry"]["coordinates"]
    out = {
        COLUMNS["Time"]: time,
        features[feature]: float(value),
        COLUMNS["StationID"]: measurement["properties"]["stationId"],
        COLUMNS["WestDK"]: is_west(float(lng)),
    }
    return out


def get_feature_data(path, features):
    files = get_files(path)
    parsed = []
    for file in files:
        parse_file(path, file, features, parsed)
        print(f"EXTRACTING DATA FROM FILE: {file}")
    return parsed
