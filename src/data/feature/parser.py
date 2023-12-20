import os
import json
from abc import ABC, abstractmethod
from config import COLUMNS, FEATURES
from helpers import ObjectFactory


class MetopsParser(ABC):
    def __init__(self, path, **ignored):
        self.parsed = []
        self.path = path

    @property
    @abstractmethod
    def features(self):
        pass

    @staticmethod
    def is_greenland(lat):
        guess = 58
        return lat > guess

    @staticmethod
    def is_valid(measurement):
        try:
            coordinates = measurement["geometry"]["coordinates"]
            _, lat = coordinates
        except TypeError:
            return False  # or known station??
        if MetopsParser.is_greenland(lat):
            return False
        return True

    def parse_file(self, file):
        with open(os.path.join(self.path, file), "r") as f:
            for line in f:
                measurement = json.loads(line)
                if not self.is_feature(measurement):
                    continue
                if not self.is_valid(measurement):
                    continue
                self.parsed.append(self.parse_measurement(measurement))

    def is_feature(self, measurement):
        if measurement["properties"]["parameterId"] not in self.features:
            return False
        return True

    def parse_measurement(self, measurement):
        feature = measurement["properties"]["parameterId"]
        time = measurement["properties"]["observed"]
        value = measurement["properties"]["value"]
        out = {
            COLUMNS["Time"]: time,
            self.features[feature]: float(value),
            # COLUMNS["StationID"]: measurement["properties"]["stationId"],
        }
        return out


def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def get_feature_data(path, type_):
    files = get_files(path)
    parser = factory.create(type_, path=path)
    for file in files:
        parser.parse_file(file)
        print(f"EXTRACTING DATA FROM FILE: {file}")
    return parser.parsed


class SolarWindParser(MetopsParser):
    def parse_measurement(self, measurement):
        out = super().parse_measurement(measurement)
        out[COLUMNS["StationID"]] = measurement["properties"]["stationId"]
        return out


class SolarParser(SolarWindParser):
    @property
    def features(self):
        return FEATURES["SOLAR"]


class WindParser(SolarWindParser):
    @property
    def features(self):
        return FEATURES["WIND"]


class MeanWeatherParser(MetopsParser):
    def parse_measurement(self, measurement):
        out = super().parse_measurement(measurement)
        lng, _ = measurement["geometry"]["coordinates"]
        out[COLUMNS["WestDK"]] = self.is_west(float(lng))
        return out

    @staticmethod
    def is_west(lng):
        guess = 10.9
        return lng < guess

    @property
    def features(self):
        return FEATURES["MEAN"]


factory = ObjectFactory()
factory.register_builder("SOLAR", SolarParser)
factory.register_builder("WIND", WindParser)
factory.register_builder("MEAN", MeanWeatherParser)
