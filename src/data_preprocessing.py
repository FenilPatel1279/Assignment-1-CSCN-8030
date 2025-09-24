import numpy as np
import pandas as pd
import os


def generate_synthetic_data(n_samples=300, random_state=42):
    """
    Generate synthetic wildfire dataset.
    Features: Temperature, Humidity, SmokeLevel, SatelliteHeat
    Target: Wildfire (0 = No, 1 = Yes)
    """
    np.random.seed(random_state)

    temperature = np.random.normal(32, 5, n_samples)       # °C
    humidity = np.random.normal(40, 10, n_samples)         # %
    smoke_level = np.random.normal(50, 15, n_samples)      # sensor intensity
    satellite_heat = np.random.normal(70, 20, n_samples)   # heat index

    # Fire risk depends on high temp, low humidity, high smoke/heat
    fire_risk = (temperature > 34) & (humidity < 35) & (smoke_level > 55) & (satellite_heat > 80)
    fire_risk = fire_risk.astype(int)

    data = pd.DataFrame({
        "Temperature": temperature,
        "Humidity": humidity,
        "SmokeLevel": smoke_level,
        "SatelliteHeat": satellite_heat,
        "Wildfire": fire_risk
    })

    return data


def save_dataset(data, path="./data/wildfire_sample.csv"):
    """Save dataset to CSV file, ensuring directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)  # create folder if missing
    data.to_csv(path, index=False)


def load_dataset(path="./data/wildfire_sample.csv"):
    """Load dataset from CSV file."""
    return pd.read_csv(path)
