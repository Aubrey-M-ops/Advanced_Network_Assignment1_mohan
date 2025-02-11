# Assignment 1 base code

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sionna.channel.tr38901 import TDL
from sionna.ofdm import ResourceGrid
from sionna.channel import OFDMChannel

# Simulation Parameters
NUM_BS = 3  # Number of base stations
P_BS = 10  # Base station transmit power (W)
NOISE_POWER = 1e-9  # Noise power (W)
SINR_MIN = 5  # Minimum SINR threshold
CARRIER_FREQ = 3.5e9  # Carrier frequency (3.5 GHz)
BANDWIDTH = 20e6  # Channel bandwidth (20 MHz)
OFDM_SYMBOLS = 14  # Number of OFDM symbols per frame
FFT_SIZE = 64  # Number of subcarriers
SUBCARRIER_SPACING = 15e3  # Subcarrier spacing in Hz
DATASET_PATH = "mnt/data"

# Load base station locations


def load_base_stations(file_path):
    """Load base station locations from a CSV file."""
    print(f"Loading base station locations from {file_path}")
    df = pd.read_csv(file_path)
    return df.values

# Load mobility data


def load_mobility_data(file_path):
    """Load mobility data from a CSV file."""
    print(f"Loading mobility data from {file_path}")
    df = pd.read_csv(file_path)
    return df


# Initialize Sionna OFDM Channel Model
print("Initializing Sionna OFDM Channel Model...")
tdl_model = TDL(model="A", delay_spread=100e-9, carrier_frequency=CARRIER_FREQ)
resource_grid = ResourceGrid(
    num_ofdm_symbols=OFDM_SYMBOLS,
    fft_size=FFT_SIZE,
    subcarrier_spacing=SUBCARRIER_SPACING
)
channel = OFDMChannel(
    resource_grid=resource_grid,
    channel_model=tdl_model,
    add_awgn=True
)

# Compute channel gains


def compute_channel_gain(bs_positions, mn_positions):
    """Compute channel gains based on distance, path loss, fading, and shadowing."""
    print("Computing channel gains...")
    # Calculate distance between BS and MN
    
    # Compute path loss
    # Apply fading and shadowing
    # Return channel gain values
    pass  # Students to implement

# Compute SINR values


def compute_sinr(channel_gain, power_bs, noise_power):
    """Compute SINR based on received signal power, interference, and noise."""
    print("Computing SINR...")
    # Calculate signal power for each MN from assigned BS
    # Compute interference power from other BSs
    # Calculate SINR for each MN-BS pair
    # Return SINR values
    pass  # Students to implement

# Optimization Problem using Gurobi


def optimize_throughput():
    """Formulate and solve an optimization problem to maximize throughput."""
    print("Setting up optimization problem...")
    # Define decision variables
    # Define constraints (assignment, SINR constraints, binary constraints)
    # Set the objective function
    # Solve the optimization problem
    pass  # Students to implement

# Process all mobility datasets


def process_all_datasets():
    """Load datasets, compute channel gains, compute SINR, and optimize throughput."""
    print("Processing all datasets...")
    bs_positions = load_base_stations(
        os.path.join(DATASET_PATH, "base_stations.csv"))
    mobility_files = sorted([f for f in os.listdir(
        DATASET_PATH) if f.startswith("mobility_data_t") and f.endswith(".csv")])

    for file in mobility_files:
        print(f"Processing {file}")
        mobility_data = load_mobility_data(os.path.join(DATASET_PATH, file))
        mn_positions = mobility_data[['x', 'y']].values

        # Compute channel gains
        channel_gain = compute_channel_gain(bs_positions, mn_positions)

        # Compute SINR
        sinr = compute_sinr(channel_gain, P_BS, NOISE_POWER)

        # Optimize throughput
        optimize_throughput()

    print("Processing Complete.")


# Run the dataset processing
print("Starting dataset processing...")
process_all_datasets()
print("All processing finished.")
