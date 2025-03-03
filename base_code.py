# Assignment 1 base code

# Import necessary libraries
import os
import numpy as np
import pandas as pd
# import tensorflow as tf
from sionna.channel.tr38901 import TDL
from sionna.ofdm import ResourceGrid
from sionna.channel import OFDMChannel
import gurobipy as gp
from gurobipy import GRB

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
DATASET_PATH = "database"

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
    total_bs = bs_positions.shape[0]
    total_mn = mn_positions.shape[0]
    channel_gains = np.zeros((total_bs, total_mn))

    # Calculate distance between BS and MN
    distances = np.linalg.norm(
        bs_positions[:, np.newaxis, :] - mn_positions[np.newaxis, :, :], axis=-1)
    # Compute path loss using log-distance path loss model
    path_loss =10 * np.log10(distances**4 + 1e-9)
    # Apply fading (Rayleigh fading)
    fading = np.random.rayleigh(scale=1.0, size=(total_bs, total_mn))
    # Apply shadowing (log-normal shadowing)
    shadowing = np.random.normal(loc=0, scale=2, size=(total_bs, total_mn))
    # Compute channel gain
    channel_gains = 10**((-path_loss + shadowing + fading) / 10)

    return channel_gains

# Compute SINR values
def compute_sinr(channel_gain, power_bs, noise_power):
    """Compute SINR based on received signal power, interference, and noise."""
    print("Computing SINR...")
    total_bs, total_mn = channel_gain.shape
    sinr = np.zeros((total_bs, total_mn))

    for j in range(total_mn):
        for i in range(total_bs):
            # Calculate signal power for each MN from assigned BS(linear)
            signal_power = power_bs * channel_gain[i, j] 
            # Compute interference power from other BSs
            interference_power = np.sum([power_bs * channel_gain[k, j] for k in range(total_bs) if k != i])
            # calculate SINR 
            sinr[i][j] = signal_power / (noise_power + interference_power)
            sinr[i][j] = np.maximum(sinr[i][j], 1e-6)
    return sinr

# Optimization Problem using Gurobi
def optimize_throughput(bs_positions, mn_positions, sinr):
    """Formulate and solve an optimization problem to maximize throughput."""
    print("Setting up optimization problem...")
    model = gp.Model("Throughput Maximization")
    total_bs = bs_positions.shape[0]
    total_mn = mn_positions.shape[0]
    # 1️⃣ Define decision variables
    # Create a 2D matrix of binary variables a[i, j] for assignment
    a = model.addVars(total_bs, total_mn, vtype=GRB.BINARY, name="a")
    # 2️⃣ Define constraints (assignment, SINR constraints, binary constraints)
    # Constraint 1 (assignment): Each mobile node can only be assigned to one base station
    for j in range(total_mn):
        model.addConstr(gp.quicksum(a[i, j] for i in range(
            total_bs)) == 1, name=f"assign_mn_{j}")
    # Constraint 2(SINR constraint): Ensure assigned mobile nodes meet the minimum SINR threshold
    for i in range(total_bs):
        for j in range(total_mn):
            model.addConstr(a[i, j] * sinr[i, j] >= SINR_MIN,
                            name=f"SINR_constraint_{i}_{j}")
    # Constraint 3(Binary constraint): Association variable must be binary (0 or 1)
    # This is handled implicitly by vtype=GRB.BINARY
    # 3️⃣ Set the objective function
    model.setObjective(
        gp.quicksum(a[i, j] * np.log2(1 + sinr[i, j])
                    for i in range(total_bs) for j in range(total_mn)),
        GRB.MAXIMIZE
    )
    # 4️⃣ Solve the optimization problem
    model.optimize()
    # Retrieve results
    assignment_matrix = np.zeros((total_bs, total_mn))
    max_throughput = 0
    if model.status == GRB.OPTIMAL:
        for i in range(total_bs):
            for j in range(total_mn):
                assignment_matrix[i, j] = a[i, j].X  # Extract binary values
    # Get the optimized throughput (objective function value)
    max_throughput = model.objVal

    return assignment_matrix, max_throughput

# Process all mobility datasets

def process_all_datasets():
    """Load datasets, compute channel gains, compute SINR, and optimize throughput."""
    print("Processing all datasets...")
    bs_positions = load_base_stations(
        os.path.join(DATASET_PATH, "base_stations.csv"))
    mobility_files = sorted([f for f in os.listdir(
        DATASET_PATH) if f.startswith("mobility_data_t") and f.endswith(".csv")])

    for file in mobility_files:
        print(f"Processing {file}👉👉👉👉👉👉👉👉")
        mobility_data = load_mobility_data(os.path.join(DATASET_PATH, file))
        mn_positions = mobility_data[['x', 'y']].values
        # Compute channel gains
        channel_gain = compute_channel_gain(bs_positions, mn_positions)
        # Compute SINR
        sinr = compute_sinr(channel_gain, P_BS, NOISE_POWER)
        # FIXME: transfer to number
        np.set_printoptions(suppress=True, precision=6)
        print("Channel Gain Matrix:\n", channel_gain)
        print("SINR Matrix:\n", sinr)

        # Optimize throughput
        best_assignment, max_throughput = optimize_throughput(
            bs_positions, mn_positions, sinr)
        print("Optimized Assignment Matrix 👇 \n", best_assignment)
        print("max throughput: ", max_throughput)

    print("Processing Complete.")


# Run the dataset processing
print("Starting dataset processing...")
process_all_datasets()
print("All processing finished.")
