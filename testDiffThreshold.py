from base_code import load_base_stations, DATASET_PATH, load_mobility_data, compute_channel_gain, compute_sinr, optimize_throughput, P_BS, NOISE_POWER, SINR_MIN
import os
import matplotlib.pyplot as plt

# Test the impact of different SINR_MIN values on throughput
SINR_MIN_VALUES = [0, 1, 3, 5, 10]


def process_with_varied_sinr():
    """Test different SINR thresholds and observe throughput impact."""
    bs_positions = load_base_stations(
        os.path.join(DATASET_PATH, "base_stations.csv"))
    mobility_files = sorted([f for f in os.listdir(
        DATASET_PATH) if f.startswith("mobility_data_t") and f.endswith(".csv")])

    throughput_values = []

    for sinr_threshold in SINR_MIN_VALUES:
        print(f"\nTesting SINR threshold: {sinr_threshold}")
        total_throughput = 0
        
        for file in mobility_files:
            print(f"Processing {file} with SINR_MIN = {sinr_threshold}")
            mobility_data = load_mobility_data(
                os.path.join(DATASET_PATH, file))
            mn_positions = mobility_data[['x', 'y']].values

            # Compute channel gains and SINR
            channel_gain = compute_channel_gain(bs_positions, mn_positions)
            sinr = compute_sinr(channel_gain, P_BS, NOISE_POWER)

            # Adjust the global SINR threshold
            global SINR_MIN
            SINR_MIN = sinr_threshold

            # Optimize throughput
            assignment_matrix, max_throughput = optimize_throughput(
                bs_positions, mn_positions, sinr)

            # Accumulate throughput
            total_throughput += max_throughput
            print(f"Max Throughput for SINR_MIN = {sinr_threshold}: {max_throughput}\n")
        
        # Store the aggregated throughput
        throughput_values.append(total_throughput)

    # Return the final throughput values after all iterations
    return throughput_values



# Run the SINR variation test
throughput_values = process_with_varied_sinr()
print("Throughput values:", throughput_values)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(SINR_MIN_VALUES, throughput_values,
         marker='o', color='b', linestyle='--')
plt.xlabel('SINR Threshold (dB)')
plt.ylabel('Optimized Throughput (bps/Hz)')
plt.title('Impact of SINR Threshold on Throughput')
plt.grid()
plt.show()


print("Testing complete.")
