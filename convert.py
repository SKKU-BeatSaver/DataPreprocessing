import numpy as np

def convert_data(input_file, output_file, input_sampling_rate, output_sampling_rate):
    # Read data from the input file
    with open(input_file, 'r') as f:
        data = [float(line.strip()) for line in f if line.strip()]

    # Convert to numpy array for easier operations
    data_array = np.array(data)

    # Define the new range
    min_new, max_new = 2044, 2067

    # Scale the data
    scaled_data = min_new + (data_array - np.min(data_array)) * (max_new - min_new) / (np.max(data_array) - np.min(data_array))

    # Round to integers
    rounded_data = np.round(scaled_data).astype(int)

    # Resample the data to match the new sampling rate
    original_time = np.arange(len(rounded_data)) / input_sampling_rate
    new_time = np.arange(0, original_time[-1], 1/output_sampling_rate)
    resampled_data = np.interp(new_time, original_time, rounded_data)

    # Write the result to the output file
    with open(output_file, 'w') as f:
        f.write("# Simple Text Format\n")
        f.write(f"# Sampling Rate (Hz):= {output_sampling_rate:.2f}\n")
        f.write("# Resolution:= 12\n")
        f.write("# Labels:= ECG\n")
        for value in resampled_data:
            f.write(f"{int(value)}\n")

    print(f"Conversion complete. Results written to {output_file}")

# Example usage
input_file = './examples/test.txt'  # Update this path to your actual file location
output_file = "output_data.txt"  # Replace with your desired output file name
input_sampling_rate = 125  # Hz
output_sampling_rate = 125  # Hz

convert_data(input_file, output_file, input_sampling_rate, output_sampling_rate)