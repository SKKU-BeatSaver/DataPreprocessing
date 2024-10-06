import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    return resampled_data, output_sampling_rate

def load_ecg_data(file_path):
    """Load ECG data from a file, handling scientific notation and empty lines."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    sampling_rate = 125.0  # Assuming 125 Hz sampling rate
    resolution = 12  # As specified in the conversion function
    
    # Extract ECG data, skipping header lines and empty lines
    ecg_data = []
    for line in lines[4:]:  # Skip the first 4 lines (header)
        line = line.strip()
        if line:  # Skip empty lines
            try:
                ecg_data.append(int(line))
            except ValueError:
                print(f"Warning: Skipping invalid data point: {line}")
    
    ecg_data = np.array(ecg_data)
    
    return ecg_data, sampling_rate, resolution

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def detect_r_peaks(ecg_data, sampling_rate):
    """Detect R-peaks in the ECG signal using adaptive thresholding."""
    # Apply bandpass filter
    filtered_ecg = butter_bandpass_filter(ecg_data, 5, 15, sampling_rate)
    
    # Square the signal
    squared_ecg = filtered_ecg ** 2
    
    # Apply moving average
    window_size = int(sampling_rate * 0.15)
    moving_avg = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks using adaptive thresholding
    threshold = moving_avg.mean() + 2 * moving_avg.std()
    peaks, _ = find_peaks(moving_avg, height=threshold, distance=int(sampling_rate * 0.2))
    
    return peaks

def calculate_rmssd(rr_intervals):
    """Calculate RMSSD from RR intervals."""
    if len(rr_intervals) < 2:
        return None
    diff_rr = np.diff(rr_intervals)
    return np.sqrt(np.mean(np.square(diff_rr)))

def calculate_hrv(ecg_data, sampling_rate):
    try:
        if len(ecg_data) == 0:
            raise ValueError("No valid data points found in the file.")
        
        print(f"Analyzing {len(ecg_data)} data points.")
        print(f"First few data points: {ecg_data[:5]}")
        print(f"Sampling rate: {sampling_rate} Hz")
        
        # Detect R-peaks
        r_peaks = detect_r_peaks(ecg_data, sampling_rate)
        
        if len(r_peaks) < 2:
            raise ValueError(f"Not enough R-peaks detected (found {len(r_peaks)}). Check your ECG data and detection algorithm.")
        
        print(f"Detected {len(r_peaks)} R-peaks.")
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        # Calculate RMSSD
        rmssd = calculate_rmssd(rr_intervals)
        
        # Calculate mean heart rate
        mean_hr = 60 / np.mean(rr_intervals)
        
        # Plot ECG with detected R-peaks
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data)
        plt.plot(r_peaks, ecg_data[r_peaks], "ro")
        plt.title("ECG Signal with Detected R-peaks")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.savefig('ecg_with_rpeaks.png')
        plt.close()
        
        return rmssd, mean_hr, len(r_peaks)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, 0

# Main execution
if __name__ == "__main__":
    input_file = './examples/test.txt'  # Update this path to your actual input file location
    converted_file = "converted_data.txt"  # File to store converted data
    input_sampling_rate = 125  # Hz
    output_sampling_rate = 125  # Hz

    print(f"Converting data from: {input_file}")
    converted_data, sampling_rate = convert_data(input_file, converted_file, input_sampling_rate, output_sampling_rate)

    print(f"Analyzing converted ECG data")
    print(f"Using sampling rate: {sampling_rate} Hz")
    rmssd, mean_hr, num_peaks = calculate_hrv(converted_data, sampling_rate)

    if rmssd is not None and mean_hr is not None:
        print(f"Number of R-peaks detected: {num_peaks}")
        print(f"RMSSD: {rmssd:.4f} seconds")
        print(f"Mean Heart Rate: {mean_hr:.2f} bpm")

        # Save results to a file
        with open('hrv_results.txt', 'w') as f:
            f.write(f"Sampling Rate: {sampling_rate} Hz\n")
            f.write(f"Number of R-peaks detected: {num_peaks}\n")
            f.write(f"RMSSD: {rmssd:.4f} seconds\n")
            f.write(f"Mean Heart Rate: {mean_hr:.2f} bpm\n")
        
        print("Results saved to hrv_results.txt")
        print("ECG plot with R-peaks saved as ecg_with_rpeaks.png")
    else:
        print("Failed to calculate HRV measures. Please check your input data and error messages.")