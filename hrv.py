import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_ecg_data(file_path):
    """Load ECG data from a file, handling scientific notation and empty lines."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Update the default sampling rate to 125 Hz
    sampling_rate = 125.0  # Changed from 1000.0 to 125.0
    resolution = 12  # Assuming 12-bit resolution as default
    
    # Extract ECG data, skipping empty lines and converting from scientific notation
    ecg_data = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                ecg_data.append(float(line))
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
    window_size = int(sampling_rate * 0.15)  # This will now be smaller due to lower sampling rate
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

def calculate_hrv(file_path):
    try:
        # Load ECG data
        ecg_data, sampling_rate, resolution = load_ecg_data(file_path)
        
        if len(ecg_data) == 0:
            raise ValueError("No valid data points found in the file.")
        
        print(f"Loaded {len(ecg_data)} data points.")
        print(f"First few data points: {ecg_data[:5]}")
        print(f"Sampling rate: {sampling_rate} Hz, Resolution: {resolution} bits")
        
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
    file_path = './examples/conv_1.txt'  # Update this path to your actual file location
    
    print(f"Analyzing ECG data from: {file_path}")
    print(f"Using sampling rate: 125 Hz")  # Add this line to confirm the new sampling rate
    rmssd, mean_hr, num_peaks = calculate_hrv(file_path)

    if rmssd is not None and mean_hr is not None:
        print(f"Number of R-peaks detected: {num_peaks}")
        print(f"RMSSD: {rmssd:.4f} seconds")
        print(f"Mean Heart Rate: {mean_hr:.2f} bpm")

        # Save results to a file
        with open('hrv_results.txt', 'w') as f:
            f.write(f"Sampling Rate: 125 Hz\n")  # Add this line to the results file
            f.write(f"Number of R-peaks detected: {num_peaks}\n")
            f.write(f"RMSSD: {rmssd:.4f} seconds\n")
            f.write(f"Mean Heart Rate: {mean_hr:.2f} bpm\n")
        
        print("Results saved to hrv_results.txt")
        print("ECG plot with R-peaks saved as ecg_with_rpeaks.png")
    else:
        print("Failed to calculate HRV measures. Please check your input data and error messages.")