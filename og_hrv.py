import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def load_ecg_data(file_path):
    """Load ECG data from the Simple Text Format file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata
    sampling_rate = float(lines[1].split(':=')[1].strip())
    resolution = int(lines[2].split(':=')[1].strip())
    
    # Extract ECG data
    ecg_data = np.array([float(line.strip()) for line in lines[4:]])
    
    return ecg_data, sampling_rate, resolution

def detect_r_peaks(ecg_data, sampling_rate):
    """Detect R-peaks in the ECG signal."""
    # Normalize the signal
    ecg_normalized = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
    
    # Find peaks
    peaks, _ = find_peaks(ecg_normalized, height=0.5, distance=sampling_rate//2)
    
    return peaks

def calculate_rmssd(rr_intervals):
    """Calculate RMSSD from RR intervals."""
    diff_rr = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff_rr**2))

def calculate_hrv(file_path):
    # Load ECG data
    ecg_data, sampling_rate, resolution = load_ecg_data(file_path)
    
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_data, sampling_rate)
    
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
    
    return rmssd, mean_hr

# Main execution
if __name__ == "__main__":
    file_path = './examples/ecg.txt'
    # file_path = './examples/conv_1.txt'
    rmssd, mean_hr = calculate_hrv(file_path)
    
    print(f"RMSSD: {rmssd:.4f} seconds")
    print(f"Mean Heart Rate: {mean_hr:.2f} bpm")
    
    # Save results to a file
    with open('hrv_results.txt', 'w') as f:
        f.write(f"RMSSD: {rmssd:.4f} seconds\n")
        f.write(f"Mean Heart Rate: {mean_hr:.2f} bpm\n")