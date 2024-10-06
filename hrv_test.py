import os
import sys
import numpy as np
from biosppy import storage
from biosppy.signals import ecg
from biosppy.signals.acc import acc
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_hrv_rmssd(r_peaks, sampling_rate=1000.0):
    print("R-peaks: " + str(r_peaks))
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / sampling_rate
    print("RR intervals (seconds):", rr_intervals)
    
    # Calculate differences between successive RR intervals
    rr_diffs = np.diff(rr_intervals)
    print("RR interval differences (seconds):", rr_diffs)
    
    # Calculate RMSSD using the correct formula
    rmssd = np.sqrt(np.mean(np.square(rr_diffs)))
    
    # Calculate mean heart rate
    mean_hr = 60 / np.mean(rr_intervals)
    
    return rmssd, mean_hr, len(r_peaks)

def process_ecg_and_calculate_hrv(file_path, sampling_rate=1000.0):
    # load raw ECG signal
    ecg_signal, _ = storage.load_txt(file_path)

    # Setting current path
    current_dir = os.path.dirname(sys.argv[0])
    ecg_plot_path = os.path.join(current_dir, 'ecg.png')

    # Process ECG and plot
    out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, path=ecg_plot_path, show=False)

    # Calculate HRV using RMSSD method
    rmssd, mean_hr, num_peaks = calculate_hrv_rmssd(out_ecg['rpeaks'], sampling_rate)

    # Print results
    print(f"RMSSD: {rmssd:.4f} seconds")
    print(f"Mean Heart Rate: {mean_hr:.2f} bpm")
    print(f"Number of R-peaks detected: {num_peaks}")

    # Plot the ECG signal with detected R-peaks
    plt.figure(figsize=(12, 6))
    plt.plot(ecg_signal)
    plt.plot(out_ecg['rpeaks'], ecg_signal[out_ecg['rpeaks']], "ro")
    plt.title("ECG Signal with Detected R-peaks")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig('ecg_with_rpeaks_biosppy.png')
    plt.close()
    
    print("ECG plot with R-peaks saved as ecg_with_rpeaks_biosppy.png")

    return rmssd, mean_hr, num_peaks

# Main execution
if __name__ == "__main__":
    # Specify the path to your ECG file
    # ecg_file_path = './examples/ecg.txt'
    # ecg_file_path = './examples/test.txt'
    ecg_file_path = './examples/norm.txt'

    # Process ECG and calculate HRV
    rmssd, mean_hr, num_peaks = process_ecg_and_calculate_hrv(ecg_file_path)

    # Save results to a file
    with open('hrv_results.txt', 'w') as f:
        f.write(f"Sampling Rate: 125 Hz\n")
        f.write(f"Number of R-peaks detected: {num_peaks}\n")
        f.write(f"RMSSD: {rmssd:.4f} seconds\n")
        f.write(f"Mean Heart Rate: {mean_hr:.2f} bpm\n")
    
    print("Results saved to hrv_results.txt")

# The ACC processing part is commented out as it's not directly related to HRV calculation
# If you need to process ACC data, you can uncomment and modify as needed:
# acc_signal, _ = storage.load_txt('./examples/acc.txt')
# acc_plot_path = os.path.join(current_dir, 'acc.png')
# out_acc = acc(signal=acc_signal, sampling_rate=1000., path=acc_plot_path, interactive=True)