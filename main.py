import os
import sys
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from biosppy import storage
from biosppy.plotting import plot_ecg

def process_ecg(record_path, output_dir, channel=0):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the record and header
    record = wfdb.rdrecord(record_path, channels=[channel])
    header = wfdb.rdheader(record_path)

    # Access the signal data (first channel)
    signal = record.p_signal[:, 0]

    # Save signal to txt file
    txt_file_path = os.path.join(output_dir, f"{record.record_name}_channel_{channel}.txt")
    np.savetxt(txt_file_path, signal, delimiter=',')

    # Load the saved txt file
    ecg_signal, _ = storage.load_txt(txt_file_path)

    # Setting current path and plot paths
    ecg_plot_path = os.path.join(output_dir, f"{record.record_name}_ecg.png")
    ecg_with_rpeaks_path = os.path.join(output_dir, f"{record.record_name}_ecg_with_rpeaks_biosppy.png")
    ecg_summary_path = os.path.join(output_dir, f"{record.record_name}_ecg-summary.png")
    ecg_templates_path = os.path.join(output_dir, f"{record.record_name}_ecg-templates.png")

    # Process ECG
    out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=record.fs, show=False, interactive=False)

    # Plot and save ECG with R-peaks
    plt.figure(figsize=(12, 6))
    plt.plot(ecg_signal)
    plt.plot(out_ecg['rpeaks'], ecg_signal[out_ecg['rpeaks']], "ro")
    plt.title(f"ECG Signal with Detected R-peaks - {record.record_name}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(ecg_with_rpeaks_path)
    plt.close()

    # Plot and save ECG summary
    plot_ecg(ts=out_ecg['ts'], 
             raw=out_ecg['filtered'], 
             filtered=out_ecg['filtered'], 
             rpeaks=out_ecg['rpeaks'], 
             templates_ts=out_ecg['templates_ts'], 
             templates=out_ecg['templates'], 
             heart_rate_ts=out_ecg['heart_rate_ts'], 
             heart_rate=out_ecg['heart_rate'],
             path=ecg_summary_path,
             show=False)

    # Plot and save ECG templates
    plt.figure(figsize=(10, 4))
    plt.plot(out_ecg['templates_ts'], out_ecg['templates'].T)
    plt.title(f"ECG Templates - {record.record_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(ecg_templates_path)
    plt.close()

    # Calculate HRV
    rr_intervals = np.diff(out_ecg['rpeaks']) / record.fs
    rr_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(rr_diffs)))
    mean_hr = 60 / np.mean(rr_intervals)

    # Extract additional information
    age = header.comments[0].split(": ")[1] if header.comments else "Unknown"
    sex = header.comments[1].split(": ")[1] if len(header.comments) > 1 else "Unknown"
    dx = header.comments[2].split(": ")[1] if len(header.comments) > 2 else "Unknown"

    # Save results to txt file
    with open(os.path.join(output_dir, f"{record.record_name}_results.txt"), 'w') as f:
        f.write(f"#Age: {age}\n")
        f.write(f"#Sex: {sex}\n")
        f.write(f"#Dx: {dx}\n")
        f.write(f"Record name: {record.record_name}\n")
        f.write(f"Number of signals: {record.n_sig}\n")
        f.write(f"Sampling frequency: {record.fs}\n")
        f.write(f"Number of samples: {record.sig_len}\n")
        f.write(f"Processed {record.record_name} (Channel {channel}):\n")
        f.write(f"RMSSD = {rmssd:.4f} s\n")
        f.write(f"Mean HR = {mean_hr:.2f} bpm\n")
        f.write(f"Number of R-peaks = {len(out_ecg['rpeaks'])}\n")

    print(f"ECG plot with R-peaks saved as {ecg_with_rpeaks_path}")
    print(f"ECG summary plot saved as {ecg_summary_path}")
    print(f"ECG templates plot saved as {ecg_templates_path}")

    return {
        'record_name': record.record_name,
        'channel': channel,
        'rmssd': rmssd,
        'mean_hr': mean_hr,
        'num_peaks': len(out_ecg['rpeaks']),
        'age': age,
        'sex': sex,
        'dx': dx
    }

if __name__ == "__main__":
    input_directory = r"./Training_PTB"
    output_directory = r"./PTB_Results"

    # Process a single record
    # record_name = "S0001"  # Change this to process different records
    # record_path = os.path.join(input_directory, record_name)
    
    # results = process_ecg(record_path, output_directory)

    # print(f"Processed {results['record_name']} (Channel {results['channel']}):")
    # print(f"Age: {results['age']}")
    # print(f"Sex: {results['sex']}")
    # print(f"Dx: {results['dx']}")
    # print(f"RMSSD = {results['rmssd']:.4f} s")
    # print(f"Mean HR = {results['mean_hr']:.2f} bpm")
    # print(f"Number of R-peaks = {results['num_peaks']}")
    # print(f"Results saved in {output_directory}")

    # If you want to process all records in the directory, you can use this:
    for filename in os.listdir(input_directory):
        if filename.endswith('.hea'):
            record_name = os.path.splitext(filename)[0]
            record_path = os.path.join(input_directory, record_name)
            results = process_ecg(record_path, output_directory)
            print(f"Processed {results['record_name']} (Channel {results['channel']}):")
            print(f"Age: {results['age']}")
            print(f"Sex: {results['sex']}")
            print(f"Dx: {results['dx']}")
            print(f"RMSSD = {results['rmssd']:.4f} s")
            print(f"Mean HR = {results['mean_hr']:.2f} bpm")
            print(f"Number of R-peaks = {results['num_peaks']}")
            print(f"Results saved in {output_directory}")