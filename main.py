import os
import wfdb
import numpy as np
from biosppy.signals import ecg
from biosppy import storage

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

    try:
        # Process ECG
        out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=record.fs, show=False, interactive=False)

        # Skip processing if not enough R-peaks are detected
        if len(out_ecg['rpeaks']) < 2:
            print(f"Skipping {record.record_name}: Not enough R-peaks detected.")
            return None

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

    except ValueError as e:
        if "Not enough beats to compute heart rate" in str(e):
            print(f"Skipping {record.record_name}: Not enough R-peaks detected.")
            return None
        else:
            raise

if __name__ == "__main__":
    input_directory = r"./Shao/WFDB_ShaoxingUniv"
    output_directory = r"./Shao_Results"

    # Process all records in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.hea'):
            record_name = os.path.splitext(filename)[0]
            record_path = os.path.join(input_directory, record_name)
            results = process_ecg(record_path, output_directory)
            if results is None:
                continue
            print(f"Processed {results['record_name']} (Channel {results['channel']}):")
            print(f"Age: {results['age']}")
            print(f"Sex: {results['sex']}")
            print(f"Dx: {results['dx']}")
            print(f"RMSSD = {results['rmssd']:.4f} s")
            print(f"Mean HR = {results['mean_hr']:.2f} bpm")
            print(f"Number of R-peaks = {results['num_peaks']}")
            print(f"Results saved in {output_directory}")
