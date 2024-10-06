import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Read the record
record = wfdb.rdrecord('Training_PTB/S0001', channels=[0])  # Read first two channels (I and II)

# Print some information about the record
print(f"Record name: {record.record_name}")
print(f"Number of signals: {record.n_sig}")
print(f"Sampling frequency: {record.fs}")
print(f"Number of samples: {record.sig_len}")

# Access the signal data
signal_I = record.p_signal[:, 0]  # First channel
# signal_II = record.p_signal[:, 1]  # Second channel

# Plot the signals
plt.figure(figsize=(12, 6))
plt.plot(signal_I)
# plt.plot(signal_II)
plt.title("ECG Signals I")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")
plt.legend(['I'])
plt.show()

np.savetxt('ecg_data.txt', record.p_signal, delimiter=',')
