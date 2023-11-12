import numpy as np

signal_length = 1024
time_signal = np.cos(np.pi / 10 * np.arange(signal_length))

# print(time_signal)

signal_fft = np.fft.fft(time_signal)

print(signal_fft)
print(np.abs(signal_fft))