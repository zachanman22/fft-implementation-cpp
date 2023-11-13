import numpy as np
import time

signal_length = 268435456 / 4
# signal_length = 1024
time_signal = np.cos(np.pi / 10 * np.arange(signal_length))

# print(time_signal)

start = time.perf_counter()

signal_fft = np.fft.fft(time_signal)

stop = time.perf_counter()

diff = (stop - start) * 10 ** 6

print(diff)

# print(signal_fft)
# print(np.abs(signal_fft))