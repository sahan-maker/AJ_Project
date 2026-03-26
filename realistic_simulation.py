import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(42)

# -----------------------------
# Simulation parameters
# -----------------------------
fs = 10000          # Sampling frequency (Hz)
T = 0.5             # Duration of simulation (seconds)
t = np.arange(0, T, 1/fs)

# -----------------------------
# 1. Baseband noise
# -----------------------------
thermal_noise_power = 0.001
noise = np.sqrt(thermal_noise_power) * np.random.randn(len(t))

# -----------------------------
# 2. Narrowband interference
# -----------------------------
interference_freqs = [200, 500]  # Hz (example baseband tones)
interference_amp = [0.05, 0.03]
interference = np.zeros(len(t))
for f, a in zip(interference_freqs, interference_amp):
    interference += a * np.sin(2 * np.pi * f * t)

# -----------------------------
# 3. RTCM-like digital signal (BPSK)
# -----------------------------
bps = 200                  # bits per second
num_bits = int(T * bps)
bits = np.random.randint(0, 2, num_bits)  # random binary
symbols = 2*bits - 1  # Map 0->-1, 1->+1

# Upsample to sampling frequency
samples_per_bit = int(fs / bps)
signal_baseband = np.repeat(symbols, samples_per_bit)

# Pulse shaping: simple rectangular (for now)
# Truncate to match t length
signal_baseband = signal_baseband[:len(t)]

# -----------------------------
# 4. Total received signal
# -----------------------------
rx_signal = signal_baseband + noise + interference

# -----------------------------
# 5. Time-domain plot
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(t, rx_signal, label='Received Signal')
plt.plot(t, signal_baseband, alpha=0.7, label='Clean RTCM-like Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('UAV RF Baseband Signal with Noise & Interference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Frequency-domain (FFT)
# -----------------------------
fft_rx = np.fft.fft(rx_signal)
freqs = np.fft.fftfreq(len(t), 1/fs)

plt.figure(figsize=(12,6))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_rx)[:len(freqs)//2])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('FFT of Received UAV Signal')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Spectrogram
# -----------------------------
plt.figure(figsize=(12,6))
f, t_spec, Sxx = signal.spectrogram(rx_signal, fs, nperseg=256)
plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of UAV RF Baseband Signal')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()