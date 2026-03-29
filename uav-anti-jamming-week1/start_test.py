import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# -----------------------------
# 1. Reproducibility
# -----------------------------
np.random.seed(42)

# -----------------------------
# 2. User-configurable parameters
# -----------------------------
fs = 1000          # Sampling frequency (Hz)
T = 1              # Duration (seconds)
t = np.arange(0, T, 1/fs)  # Time vector

# Frequencies and amplitudes
components = [
    {"freq": 50, "amp": 1.0},
    {"freq": 120, "amp": 0.5},
]

noise_power = 0.2   # Variance of Gaussian noise

# -----------------------------
# 3. Generate clean signal
# -----------------------------
signal_clean = np.zeros(len(t))
for comp in components:
    signal_clean += comp["amp"] * np.sin(2 * np.pi * comp["freq"] * t)

# -----------------------------
# 4. Add Gaussian noise
# -----------------------------
noise = np.sqrt(noise_power) * np.random.randn(len(t))
signal_noisy = signal_clean + noise

# -----------------------------
# 5. Plot time-domain signals
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(t, signal_clean, label='Clean Signal', linewidth=2)
plt.plot(t, signal_noisy, label='Noisy Signal', alpha=0.7)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Time-domain: Clean vs Noisy Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Frequency-domain (FFT)
# -----------------------------
fft_clean = np.fft.fft(signal_clean)
fft_noisy = np.fft.fft(signal_noisy)
freqs = np.fft.fftfreq(len(t), 1/fs)

plt.figure(figsize=(12,6))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_clean)[:len(freqs)//2], label='Clean FFT', linewidth=2)
plt.plot(freqs[:len(freqs)//2], np.abs(fft_noisy)[:len(freqs)//2], label='Noisy FFT', alpha=0.7)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('Frequency-domain: Clean vs Noisy Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Spectrogram (time-frequency)
# -----------------------------
plt.figure(figsize=(12,6))
f, t_spec, Sxx = signal.spectrogram(signal_noisy, fs, nperseg=128)
plt.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Noisy Signal')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Save plots (optional)
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(t, signal_noisy)
plt.title('Noisy Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig("noisy_signal_plot.png")