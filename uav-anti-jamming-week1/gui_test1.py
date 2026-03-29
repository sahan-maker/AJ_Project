import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal

# -----------------------------
# Simulation Function
# -----------------------------
def run_simulation():
    # Get parameters from GUI
    fs = int(fs_entry.get())
    duration = float(duration_entry.get())
    noise_power = float(noise_entry.get())
    bps = int(bps_entry.get())

    t = np.arange(0, duration, 1/fs)

    # -----------------------------
    # RTCM-like digital signal (BPSK)
    # -----------------------------
    num_bits = int(duration * bps)
    bits = np.random.randint(0, 2, num_bits)           # random bits
    symbols = 2*bits - 1                               # map 0->-1, 1->+1
    samples_per_bit = int(fs / bps)
    signal_baseband = np.repeat(symbols, samples_per_bit)
    signal_baseband = signal_baseband[:len(t)]

    # -----------------------------
    # Noise
    # -----------------------------
    noise = np.sqrt(noise_power) * np.random.randn(len(t))

    # -----------------------------
    # Narrowband interference (random tones)
    # -----------------------------
    interference_freqs = np.random.uniform(100, 500, size=2)  # random Hz
    interference_amp = [0.05, 0.03]
    interference = np.zeros(len(t))
    for f, a in zip(interference_freqs, interference_amp):
        interference += a * np.sin(2 * np.pi * f * t)

    # -----------------------------
    # Total received signal
    # -----------------------------
    rx_signal = signal_baseband + noise + interference

    # -----------------------------
    # Clear previous plots
    # -----------------------------
    ax_time.clear()
    ax_fft.clear()
    ax_spec.clear()

    # -----------------------------
    # Time-domain plot
    # -----------------------------
    ax_time.plot(t, rx_signal, label='Received Signal')
    ax_time.plot(t, signal_baseband, alpha=0.7, label='Clean RTCM-like Signal')
    ax_time.set_title('Time-domain Signal')
    ax_time.set_xlabel('Time [s]')
    ax_time.set_ylabel('Amplitude')
    ax_time.legend()
    ax_time.grid(True)

    # -----------------------------
    # Frequency-domain (FFT)
    # -----------------------------
    fft_rx = np.fft.fft(rx_signal)
    freqs = np.fft.fftfreq(len(t), 1/fs)
    ax_fft.plot(freqs[:len(freqs)//2], np.abs(fft_rx)[:len(freqs)//2])
    ax_fft.set_title('Frequency-domain (FFT)')
    ax_fft.set_xlabel('Frequency [Hz]')
    ax_fft.set_ylabel('Magnitude')
    ax_fft.grid(True)

    # -----------------------------
    # Spectrogram
    # -----------------------------
    f, t_spec, Sxx = signal.spectrogram(rx_signal, fs, nperseg=256)
    ax_spec.pcolormesh(t_spec, f, 10*np.log10(Sxx), shading='gouraud')
    ax_spec.set_title('Spectrogram')
    ax_spec.set_xlabel('Time [s]')
    ax_spec.set_ylabel('Frequency [Hz]')

    canvas.draw()

# -----------------------------
# Tkinter GUI Setup
# -----------------------------
root = tk.Tk()
root.title("Week 1 UAV RF Simulation GUI")

# -----------------------------
# Control Frame
# -----------------------------
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

ttk.Label(frame, text="Sampling Frequency (Hz)").pack(pady=2)
fs_entry = ttk.Entry(frame)
fs_entry.insert(0, "10000")
fs_entry.pack(pady=2)

ttk.Label(frame, text="Duration (s)").pack(pady=2)
duration_entry = ttk.Entry(frame)
duration_entry.insert(0, "0.5")
duration_entry.pack(pady=2)

ttk.Label(frame, text="Noise Power").pack(pady=2)
noise_entry = ttk.Entry(frame)
noise_entry.insert(0, "0.001")
noise_entry.pack(pady=2)

ttk.Label(frame, text="Bits per second (RTCM)").pack(pady=2)
bps_entry = ttk.Entry(frame)
bps_entry.insert(0, "200")
bps_entry.pack(pady=2)

ttk.Button(frame, text="Run Simulation", command=run_simulation).pack(pady=10)

# -----------------------------
# Plot Frame
# -----------------------------
fig, (ax_time, ax_fft, ax_spec) = plt.subplots(3,1, figsize=(8,8))
plt.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

root.mainloop()