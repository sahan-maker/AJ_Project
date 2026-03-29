# Week 1: Multi-Signal RF Simulator (UAV Anti‑Jamming Research) - IMPROVED VERSION
# Changes:
#   • Spectrum now uses one-sided FFT (only positive frequencies 0 → fs/2) → no more negative frequencies
#   • Magnitude displayed in dB (normalized to 0 dB peak) for realistic RF spectrum analyzer look
#   • "Show Spectrum" button now shows BOTH time-domain AND frequency-domain plots (stacked) for a complete realistic view
#   • Digital signals (BPSK, QPSK, RTCM-like) now always use full length (no truncation) → accurate carrier frequency display
#   • Cleaner, more professional plots with grids and tight layout

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RFSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi‑Signal RF Environment Simulator")
        self.resize(1200, 800)

        layout = QVBoxLayout()

        # ------------------ Controls ------------------
        control_group = QGroupBox("Simulation Controls")
        control_layout = QGridLayout()

        # Signal type
        self.signal_type = QComboBox()
        self.signal_type.addItems([
            "Sine",
            "BPSK",
            "QPSK",
            "Chirp",
            "RTCM-like"
        ])

        # Frequency
        self.freq = QDoubleSpinBox()
        self.freq.setRange(1, 5000)
        self.freq.setValue(100)
        self.freq.setSuffix(" Hz")

        # Sampling rate
        self.fs = QSpinBox()
        self.fs.setRange(1000, 100000)
        self.fs.setValue(10000)

        # Duration
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 5)
        self.duration.setValue(1)
        self.duration.setSuffix(" s")

        # Noise level
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0, 2)
        self.noise.setSingleStep(0.05)
        self.noise.setValue(0.1)

        # Jammer amplitude
        self.jammer = QDoubleSpinBox()
        self.jammer.setRange(0, 3)
        self.jammer.setValue(0.5)

        control_layout.addWidget(QLabel("Signal Type"), 0, 0)
        control_layout.addWidget(self.signal_type, 0, 1)

        control_layout.addWidget(QLabel("Carrier Frequency"), 1, 0)
        control_layout.addWidget(self.freq, 1, 1)

        control_layout.addWidget(QLabel("Sampling Rate"), 2, 0)
        control_layout.addWidget(self.fs, 2, 1)

        control_layout.addWidget(QLabel("Duration"), 3, 0)
        control_layout.addWidget(self.duration, 3, 1)

        control_layout.addWidget(QLabel("Noise Level"), 4, 0)
        control_layout.addWidget(self.noise, 4, 1)

        control_layout.addWidget(QLabel("Jammer Amplitude"), 5, 0)
        control_layout.addWidget(self.jammer, 5, 1)

        control_group.setLayout(control_layout)

        # ------------------ Buttons ------------------
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate Signal")
        self.btn_noise = QPushButton("Add Noise")
        self.btn_jam = QPushButton("Add Jammer")
        self.btn_fft = QPushButton("Show Spectrum (dB)")
        self.btn_reset = QPushButton("Reset")

        self.btn_generate.clicked.connect(self.generate_signal)
        self.btn_noise.clicked.connect(self.add_noise)
        self.btn_jam.clicked.connect(self.add_jammer)
        self.btn_fft.clicked.connect(self.show_fft)
        self.btn_reset.clicked.connect(self.reset_signal)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_noise)
        button_layout.addWidget(self.btn_jam)
        button_layout.addWidget(self.btn_fft)
        button_layout.addWidget(self.btn_reset)

        # ------------------ Plot ------------------
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.t = None
        self.signal = None

    # ------------------ Signal Generators (FIXED: full length, no truncation) ------------------
    def generate_signal(self):
        fs = self.fs.value()
        duration = self.duration.value()
        f = self.freq.value()

        self.t = np.arange(0, duration, 1/fs)
        sig_type = self.signal_type.currentText()
        n = len(self.t)

        if sig_type == "Sine":
            self.signal = np.sin(2 * np.pi * f * self.t)

        elif sig_type == "BPSK":
            samples_per_sym = 50
            num_sym = (n + samples_per_sym - 1) // samples_per_sym
            bits = np.random.choice([-1, 1], size=num_sym)
            symbols = np.repeat(bits, samples_per_sym)[:n]
            carrier = np.sin(2 * np.pi * f * self.t)
            self.signal = symbols * carrier

        elif sig_type == "QPSK":
            samples_per_sym = 50
            num_sym = (n + samples_per_sym - 1) // samples_per_sym
            symbols = np.random.randint(0, 4, size=num_sym)
            phase = symbols * (np.pi / 2)
            phases = np.repeat(phase, samples_per_sym)[:n]
            self.signal = np.cos(2 * np.pi * f * self.t + phases)

        elif sig_type == "Chirp":
            f1 = f
            f2 = f * 5
            self.signal = np.sin(2 * np.pi * (f1 + (f2 - f1) * self.t / self.t[-1]) * self.t)

        elif sig_type == "RTCM-like":
            samples_per_sym = 200
            num_sym = (n + samples_per_sym - 1) // samples_per_sym
            packet = np.random.choice([-1, 1], size=num_sym)
            burst = np.repeat(packet, samples_per_sym)[:n]
            carrier = np.sin(2 * np.pi * f * self.t)
            self.signal = burst * carrier

        self.plot_signal("Generated Signal")

    def add_noise(self):
        if self.signal is None:
            return
        noise = np.random.normal(0, self.noise.value(), len(self.signal))
        self.signal += noise
        self.plot_signal("Signal + Noise")

    def add_jammer(self):
        if self.signal is None:
            return
        jammer_freq = self.freq.value() * 1.8
        jammer = self.jammer.value() * np.sin(2 * np.pi * jammer_freq * self.t[:len(self.signal)])
        self.signal += jammer
        self.plot_signal("Signal + Jammer")

    # ------------------ IMPROVED: Realistic dB Spectrum (one-sided, both plots) ------------------
    def show_fft(self):
        if self.signal is None:
            return

        self.figure.clear()

        # Top: Time domain
        ax_time = self.figure.add_subplot(211)
        ax_time.plot(self.t[:len(self.signal)], self.signal)
        ax_time.set_title("Time Domain Signal")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        ax_time.grid(True)

        # Bottom: Frequency domain (one-sided, dB)
        ax_freq = self.figure.add_subplot(212)

        N = len(self.signal)
        fft_vals = np.fft.rfft(self.signal)          # one-sided FFT → only positive frequencies
        freqs = np.fft.rfftfreq(N, d=1 / self.fs.value())

        magnitude = np.abs(fft_vals)
        # dB scale, normalized to peak (0 dB max) → realistic RF spectrum look
        db_values = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

        ax_freq.plot(freqs, db_values, color='tab:blue')
        ax_freq.set_title("Frequency Spectrum (dB) – One-Sided")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude (dB)")
        ax_freq.grid(True)

        # Auto-limit to positive frequencies (0 to Nyquist)
        ax_freq.set_xlim(0, self.fs.value() / 2)

        self.figure.tight_layout()
        self.canvas.draw()

    def reset_signal(self):
        self.signal = None
        self.t = None
        self.figure.clear()
        self.canvas.draw()

    def plot_signal(self, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.t[:len(self.signal)], self.signal)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RFSimulator()
    window.show()
    sys.exit(app.exec_())