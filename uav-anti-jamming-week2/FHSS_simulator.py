# Week 2: Frequency Hopping Spread Spectrum (FHSS) Simulator
# For UAV Anti-Jamming Research
# Features:
#   • Configurable channel list (evenly spaced or custom range)
#   • Pseudorandom hopping sequence (repeatable with seed)
#   • Beautiful Frequency vs Time hopping visualization (clear horizontal hops)
#   • Time-domain waveform showing actual RF signal
#   • Spectrum plot showing spread-spectrum effect
#   • Live hopping mode with real-time marker and current frequency display
#   • Clean professional GUI

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QGridLayout, QCheckBox, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


class FHSSSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 2: Frequency Hopping Spread Spectrum (FHSS) Simulator")
        self.resize(1350, 920)

        layout = QVBoxLayout()

        # ==================== CONTROLS ====================
        control_group = QGroupBox("FHSS Parameters")
        control_layout = QGridLayout()

        # Number of channels
        self.num_channels = QSpinBox()
        self.num_channels.setRange(4, 64)
        self.num_channels.setValue(16)

        # Frequency range
        self.f_min = QDoubleSpinBox()
        self.f_min.setRange(10, 2000)
        self.f_min.setValue(100)
        self.f_min.setSuffix(" Hz")

        self.f_max = QDoubleSpinBox()
        self.f_max.setRange(100, 5000)
        self.f_max.setValue(800)
        self.f_max.setSuffix(" Hz")

        # Hop rate
        self.hop_rate = QDoubleSpinBox()
        self.hop_rate.setRange(1, 100)
        self.hop_rate.setValue(10)
        self.hop_rate.setSuffix(" hops/s")

        # Duration
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 10)
        self.duration.setValue(4)
        self.duration.setSuffix(" s")

        # Sampling rate
        self.fs = QSpinBox()
        self.fs.setRange(5000, 200000)
        self.fs.setValue(50000)

        # Random seed (for reproducible hopping)
        self.seed_input = QLineEdit("42")
        self.seed_input.setMaximumWidth(80)

        control_layout.addWidget(QLabel("Number of Channels"), 0, 0)
        control_layout.addWidget(self.num_channels, 0, 1)
        control_layout.addWidget(QLabel("Min Frequency"), 1, 0)
        control_layout.addWidget(self.f_min, 1, 1)
        control_layout.addWidget(QLabel("Max Frequency"), 2, 0)
        control_layout.addWidget(self.f_max, 2, 1)
        control_layout.addWidget(QLabel("Hop Rate"), 3, 0)
        control_layout.addWidget(self.hop_rate, 3, 1)
        control_layout.addWidget(QLabel("Duration"), 4, 0)
        control_layout.addWidget(self.duration, 4, 1)
        control_layout.addWidget(QLabel("Sampling Rate (fs)"), 5, 0)
        control_layout.addWidget(self.fs, 5, 1)
        control_layout.addWidget(QLabel("Random Seed"), 6, 0)
        control_layout.addWidget(self.seed_input, 6, 1)

        control_group.setLayout(control_layout)

        # ==================== BUTTONS ====================
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate Channels & Sequence")
        self.btn_simulate = QPushButton("Simulate FHSS")
        self.btn_live = QPushButton("▶ Start Live Hopping")
        self.btn_reset = QPushButton("Reset")

        self.btn_generate.clicked.connect(self.generate_channels_and_sequence)
        self.btn_simulate.clicked.connect(self.simulate_fhss)
        self.btn_live.clicked.connect(self.toggle_live_hopping)
        self.btn_reset.clicked.connect(self.reset_all)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_simulate)
        button_layout.addWidget(self.btn_live)
        button_layout.addWidget(self.btn_reset)

        # ==================== INFO DISPLAY ====================
        info_group = QGroupBox("Channel List & Hopping Sequence")
        info_layout = QHBoxLayout()

        self.lbl_channels = QLabel("Channels: (click Generate)")
        self.lbl_channels.setWordWrap(True)
        self.lbl_sequence = QLabel("Hopping Sequence: (click Generate)")
        self.lbl_sequence.setWordWrap(True)

        info_layout.addWidget(self.lbl_channels)
        info_layout.addWidget(self.lbl_sequence)
        info_group.setLayout(info_layout)

        # ==================== PLOT AREA ====================
        self.figure = Figure(figsize=(12, 9))
        self.canvas = FigureCanvas(self.figure)

        # Live variables
        self.channels = None
        self.hopping_sequence = None          # list of frequencies
        self.t = None
        self.signal = None
        self.hop_starts = None
        self.hop_ends = None
        self.hop_freqs = None
        self.is_live = False
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_hop_update)
        self.live_timer.setInterval(80)       # smooth animation
        self.live_time = 0.0
        self.current_hop_index = 0

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(info_group)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    # ==================== GENERATION ====================
    def generate_channels_and_sequence(self):
        n_ch = self.num_channels.value()
        fmin = self.f_min.value()
        fmax = self.f_max.value()

        self.channels = np.linspace(fmin, fmax, n_ch)

        # Display channels
        ch_str = ", ".join([f"{f:.1f}" for f in self.channels[:12]])
        if len(self.channels) > 12:
            ch_str += f", ... ({len(self.channels)} total)"
        self.lbl_channels.setText(f"<b>Channels ({n_ch}):</b> {ch_str}")

        # Generate hopping sequence
        try:
            seed = int(self.seed_input.text())
        except:
            seed = 42
        np.random.seed(seed)

        hop_rate = self.hop_rate.value()
        duration = self.duration.value()
        num_hops = int(np.ceil(duration * hop_rate)) + 2

        # Random hopping (with replacement - typical for FHSS)
        hop_indices = np.random.randint(0, n_ch, size=num_hops)
        self.hopping_sequence = self.channels[hop_indices].tolist()

        # Display sequence (first 20 hops)
        seq_str = " → ".join([f"{f:.1f}" for f in self.hopping_sequence[:20]])
        if len(self.hopping_sequence) > 20:
            seq_str += " → ..."
        self.lbl_sequence.setText(f"<b>Hopping Sequence:</b> {seq_str}")

        # Auto-simulate after generation
        self.simulate_fhss()

    def simulate_fhss(self):
        if self.channels is None or self.hopping_sequence is None:
            self.generate_channels_and_sequence()
            return

        fs = self.fs.value()
        duration = self.duration.value()
        hop_rate = self.hop_rate.value()
        dwell = 1.0 / hop_rate

        self.t = np.arange(0, duration, 1.0 / fs)
        n_samples = len(self.t)
        self.signal = np.zeros(n_samples)

        # Hop timing
        hop_times = np.arange(0, duration + dwell, dwell)
        self.hop_starts = hop_times[:-1]
        self.hop_ends = hop_times[1:]
        self.hop_freqs = []

        # Build signal segment by segment
        for i in range(len(self.hop_starts)):
            start_idx = int(self.hop_starts[i] * fs)
            end_idx = min(int(self.hop_ends[i] * fs), n_samples)
            if start_idx >= n_samples:
                break
            f = self.hopping_sequence[i % len(self.hopping_sequence)]
            self.hop_freqs.append(f)
            segment_t = self.t[start_idx:end_idx]
            self.signal[start_idx:end_idx] = np.sin(2 * np.pi * f * segment_t)

        self.plot_fhss_static()

    # ==================== STATIC PLOTS ====================
    def plot_fhss_static(self):
        self.figure.clear()

        # Top: Frequency vs Time (classic FHSS visualization)
        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors='tab:blue', linewidth=10, alpha=0.85)
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title("FHSS Hopping Pattern – Frequency vs Time")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        # Middle: Time-domain signal
        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, self.signal, color='tab:orange', linewidth=0.8)
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Transmitted FHSS Signal (Time Domain)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        # Bottom: Spectrum (shows spread-spectrum effect)
        ax3 = self.figure.add_subplot(313)
        N = len(self.signal)
        fft_vals = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())
        magnitude_db = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-12)

        ax3.plot(freqs, magnitude_db, color='tab:purple', linewidth=1.1)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_title("FHSS Power Spectrum (Spread Spectrum)")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(self.f_min.value() * 0.9, self.f_max.value() * 1.1)

        self.figure.tight_layout()
        self.canvas.draw()

    # ==================== LIVE HOPPING MODE ====================
    def toggle_live_hopping(self):
        if not self.is_live:
            if self.signal is None:
                self.simulate_fhss()
            self.is_live = True
            self.btn_live.setText("⏹ Stop Live Hopping")
            self.live_time = 0.0
            self.current_hop_index = 0
            self.live_timer.start()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Live Hopping")

    def live_hop_update(self):
        if not self.is_live:
            return

        dwell = 1.0 / self.hop_rate.value()
        self.live_time += 0.08   # advance simulated time

        # Find current hop
        current_idx = int(self.live_time / dwell) % len(self.hop_freqs)
        current_freq = self.hop_freqs[current_idx]
        current_start = self.hop_starts[current_idx % len(self.hop_starts)]

        # Re-plot with live marker
        self.figure.clear()

        # 1. Frequency vs Time with live highlight
        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            color = 'tab:green' if i == current_idx else 'tab:blue'
            lw = 12 if i == current_idx else 8
            alpha = 1.0 if i == current_idx else 0.7
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors=color, linewidth=lw, alpha=alpha)

        # Live vertical line
        ax1.axvline(x=self.live_time, color='red', linewidth=2, linestyle='--')
        ax1.text(self.live_time + 0.05, self.f_max.value() * 1.05,
                 f"NOW", color='red', fontsize=11, fontweight='bold')

        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title(f"LIVE FHSS Hopping – Current Freq: {current_freq:.1f} Hz  |  t = {self.live_time:.2f} s")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        # 2. Time-domain (full signal with live window)
        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, self.signal, color='tab:orange', linewidth=0.8)
        # Highlight current dwell
        start_idx = int(current_start * self.fs.value())
        end_idx = min(start_idx + int(dwell * self.fs.value()), len(self.t))
        ax2.axvspan(self.t[start_idx], self.t[end_idx-1], alpha=0.3, color='green')
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Transmitted Signal (Live)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        # 3. Spectrum (static for reference)
        ax3 = self.figure.add_subplot(313)
        N = len(self.signal)
        fft_vals = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())
        magnitude_db = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-12)
        ax3.plot(freqs, magnitude_db, color='tab:purple', linewidth=1.1)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_title("Overall FHSS Spectrum")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(self.f_min.value() * 0.9, self.f_max.value() * 1.1)

        self.figure.tight_layout()
        self.canvas.draw()

        # Stop when we reach end of duration
        if self.live_time >= self.duration.value():
            self.toggle_live_hopping()

    def reset_all(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Live Hopping")
        self.channels = None
        self.hopping_sequence = None
        self.signal = None
        self.t = None
        self.lbl_channels.setText("Channels: (click Generate)")
        self.lbl_sequence.setText("Hopping Sequence: (click Generate)")
        self.figure.clear()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FHSSSimulator()
    window.show()
    sys.exit(app.exec_())