# Week 2: Frequency Hopping Spread Spectrum (FHSS) Simulator - ENHANCED
# Features added per your request:
#   • Select Signal Type (Sine, BPSK, QPSK, Chirp, RTCM-like) — each hop carries the chosen modulation
#   • Live hopping mode now shows "fake real-time" FFT of the CURRENT DWELL (small time window only)
#   • During live playback you see a dynamic spectrum analyzer view of exactly what is being transmitted right now
#   • Clean 3-plot live view: Hopping pattern + full waveform highlight + current-dwell dB spectrum

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FHSSSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 2: FHSS Simulator with Modulation & Live Dwell Spectrum")
        self.resize(1400, 950)

        layout = QVBoxLayout()

        # ==================== CONTROLS ====================
        control_group = QGroupBox("FHSS + Modulation Parameters")
        control_layout = QGridLayout()

        self.signal_type = QComboBox()
        self.signal_type.addItems(["Sine", "BPSK", "QPSK", "Chirp", "RTCM-like"])

        self.num_channels = QSpinBox()
        self.num_channels.setRange(4, 64)
        self.num_channels.setValue(16)

        self.f_min = QDoubleSpinBox()
        self.f_min.setRange(10, 2000)
        self.f_min.setValue(100)
        self.f_min.setSuffix(" Hz")

        self.f_max = QDoubleSpinBox()
        self.f_max.setRange(100, 5000)
        self.f_max.setValue(800)
        self.f_max.setSuffix(" Hz")

        self.hop_rate = QDoubleSpinBox()
        self.hop_rate.setRange(1, 100)
        self.hop_rate.setValue(10)
        self.hop_rate.setSuffix(" hops/s")

        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 10)
        self.duration.setValue(4)
        self.duration.setSuffix(" s")

        self.fs = QSpinBox()
        self.fs.setRange(10000, 300000)
        self.fs.setValue(100000)

        self.seed_input = QLineEdit("42")
        self.seed_input.setMaximumWidth(80)

        control_layout.addWidget(QLabel("Signal Type (per hop)"), 0, 0)
        control_layout.addWidget(self.signal_type, 0, 1)
        control_layout.addWidget(QLabel("Number of Channels"), 1, 0)
        control_layout.addWidget(self.num_channels, 1, 1)
        control_layout.addWidget(QLabel("Min Frequency"), 2, 0)
        control_layout.addWidget(self.f_min, 2, 1)
        control_layout.addWidget(QLabel("Max Frequency"), 3, 0)
        control_layout.addWidget(self.f_max, 3, 1)
        control_layout.addWidget(QLabel("Hop Rate"), 4, 0)
        control_layout.addWidget(self.hop_rate, 4, 1)
        control_layout.addWidget(QLabel("Duration"), 5, 0)
        control_layout.addWidget(self.duration, 5, 1)
        control_layout.addWidget(QLabel("Sampling Rate (fs)"), 6, 0)
        control_layout.addWidget(self.fs, 6, 1)
        control_layout.addWidget(QLabel("Random Seed"), 7, 0)
        control_layout.addWidget(self.seed_input, 7, 1)

        control_group.setLayout(control_layout)

        # ==================== BUTTONS ====================
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate Channels & Sequence")
        self.btn_simulate = QPushButton("Simulate Full FHSS")
        self.btn_live = QPushButton("▶ Start Live Hopping (with Dwell Spectrum)")
        self.btn_reset = QPushButton("Reset")

        self.btn_generate.clicked.connect(self.generate_channels_and_sequence)
        self.btn_simulate.clicked.connect(self.simulate_fhss)
        self.btn_live.clicked.connect(self.toggle_live_hopping)
        self.btn_reset.clicked.connect(self.reset_all)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_simulate)
        button_layout.addWidget(self.btn_live)
        button_layout.addWidget(self.btn_reset)

        # ==================== INFO ====================
        info_group = QGroupBox("Channel List & Hopping Sequence")
        info_layout = QHBoxLayout()
        self.lbl_channels = QLabel("Channels: (click Generate)")
        self.lbl_sequence = QLabel("Hopping Sequence: (click Generate)")
        self.lbl_channels.setWordWrap(True)
        self.lbl_sequence.setWordWrap(True)
        info_layout.addWidget(self.lbl_channels)
        info_layout.addWidget(self.lbl_sequence)
        info_group.setLayout(info_layout)

        # ==================== PLOT ====================
        self.figure = Figure(figsize=(13, 9.5))
        self.canvas = FigureCanvas(self.figure)

        # Simulation data
        self.channels = None
        self.hopping_sequence = None
        self.t = None
        self.signal = None
        self.hop_starts = None
        self.hop_ends = None
        self.hop_freqs = None
        self.hop_segments = None          # list of signal per hop (for live dwell FFT)
        self.is_live = False
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_hop_update)
        self.live_timer.setInterval(90)
        self.live_time = 0.0

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(info_group)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    # ==================== MODULATED SEGMENT GENERATOR ====================
    def _generate_modulated_segment(self, segment_t, f_hop, sig_type, fs):
        n = len(segment_t)
        if sig_type == "Sine":
            return np.sin(2 * np.pi * f_hop * segment_t)

        elif sig_type == "BPSK":
            spb = 40
            num_sym = max(1, (n + spb - 1) // spb)
            bits = np.random.choice([-1, 1], size=num_sym)
            symbols = np.repeat(bits, spb)[:n]
            return symbols * np.sin(2 * np.pi * f_hop * segment_t)

        elif sig_type == "QPSK":
            spb = 40
            num_sym = max(1, (n + spb - 1) // spb)
            symbols = np.random.randint(0, 4, size=num_sym)
            phase = symbols * (np.pi / 2)
            phases = np.repeat(phase, spb)[:n]
            return np.cos(2 * np.pi * f_hop * segment_t + phases)

        elif sig_type == "Chirp":
            # Short chirp within the hop (realistic for testing)
            f1 = f_hop * 0.95
            f2 = f_hop * 1.05
            return np.sin(2 * np.pi * (f1 + (f2 - f1) * segment_t / segment_t[-1]) * segment_t)

        elif sig_type == "RTCM-like":
            spb = 80
            num_sym = max(1, (n + spb - 1) // spb)
            packet = np.random.choice([-1, 1], size=num_sym)
            burst = np.repeat(packet, spb)[:n]
            return burst * np.sin(2 * np.pi * f_hop * segment_t)

        return np.sin(2 * np.pi * f_hop * segment_t)  # fallback

    # ==================== GENERATION ====================
    def generate_channels_and_sequence(self):
        n_ch = self.num_channels.value()
        fmin = self.f_min.value()
        fmax = self.f_max.value()

        self.channels = np.linspace(fmin, fmax, n_ch)

        ch_str = ", ".join([f"{f:.1f}" for f in self.channels[:10]])
        if n_ch > 10:
            ch_str += f", ... ({n_ch} total)"
        self.lbl_channels.setText(f"<b>Channels ({n_ch}):</b> {ch_str}")

        try:
            seed = int(self.seed_input.text())
        except:
            seed = 42
        np.random.seed(seed)

        hop_rate = self.hop_rate.value()
        duration = self.duration.value()
        num_hops = int(np.ceil(duration * hop_rate)) + 2

        hop_indices = np.random.randint(0, n_ch, size=num_hops)
        self.hopping_sequence = self.channels[hop_indices].tolist()

        seq_str = " → ".join([f"{f:.1f}" for f in self.hopping_sequence[:15]])
        if len(self.hopping_sequence) > 15:
            seq_str += " → ..."
        self.lbl_sequence.setText(f"<b>Hopping Sequence:</b> {seq_str}")

        self.simulate_fhss()

    def simulate_fhss(self):
        if self.channels is None or self.hopping_sequence is None:
            return

        fs = self.fs.value()
        duration = self.duration.value()
        hop_rate = self.hop_rate.value()
        dwell = 1.0 / hop_rate

        self.t = np.arange(0, duration, 1.0 / fs)
        n_samples = len(self.t)
        self.signal = np.zeros(n_samples)

        hop_times = np.arange(0, duration + dwell, dwell)
        self.hop_starts = hop_times[:-1]
        self.hop_ends = hop_times[1:]
        self.hop_freqs = []
        self.hop_segments = []

        sig_type = self.signal_type.currentText()

        for i in range(len(self.hop_starts)):
            start_idx = int(self.hop_starts[i] * fs)
            end_idx = min(int(self.hop_ends[i] * fs), n_samples)
            if start_idx >= n_samples:
                break

            f = self.hopping_sequence[i % len(self.hopping_sequence)]
            self.hop_freqs.append(f)

            segment_t = self.t[start_idx:end_idx]
            segment = self._generate_modulated_segment(segment_t, f, sig_type, fs)

            self.signal[start_idx:end_idx] = segment
            self.hop_segments.append(segment.copy())   # save for live dwell FFT

        self.plot_fhss_static()

    # ==================== STATIC FULL SIMULATION PLOT ====================
    def plot_fhss_static(self):
        self.figure.clear()

        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors='tab:blue', linewidth=10, alpha=0.8)
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title("FHSS Hopping Pattern – Frequency vs Time")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, self.signal, color='tab:orange', linewidth=0.7)
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Full FHSS Transmitted Signal (Time Domain)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        ax3 = self.figure.add_subplot(313)
        N = len(self.signal)
        fft_vals = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())
        db = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-12)
        ax3.plot(freqs, db, color='tab:purple', linewidth=1.1)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_title("Overall FHSS Power Spectrum")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(self.f_min.value() * 0.9, self.f_max.value() * 1.1)

        self.figure.tight_layout()
        self.canvas.draw()

    # ==================== LIVE HOPPING WITH SMALL-TIME DWELL FFT ====================
    def toggle_live_hopping(self):
        if not self.is_live:
            if self.signal is None:
                self.simulate_fhss()
            self.is_live = True
            self.btn_live.setText("⏹ Stop Live Hopping")
            self.live_time = 0.0
            self.live_timer.start()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Live Hopping (with Dwell Spectrum)")

    def live_hop_update(self):
        if not self.is_live or self.hop_freqs is None:
            return

        dwell = 1.0 / self.hop_rate.value()
        self.live_time += 0.09

        # Current hop
        hop_idx = int(self.live_time / dwell) % len(self.hop_freqs)
        current_f = self.hop_freqs[hop_idx]
        current_start = self.hop_starts[hop_idx % len(self.hop_starts)]

        # Current dwell segment (small time window)
        seg = self.hop_segments[hop_idx % len(self.hop_segments)]

        self.figure.clear()

        # 1. Frequency vs Time (with live highlight)
        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            color = 'lime' if i == hop_idx else 'tab:blue'
            lw = 12 if i == hop_idx else 8
            alpha = 1.0 if i == hop_idx else 0.65
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors=color, linewidth=lw, alpha=alpha)

        ax1.axvline(x=self.live_time, color='red', linewidth=2.5, linestyle='--')
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title(f"LIVE FHSS – Current Hop: {current_f:.1f} Hz   |   t = {self.live_time:.2f} s")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        # 2. Full time-domain with current dwell highlight
        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, self.signal, color='tab:orange', linewidth=0.8)
        start_idx = int(current_start * self.fs.value())
        end_idx = min(start_idx + len(seg), len(self.t))
        ax2.axvspan(self.t[start_idx], self.t[end_idx-1], alpha=0.35, color='lime')
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Transmitted Signal (full) – Green = Current Dwell")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        # 3. LIVE DWELL SPECTRUM (small time window FFT – exactly what you asked for)
        ax3 = self.figure.add_subplot(313)
        if len(seg) > 16:
            fft_vals = np.fft.rfft(seg)
            freqs = np.fft.rfftfreq(len(seg), d=1.0 / self.fs.value())
            magnitude = np.abs(fft_vals)
            db_values = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

            ax3.plot(freqs, db_values, color='tab:red', linewidth=1.6)
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Magnitude (dB)")
            ax3.set_title(f"Live Dwell Spectrum (current hop only) – {self.signal_type.currentText()}")
            ax3.grid(True, alpha=0.3)
            # Zoom to reasonable range around current frequency
            ax3.set_xlim(current_f * 0.7, current_f * 1.3)

        self.figure.tight_layout()
        self.canvas.draw()

        # Auto-stop at end of simulation
        if self.live_time >= self.duration.value():
            self.toggle_live_hopping()

    def reset_all(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Live Hopping (with Dwell Spectrum)")
        self.channels = None
        self.hopping_sequence = None
        self.signal = None
        self.hop_segments = None
        self.lbl_channels.setText("Channels: (click Generate)")
        self.lbl_sequence.setText("Hopping Sequence: (click Generate)")
        self.figure.clear()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FHSSSimulator()
    window.show()
    sys.exit(app.exec_())