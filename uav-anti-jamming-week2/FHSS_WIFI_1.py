# Week 2: WiFi 2.4 GHz FHSS Simulator (Realistic WiFi Channels)
# Features added:
#   • Realistic WiFi 2.4 GHz band (standard channel centers 2412–2472 MHz when ≤13 channels)
#   • Full hopping-sequence FFT shown at once with FIXED x-axis (no zooming)
#   • Checkboxes + levels for Noise and Jamming – work live during hopping
#   • Live mode shows impairments in both time-domain waveform and full spectrum
#   • Clean professional layout with WiFi-specific labels

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QLineEdit, QCheckBox
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FHSSSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 2: WiFi 2.4 GHz FHSS Simulator – Realistic Channels + Live Noise/Jamming")
        self.resize(1420, 960)

        layout = QVBoxLayout()

        # ==================== CONTROLS ====================
        control_group = QGroupBox("FHSS Parameters (WiFi 2.4 GHz Band)")
        control_layout = QGridLayout()

        self.signal_type = QComboBox()
        self.signal_type.addItems(["Sine", "BPSK", "QPSK", "Chirp", "RTCM-like"])

        self.num_channels = QSpinBox()
        self.num_channels.setRange(4, 64)
        self.num_channels.setValue(13)

        self.f_min = QDoubleSpinBox()
        self.f_min.setRange(2400, 2480)
        self.f_min.setValue(2400)
        self.f_min.setSuffix(" MHz")

        self.f_max = QDoubleSpinBox()
        self.f_max.setRange(2420, 2500)
        self.f_max.setValue(2485)
        self.f_max.setSuffix(" MHz")

        self.hop_rate = QDoubleSpinBox()
        self.hop_rate.setRange(1, 200)
        self.hop_rate.setValue(15)
        self.hop_rate.setSuffix(" hops/s")

        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 10)
        self.duration.setValue(4)
        self.duration.setSuffix(" s")

        self.fs = QSpinBox()
        self.fs.setRange(50000, 500000)
        self.fs.setValue(200000)

        self.seed_input = QLineEdit("42")
        self.seed_input.setMaximumWidth(80)

        # Noise & Jamming controls
        self.noise_level = QDoubleSpinBox()
        self.noise_level.setRange(0, 1.0)
        self.noise_level.setValue(0.08)
        self.noise_level.setSingleStep(0.01)
        self.chk_noise = QCheckBox("Enable Noise")

        self.jammer_level = QDoubleSpinBox()
        self.jammer_level.setRange(0, 2.0)
        self.jammer_level.setValue(0.6)
        self.jammer_level.setSingleStep(0.05)
        self.chk_jammer = QCheckBox("Enable Jamming")

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
        control_layout.addWidget(QLabel("Sampling Rate"), 6, 0)
        control_layout.addWidget(self.fs, 6, 1)
        control_layout.addWidget(QLabel("Random Seed"), 7, 0)
        control_layout.addWidget(self.seed_input, 7, 1)

        control_layout.addWidget(QLabel("Noise Level"), 8, 0)
        control_layout.addWidget(self.noise_level, 8, 1)
        control_layout.addWidget(self.chk_noise, 8, 2)

        control_layout.addWidget(QLabel("Jammer Level"), 9, 0)
        control_layout.addWidget(self.jammer_level, 9, 1)
        control_layout.addWidget(self.chk_jammer, 9, 2)

        control_group.setLayout(control_layout)

        # ==================== BUTTONS ====================
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate WiFi Channels & Sequence")
        self.btn_simulate = QPushButton("Simulate Full FHSS")
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
        self.figure = Figure(figsize=(13.5, 9.8))
        self.canvas = FigureCanvas(self.figure)

        # Data storage
        self.channels = None
        self.hopping_sequence = None
        self.t = None
        self.base_signal = None          # clean FHSS signal
        self.hop_starts = None
        self.hop_ends = None
        self.hop_freqs = None
        self.is_live = False
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_hop_update)
        self.live_timer.setInterval(85)
        self.live_time = 0.0

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(info_group)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    # ==================== MODULATED SEGMENT GENERATOR ====================
    def _generate_modulated_segment(self, segment_t, f_hop, sig_type):
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
            f1 = f_hop * 0.95
            f2 = f_hop * 1.05
            return np.sin(2 * np.pi * (f1 + (f2 - f1) * (segment_t - segment_t[0]) / (segment_t[-1] - segment_t[0])) * segment_t)

        elif sig_type == "RTCM-like":
            spb = 80
            num_sym = max(1, (n + spb - 1) // spb)
            packet = np.random.choice([-1, 1], size=num_sym)
            burst = np.repeat(packet, spb)[:n]
            return burst * np.sin(2 * np.pi * f_hop * segment_t)

        return np.sin(2 * np.pi * f_hop * segment_t)

    # ==================== GENERATION ====================
    def generate_channels_and_sequence(self):
        n_ch = self.num_channels.value()

        # Realistic WiFi 2.4 GHz channels when ≤13
        if n_ch <= 13:
            base = np.array([2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467, 2472])
            self.channels = base[:n_ch].astype(float)
        else:
            self.channels = np.linspace(self.f_min.value(), self.f_max.value(), n_ch)

        # Display channels
        ch_str = ", ".join([f"{f:.1f}" for f in self.channels[:12]])
        if len(self.channels) > 12:
            ch_str += f", ... ({len(self.channels)} total)"
        self.lbl_channels.setText(f"<b>WiFi Channels ({n_ch}):</b> {ch_str} MHz")

        # Hopping sequence
        try:
            seed = int(self.seed_input.text())
        except:
            seed = 42
        np.random.seed(seed)

        hop_rate = self.hop_rate.value()
        duration = self.duration.value()
        num_hops = int(np.ceil(duration * hop_rate)) + 3

        hop_indices = np.random.randint(0, len(self.channels), size=num_hops)
        self.hopping_sequence = self.channels[hop_indices].tolist()

        seq_str = " → ".join([f"{f:.1f}" for f in self.hopping_sequence[:18]])
        if len(self.hopping_sequence) > 18:
            seq_str += " → ..."
        self.lbl_sequence.setText(f"<b>Hopping Sequence:</b> {seq_str} MHz")

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
        self.base_signal = np.zeros(n_samples)

        hop_times = np.arange(0, duration + dwell, dwell)
        self.hop_starts = hop_times[:-1]
        self.hop_ends = hop_times[1:]
        self.hop_freqs = []

        sig_type = self.signal_type.currentText()

        for i in range(len(self.hop_starts)):
            start_idx = int(self.hop_starts[i] * fs)
            end_idx = min(int(self.hop_ends[i] * fs), n_samples)
            if start_idx >= n_samples:
                break

            f = self.hopping_sequence[i % len(self.hopping_sequence)]
            self.hop_freqs.append(f)

            segment_t = self.t[start_idx:end_idx]
            segment = self._generate_modulated_segment(segment_t, f, sig_type)
            self.base_signal[start_idx:end_idx] = segment

        self.plot_fhss_static()

    # ==================== STATIC PLOT (clean FHSS) ====================
    def plot_fhss_static(self):
        self.figure.clear()

        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors='tab:blue', linewidth=10, alpha=0.85)
        ax1.set_ylabel("Frequency (MHz)")
        ax1.set_title("FHSS Hopping Pattern – Frequency vs Time (WiFi 2.4 GHz)")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, self.base_signal, color='tab:orange', linewidth=0.8)
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Clean FHSS Transmitted Signal (Time Domain)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        ax3 = self.figure.add_subplot(313)
        N = len(self.base_signal)
        fft_vals = np.fft.rfft(self.base_signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())
        db = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-12)
        ax3.plot(freqs, db, color='tab:purple', linewidth=1.2)
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_title("Full FHSS Power Spectrum (Whole Hopping Sequence)")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(self.f_min.value() * 0.95, self.f_max.value() * 1.05)

        self.figure.tight_layout()
        self.canvas.draw()

    # ==================== LIVE HOPPING ====================
    def toggle_live_hopping(self):
        if not self.is_live:
            if self.base_signal is None:
                self.simulate_fhss()
            self.is_live = True
            self.btn_live.setText("⏹ Stop Live Hopping")
            self.live_time = 0.0
            self.live_timer.start()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Live Hopping")

    def live_hop_update(self):
        if not self.is_live or self.hop_freqs is None:
            return

        dwell = 1.0 / self.hop_rate.value()
        self.live_time += 0.085

        hop_idx = int(self.live_time / dwell) % len(self.hop_freqs)
        current_f = self.hop_freqs[hop_idx]
        current_start = self.hop_starts[hop_idx % len(self.hop_starts)]

        # === Apply realistic impairments if checkboxes are enabled ===
        effective_signal = self.base_signal.copy()

        if self.chk_noise.isChecked():
            noise = np.random.normal(0, self.noise_level.value(), len(self.base_signal))
            effective_signal += noise

        if self.chk_jammer.isChecked():
            jammer_freq = 2440.0                     # fixed narrowband jammer inside WiFi band
            jammer = self.jammer_level.value() * np.sin(2 * np.pi * jammer_freq * self.t)
            effective_signal += jammer

        # === Plot update ===
        self.figure.clear()

        # 1. Hopping pattern with live marker
        ax1 = self.figure.add_subplot(311)
        for i, f in enumerate(self.hop_freqs):
            color = 'lime' if i == hop_idx else 'tab:blue'
            lw = 12 if i == hop_idx else 8
            alpha = 1.0 if i == hop_idx else 0.65
            ax1.hlines(f, self.hop_starts[i], self.hop_ends[i],
                       colors=color, linewidth=lw, alpha=alpha)

        ax1.axvline(x=self.live_time, color='red', linewidth=2.5, linestyle='--')
        ax1.set_ylabel("Frequency (MHz)")
        ax1.set_title(f"LIVE FHSS (WiFi 2.4 GHz) – Current Hop: {current_f:.1f} MHz   |   t = {self.live_time:.2f} s")
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, self.duration.value())

        # 2. Time-domain waveform (with impairments)
        ax2 = self.figure.add_subplot(312)
        ax2.plot(self.t, effective_signal, color='tab:orange', linewidth=0.9)
        start_idx = int(current_start * self.fs.value())
        end_idx = min(start_idx + int(dwell * self.fs.value()), len(self.t))
        ax2.axvspan(self.t[start_idx], self.t[end_idx-1], alpha=0.3, color='lime')
        ax2.set_ylabel("Amplitude")
        ax2.set_title("Received Signal (with Noise/Jamming if enabled)")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.duration.value())

        # 3. Full FHSS spectrum (whole hopping sequence, FIXED x-axis)
        ax3 = self.figure.add_subplot(313)
        N = len(effective_signal)
        fft_vals = np.fft.rfft(effective_signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())
        magnitude = np.abs(fft_vals)
        db_values = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

        ax3.plot(freqs, db_values, color='tab:red', linewidth=1.3)
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.set_title("Full FHSS Spectrum (Whole Sequence) – Fixed WiFi Band View")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(self.f_min.value() * 0.95, self.f_max.value() * 1.05)

        self.figure.tight_layout()
        self.canvas.draw()

        if self.live_time >= self.duration.value():
            self.toggle_live_hopping()

    def reset_all(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Live Hopping")
        self.channels = None
        self.hopping_sequence = None
        self.base_signal = None
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