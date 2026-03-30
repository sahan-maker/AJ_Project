# Week 1: Multi-Signal RF Simulator - Real-Time with Checkboxes
# Features:
#   • Checkboxes to toggle Noise and Jamming ON/OFF during live monitoring
#   • Real-time variations still active when enabled
#   • Clean, professional dual-plot layout (Time + dB Spectrum)

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout, QCheckBox
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RFSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Signal RF Simulator - Real-Time Monitoring")
        self.resize(1280, 860)

        layout = QVBoxLayout()

        # ------------------ Controls ------------------
        control_group = QGroupBox("Simulation Parameters")
        control_layout = QGridLayout()

        self.signal_type = QComboBox()
        self.signal_type.addItems(["Sine", "BPSK", "QPSK", "Chirp", "RTCM-like"])

        self.freq = QDoubleSpinBox()
        self.freq.setRange(1, 5000)
        self.freq.setValue(100)
        self.freq.setSuffix(" Hz")

        self.fs = QSpinBox()
        self.fs.setRange(1000, 100000)
        self.fs.setValue(10000)

        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 5)
        self.duration.setValue(1.0)
        self.duration.setSuffix(" s")

        self.noise_level = QDoubleSpinBox()
        self.noise_level.setRange(0, 2)
        self.noise_level.setSingleStep(0.05)
        self.noise_level.setValue(0.12)

        self.jammer_level = QDoubleSpinBox()
        self.jammer_level.setRange(0, 3)
        self.jammer_level.setSingleStep(0.1)
        self.jammer_level.setValue(0.6)

        # Checkboxes for real-time control
        self.chk_noise = QCheckBox("Enable Noise")
        self.chk_noise.setChecked(True)
        self.chk_jammer = QCheckBox("Enable Jamming")
        self.chk_jammer.setChecked(True)

        control_layout.addWidget(QLabel("Signal Type"), 0, 0)
        control_layout.addWidget(self.signal_type, 0, 1)

        control_layout.addWidget(QLabel("Carrier Frequency"), 1, 0)
        control_layout.addWidget(self.freq, 1, 1)

        control_layout.addWidget(QLabel("Sampling Rate"), 2, 0)
        control_layout.addWidget(self.fs, 2, 1)

        control_layout.addWidget(QLabel("Time Window"), 3, 0)
        control_layout.addWidget(self.duration, 3, 1)

        control_layout.addWidget(QLabel("Noise Level"), 4, 0)
        control_layout.addWidget(self.noise_level, 4, 1)
        control_layout.addWidget(self.chk_noise, 4, 2)

        control_layout.addWidget(QLabel("Jammer Level"), 5, 0)
        control_layout.addWidget(self.jammer_level, 5, 1)
        control_layout.addWidget(self.chk_jammer, 5, 2)

        control_group.setLayout(control_layout)

        # ------------------ Buttons ------------------
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate Static Signal")
        self.btn_live = QPushButton("▶ Start Real-Time Monitor")
        self.btn_reset = QPushButton("Reset")

        self.btn_generate.clicked.connect(self.generate_signal)
        self.btn_live.clicked.connect(self.toggle_live_monitor)
        self.btn_reset.clicked.connect(self.reset_signal)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_live)
        button_layout.addWidget(self.btn_reset)

        # ------------------ Plot ------------------
        self.figure = Figure(figsize=(11.5, 8.5))
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Simulation variables
        self.t = None
        self.signal = None
        self.is_live = False
        self.sim_time = 0.0
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_update)
        self.live_timer.setInterval(200)  # Faster update for smoother feel

    # ------------------ Signal Generation ------------------
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
            spb = 50
            num_sym = (n + spb - 1) // spb
            bits = np.random.choice([-1, 1], size=num_sym)
            symbols = np.repeat(bits, spb)[:n]
            self.signal = symbols * np.sin(2 * np.pi * f * self.t)

        elif sig_type == "QPSK":
            spb = 50
            num_sym = (n + spb - 1) // spb
            symbols = np.random.randint(0, 4, size=num_sym)
            phase = symbols * (np.pi / 2)
            phases = np.repeat(phase, spb)[:n]
            self.signal = np.cos(2 * np.pi * f * self.t + phases)

        elif sig_type == "Chirp":
            f1 = f
            f2 = f * 5
            self.signal = np.sin(2 * np.pi * (f1 + (f2 - f1) * self.t / self.t[-1]) * self.t)

        elif sig_type == "RTCM-like":
            spb = 200
            num_sym = (n + spb - 1) // spb
            packet = np.random.choice([-1, 1], size=num_sym)
            burst = np.repeat(packet, spb)[:n]
            self.signal = burst * np.sin(2 * np.pi * f * self.t)

        self.plot_time_domain("Generated Signal")

    # ------------------ Real-Time Live Update (with checkbox control) ------------------
    def toggle_live_monitor(self):
        if not self.is_live:
            if self.signal is None:
                self.generate_signal()
            self.is_live = True
            self.btn_live.setText("⏹ Stop Real-Time Monitor")
            self.sim_time = 0.0
            self.live_timer.start()
            self.live_update()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Real-Time Monitor")

    def live_update(self):
        if not self.is_live:
            return

        self.sim_time += 0.2

        # Re-generate clean base signal
        self.generate_signal()

        # === Apply Noise if checkbox is ON ===
        if self.chk_noise.isChecked():
            noise_var = self.noise_level.value() * (0.7 + 0.6 * np.sin(2 * np.pi * 1.1 * self.sim_time))
            noise = np.random.normal(0, noise_var, len(self.signal))
            self.signal += noise

        # === Apply Jamming if checkbox is ON ===
        if self.chk_jammer.isChecked():
            # Pulsing amplitude + slight frequency drift
            jam_amp = self.jammer_level.value() * (0.6 + 0.8 * np.sin(2 * np.pi * 0.9 * self.sim_time))
            jam_freq = self.freq.value() * (1.8 + 0.06 * np.sin(2 * np.pi * 0.4 * self.sim_time))
            jammer = jam_amp * np.sin(2 * np.pi * jam_freq * self.t)
            self.signal += jammer

        # Gentle overall amplitude modulation (realistic channel effect)
        amp_mod = 0.92 + 0.12 * np.sin(2 * np.pi * 0.35 * self.sim_time)
        self.signal *= amp_mod

        # Update plots
        status = f"Live RF Monitor  |  t = {self.sim_time:.1f}s  |  Noise: {'ON' if self.chk_noise.isChecked() else 'OFF'}  |  Jammer: {'ON' if self.chk_jammer.isChecked() else 'OFF'}"
        self.update_rf_plots(status)

    # ------------------ Plotting Functions ------------------
    def plot_time_domain(self, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.t, self.signal, color='tab:blue')
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_rf_plots(self, title):
        """Dual plot: Time domain + dB Spectrum (one-sided)"""
        self.figure.clear()

        # Time Domain
        ax_time = self.figure.add_subplot(211)
        ax_time.plot(self.t, self.signal, color='tab:blue', linewidth=1.1)
        ax_time.set_title(title)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        ax_time.grid(True, alpha=0.3)

        # Frequency Domain - dB Scale (Positive frequencies only)
        ax_freq = self.figure.add_subplot(212)
        N = len(self.signal)
        fft_vals = np.fft.rfft(self.signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs.value())

        magnitude = np.abs(fft_vals)
        db_values = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)

        ax_freq.plot(freqs, db_values, color='tab:red', linewidth=1.3)
        ax_freq.set_title("Real-Time Frequency Spectrum (dB)")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Magnitude (dB)")
        ax_freq.grid(True, alpha=0.3)
        ax_freq.set_xlim(0, self.fs.value() / 2)

        self.figure.tight_layout()
        self.canvas.draw()

    def reset_signal(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Real-Time Monitor")
        self.signal = None
        self.t = None
        self.sim_time = 0.0
        self.figure.clear()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RFSimulator()
    window.show()
    sys.exit(app.exec_())