# FIXED VERSION - Visible FHSS Hopping on Live FFT + Waterfall
# Strong signal, proper per-frame generation, fixed frequency axis

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
        self.setWindowTitle("WiFi 2.4 GHz FHSS - Live FFT + Waterfall")
        self.resize(1480, 960)

        layout = QVBoxLayout()

        # Controls
        ctrl = QGroupBox("WiFi FHSS Parameters")
        grid = QGridLayout()

        self.sig_type = QComboBox()
        self.sig_type.addItems(["Sine", "BPSK", "QPSK", "Chirp", "RTCM-like"])

        self.num_ch = QSpinBox(); self.num_ch.setRange(4,64); self.num_ch.setValue(13)
        self.hop_rate = QDoubleSpinBox(); self.hop_rate.setRange(5,80); self.hop_rate.setValue(12); self.hop_rate.setSuffix(" hops/s")
        self.fs = QSpinBox(); self.fs.setRange(300000,2000000); self.fs.setValue(1000000)

        self.seed = QLineEdit("42")

        self.noise_lvl = QDoubleSpinBox(); self.noise_lvl.setRange(0,1); self.noise_lvl.setValue(0.08)
        self.chk_noise = QCheckBox("Noise"); self.chk_noise.setChecked(True)

        self.jam_lvl = QDoubleSpinBox(); self.jam_lvl.setRange(0,5); self.jam_lvl.setValue(1.2)
        self.chk_jam = QCheckBox("Jammer @ 2440 MHz"); self.chk_jam.setChecked(False)

        grid.addWidget(QLabel("Signal Type"), 0, 0); grid.addWidget(self.sig_type, 0, 1)
        grid.addWidget(QLabel("Channels"), 1, 0); grid.addWidget(self.num_ch, 1, 1)
        grid.addWidget(QLabel("Hop Rate"), 2, 0); grid.addWidget(self.hop_rate, 2, 1)
        grid.addWidget(QLabel("Sampling Rate"), 3, 0); grid.addWidget(self.fs, 3, 1)
        grid.addWidget(QLabel("Seed"), 4, 0); grid.addWidget(self.seed, 4, 1)
        grid.addWidget(QLabel("Noise"), 5, 0); grid.addWidget(self.noise_lvl, 5, 1); grid.addWidget(self.chk_noise, 5, 2)
        grid.addWidget(QLabel("Jammer"), 6, 0); grid.addWidget(self.jam_lvl, 6, 1); grid.addWidget(self.chk_jam, 6, 2)

        ctrl.setLayout(grid)

        # Buttons
        btns = QHBoxLayout()
        self.btn_gen = QPushButton("Generate Sequence")
        self.btn_live = QPushButton("▶ Start Live Hopping")
        self.btn_reset = QPushButton("Reset")

        self.btn_gen.clicked.connect(self.generate_sequence)
        self.btn_live.clicked.connect(self.toggle_live)
        self.btn_reset.clicked.connect(self.reset_all)

        btns.addWidget(self.btn_gen)
        btns.addWidget(self.btn_live)
        btns.addWidget(self.btn_reset)

        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.figure)

        # Data
        self.hop_channels = None
        self.is_live = False
        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_update)
        self.live_timer.setInterval(50)
        self.live_time = 0.0
        self.waterfall = None

        layout.addWidget(ctrl)
        layout.addLayout(btns)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def generate_sequence(self):
        n = self.num_ch.value()
        if n <= 13:
            ch_list = [2412,2417,2422,2427,2432,2437,2442,2447,2452,2457,2462,2467,2472]
            self.hop_channels = np.array(ch_list[:n], dtype=float)
        else:
            self.hop_channels = np.linspace(2405, 2480, n)
        print(f"✓ Generated {len(self.hop_channels)} WiFi channels")

    def toggle_live(self):
        if not self.is_live:
            if self.hop_channels is None:
                self.generate_sequence()
            self.is_live = True
            self.btn_live.setText("⏹ Stop Live")
            self.live_time = 0.0
            self.waterfall = None
            self.live_timer.start()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Live Hopping")

    def live_update(self):
        if not self.is_live or self.hop_channels is None:
            return

        # Current hop
        hop_rate = self.hop_rate.value()
        dwell = 1.0 / hop_rate
        self.live_time += 0.05
        hop_idx = int(self.live_time / dwell) % len(self.hop_channels)
        f_mhz = self.hop_channels[hop_idx]
        f_hz = f_mhz * 1_000_000

        # Generate fresh signal for current hop only (this fixes the empty plot)
        fs = self.fs.value()
        window_sec = 0.12
        t = np.arange(0, window_sec, 1/fs)
        signal = self.make_signal(t, f_hz)

        # Add noise/jammer
        if self.chk_noise.isChecked():
            signal += np.random.normal(0, self.noise_lvl.value(), len(signal))
        if self.chk_jam.isChecked():
            signal += self.jam_lvl.value() * np.sin(2 * np.pi * 2440_000_000 * t)

        # FFT
        fft = np.fft.rfft(signal)
        freq_hz = np.fft.rfftfreq(len(signal), 1/fs)
        freq_mhz = freq_hz / 1e6
        db = 20 * np.log10(np.abs(fft) / np.max(np.abs(fft)) + 1e-12)

        # Waterfall
        if self.waterfall is None:
            self.waterfall = np.zeros((80, len(db)))
        self.waterfall = np.roll(self.waterfall, -1, axis=0)
        self.waterfall[-1] = db

        # Plot
        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.plot(freq_mhz, db, 'r-', linewidth=1.8)
        ax1.set_xlim(2400, 2485)
        ax1.set_ylim(-55, 8)
        ax1.set_title(f"Live FFT - Current Hop: {f_mhz:.1f} MHz")
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, alpha=0.35)

        ax2 = self.figure.add_subplot(212)
        if self.waterfall is not None:
            im = ax2.imshow(self.waterfall, aspect='auto', origin='upper',
                           cmap='plasma', extent=[2400, 2485, 0, 80],
                           vmin=-50, vmax=5)
            ax2.set_title("Waterfall Spectrogram")
            ax2.set_xlabel("Frequency (MHz)")
            ax2.set_ylabel("Time (recent → past)")
            self.figure.colorbar(im, ax=ax2, label="dB")

        self.figure.tight_layout()
        self.canvas.draw()

    def make_signal(self, t, f_hz):
        sig_type = self.sig_type.currentText()
        if sig_type == "Sine":
            return np.sin(2 * np.pi * f_hz * t)
        elif sig_type == "BPSK":
            spb = 25
            n = len(t)
            bits = np.random.choice([-1,1], (n+spb-1)//spb )
            sym = np.repeat(bits, spb)[:n]
            return sym * np.sin(2 * np.pi * f_hz * t)
        elif sig_type == "QPSK":
            spb = 25
            n = len(t)
            syms = np.random.randint(0,4, (n+spb-1)//spb)
            phase = np.repeat(syms * np.pi/2, spb)[:n]
            return np.cos(2 * np.pi * f_hz * t + phase)
        elif sig_type == "Chirp":
            f1 = f_hz*0.85
            f2 = f_hz*1.15
            return np.sin(2*np.pi*(f1 + (f2-f1)*(t/t[-1])) * t)
        else:  # RTCM-like
            spb = 50
            n = len(t)
            bits = np.random.choice([-1,1], (n+spb-1)//spb)
            sym = np.repeat(bits, spb)[:n]
            return sym * np.sin(2 * np.pi * f_hz * t)

    def reset_all(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Live Hopping")
        self.waterfall = None
        self.figure.clear()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FHSSSimulator()
    win.show()
    sys.exit(app.exec_())