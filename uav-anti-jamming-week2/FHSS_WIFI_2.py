# FHSS Simulator - Correct FFT/Waterfall Display
# Fix: composite display-domain spectrum (like a real spectrum analyser sweep)
# Each frame synthesises power across the full 2400-2480 MHz display band.

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

WIFI_CHANNELS_MHZ = [2412,2417,2422,2427,2432,2437,2442,2447,2452,2457,2462,2467,2472]
BAND_START_MHZ    = 2400.0
BAND_END_MHZ      = 2480.0
BAND_SPAN_MHZ     = BAND_END_MHZ - BAND_START_MHZ   # 80 MHz
JAMMER_FREQ_MHZ   = 2440.0
N_BINS            = 1024     # display resolution
WATERFALL_ROWS    = 80


def mhz_to_bin(f_mhz):
    """Convert a frequency in MHz to the nearest display bin index."""
    return int(np.clip(
        (f_mhz - BAND_START_MHZ) / BAND_SPAN_MHZ * N_BINS,
        0, N_BINS - 1
    ))


def build_spectrum(active_mhz, noise_std, jam_amp, jam_on,
                   sig_type, rng: np.random.Generator):
    """
    Build one frame of the display-domain power spectrum (linear scale),
    then return it as dB.  Models what a swept-spectrum analyser would show.
    """
    BIN_SPAN = BAND_SPAN_MHZ / N_BINS   # ~0.078 MHz per bin
    x = np.arange(N_BINS, dtype=float)
    power = np.zeros(N_BINS)

    # ── Noise floor ───────────────────────────────────────────────
    if noise_std > 0:
        power += noise_std ** 2 * rng.exponential(1.0, N_BINS)

    # ── Active hop channel ────────────────────────────────────────
    cb = mhz_to_bin(active_mhz)

    if sig_type == "Sine":
        # Sharp single tone
        sigma = max(1.0, 1.5 / BIN_SPAN)
        tone  = 10.0 * np.exp(-0.5 * ((x - cb) / sigma) ** 2)

    elif sig_type == "BPSK":
        # Sinc-squared envelope ≈ BPSK spectrum (null-to-null ~2× symbol rate)
        # Assume 1 Msym/s → ±1 MHz BW → ±(1/BIN_SPAN) bins
        bw_bins = max(2.0, 1.0 / BIN_SPAN)
        u = (x - cb) / bw_bins * np.pi
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(u == 0, 1.0, np.sin(u) / u)
        tone = 10.0 * sinc ** 2

    elif sig_type == "QPSK":
        # QPSK: same main lobe as BPSK but half the BW for same throughput
        bw_bins = max(1.5, 0.5 / BIN_SPAN)
        u = (x - cb) / bw_bins * np.pi
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(u == 0, 1.0, np.sin(u) / u)
        tone = 10.0 * sinc ** 2

    elif sig_type == "Chirp":
        # Flat top across ±chirp_bw/2
        chirp_bw_bins = max(2.0, 10.0 / BIN_SPAN)
        tone = 10.0 * (np.abs(x - cb) <= chirp_bw_bins / 2).astype(float)
        # taper edges slightly
        tone = np.convolve(tone, np.ones(3) / 3, mode='same')

    else:  # RTCM-like NRZ
        bw_bins = max(1.5, 0.5 / BIN_SPAN)
        u = (x - cb) / bw_bins * np.pi
        with np.errstate(divide='ignore', invalid='ignore'):
            sinc = np.where(u == 0, 1.0, np.sin(u) / u)
        tone = 10.0 * sinc ** 2

    power += tone

    # ── Jammer ────────────────────────────────────────────────────
    if jam_on and jam_amp > 0:
        jb    = mhz_to_bin(JAMMER_FREQ_MHZ)
        sigma = max(1.5, 2.0 / BIN_SPAN)
        power += jam_amp ** 2 * np.exp(-0.5 * ((x - jb) / sigma) ** 2)

    # ── Convert to dB ─────────────────────────────────────────────
    peak = power.max() if power.max() > 0 else 1.0
    db   = 10.0 * np.log10(power / peak + 1e-12)
    return db


class FHSSSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WiFi 2.4 GHz FHSS – Live FFT + Waterfall")
        self.resize(1480, 960)

        self._rng = np.random.default_rng(42)

        # ── Controls ─────────────────────────────────────────────
        ctrl = QGroupBox("WiFi FHSS Parameters")
        grid = QGridLayout()

        self.sig_type  = QComboBox()
        self.sig_type.addItems(["Sine", "BPSK", "QPSK", "Chirp", "RTCM-like"])

        self.num_ch    = QSpinBox();       self.num_ch.setRange(4, 13); self.num_ch.setValue(13)
        self.hop_rate  = QDoubleSpinBox(); self.hop_rate.setRange(1, 80); self.hop_rate.setValue(8); self.hop_rate.setSuffix(" hops/s")
        self.seed      = QLineEdit("42")

        self.noise_lvl = QDoubleSpinBox(); self.noise_lvl.setRange(0, 2); self.noise_lvl.setValue(0.15); self.noise_lvl.setSingleStep(0.05)
        self.chk_noise = QCheckBox("Noise"); self.chk_noise.setChecked(True)

        self.jam_lvl   = QDoubleSpinBox(); self.jam_lvl.setRange(0, 5); self.jam_lvl.setValue(2.0)
        self.chk_jam   = QCheckBox(f"Jammer @ {JAMMER_FREQ_MHZ:.0f} MHz"); self.chk_jam.setChecked(False)

        rows = [
            ("Signal Type",   self.sig_type,  None),
            ("Channels",      self.num_ch,    None),
            ("Hop Rate",      self.hop_rate,  None),
            ("Seed",          self.seed,      None),
            ("Noise Level",   self.noise_lvl, self.chk_noise),
            ("Jammer Level",  self.jam_lvl,   self.chk_jam),
        ]
        for r, (lbl, w, chk) in enumerate(rows):
            grid.addWidget(QLabel(lbl), r, 0)
            grid.addWidget(w, r, 1)
            if chk:
                grid.addWidget(chk, r, 2)
        ctrl.setLayout(grid)

        # ── Buttons ───────────────────────────────────────────────
        btns = QHBoxLayout()
        self.btn_gen   = QPushButton("Generate Sequence")
        self.btn_live  = QPushButton("▶ Start Live Hopping")
        self.btn_reset = QPushButton("Reset")
        self.btn_gen.clicked.connect(self.generate_sequence)
        self.btn_live.clicked.connect(self.toggle_live)
        self.btn_reset.clicked.connect(self.reset_all)
        for b in (self.btn_gen, self.btn_live, self.btn_reset):
            btns.addWidget(b)

        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(ctrl)
        layout.addLayout(btns)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # ── State ─────────────────────────────────────────────────
        self.hop_channels: np.ndarray | None = None
        self.is_live    = False
        self.live_time  = 0.0
        self.waterfall  = np.full((WATERFALL_ROWS, N_BINS), -60.0)

        self.freq_axis  = np.linspace(BAND_START_MHZ, BAND_END_MHZ, N_BINS)

        self.live_timer = QTimer()
        self.live_timer.timeout.connect(self.live_update)
        self.live_timer.setInterval(50)   # 20 fps

    def generate_sequence(self):
        n = self.num_ch.value()
        seed_val = int(self.seed.text()) if self.seed.text().isdigit() else 42
        rng = np.random.default_rng(seed_val)
        ch  = np.array(WIFI_CHANNELS_MHZ[:n])
        self.hop_channels = rng.permutation(ch)
        print(f"✓ {len(self.hop_channels)} channels: {self.hop_channels.tolist()}")

    def toggle_live(self):
        if not self.is_live:
            if self.hop_channels is None:
                self.generate_sequence()
            self.is_live  = True
            self.live_time = 0.0
            self.waterfall = np.full((WATERFALL_ROWS, N_BINS), -60.0)
            self.btn_live.setText("⏹ Stop Live")
            self.live_timer.start()
        else:
            self.is_live = False
            self.live_timer.stop()
            self.btn_live.setText("▶ Start Live Hopping")

    def reset_all(self):
        self.is_live = False
        self.live_timer.stop()
        self.btn_live.setText("▶ Start Live Hopping")
        self.waterfall = np.full((WATERFALL_ROWS, N_BINS), -60.0)
        self.figure.clear()
        self.canvas.draw()

    def live_update(self):
        if not self.is_live or self.hop_channels is None:
            return

        dwell = 1.0 / self.hop_rate.value()
        self.live_time += 0.05
        hop_idx = int(self.live_time / dwell) % len(self.hop_channels)
        f_mhz   = self.hop_channels[hop_idx]

        db = build_spectrum(
            active_mhz = f_mhz,
            noise_std  = self.noise_lvl.value() if self.chk_noise.isChecked() else 0.0,
            jam_amp    = self.jam_lvl.value(),
            jam_on     = self.chk_jam.isChecked(),
            sig_type   = self.sig_type.currentText(),
            rng        = self._rng,
        )

        # Update waterfall
        self.waterfall = np.roll(self.waterfall, -1, axis=0)
        self.waterfall[-1] = db

        # ── Plot ──────────────────────────────────────────────────
        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        ax1.plot(self.freq_axis, db, color='tomato', linewidth=1.5)
        ax1.axvline(f_mhz, color='yellow', linewidth=1.2, linestyle='--', alpha=0.85,
                    label=f"Hop: {f_mhz:.0f} MHz")
        # Mark all channels lightly
        for ch in self.hop_channels:
            ax1.axvline(ch, color='gray', linewidth=0.4, alpha=0.4)
        if self.chk_jam.isChecked():
            ax1.axvline(JAMMER_FREQ_MHZ, color='cyan', linewidth=1.2,
                        linestyle=':', alpha=0.9, label=f"Jammer: {JAMMER_FREQ_MHZ:.0f} MHz")
        ax1.set_xlim(BAND_START_MHZ, BAND_END_MHZ)
        ax1.set_ylim(-65, 3)
        ax1.set_title(
            f"Spectrum  –  Hop {hop_idx + 1}/{len(self.hop_channels)}: "
            f"{f_mhz:.0f} MHz  |  {self.sig_type.currentText()}"
        )
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Power (dB)")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(212)
        im = ax2.imshow(
            self.waterfall, aspect='auto', origin='upper',
            cmap='plasma',
            extent=[BAND_START_MHZ, BAND_END_MHZ, WATERFALL_ROWS, 0],
            vmin=-60, vmax=0,
        )
        ax2.set_xlim(BAND_START_MHZ, BAND_END_MHZ)
        ax2.set_title("Waterfall  (newest row at top)")
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Frames ago")
        self.figure.colorbar(im, ax=ax2, label="dB", fraction=0.03)

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FHSSSimulator()
    win.show()
    sys.exit(app.exec_())