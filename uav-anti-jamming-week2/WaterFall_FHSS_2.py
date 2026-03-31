"""
FHSS Simulation v3 — Live FFT + Waterfall + Modulation Selector
+ Noise Level Control + Jamming/Interference + Message Console
================================================================
Requires: PyQt5, matplotlib, numpy, scipy
Install:  pip install PyQt5 matplotlib numpy scipy
Run:      python fhss_simulation_v3.py
"""

import sys
import numpy as np
import random
import time
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox,
    QSplitter, QFrame, QTabWidget, QScrollArea, QTextEdit, QCheckBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor, QTextCharFormat

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
ACCENT  = "#00d4ff"
ACCENT2 = "#ff6b35"
GREEN   = "#39d353"
YELLOW  = "#e3b341"
MUTED   = "#8b949e"
TEXT    = "#e6edf3"
HOVER   = "#1f2937"
RED     = "#f85149"
PURPLE  = "#bc8cff"
ORANGE  = "#ff7b3b"

CHANNEL_COLORS = [
    "#00d4ff","#ff6b35","#39d353","#f0e68c",
    "#da70d6","#87ceeb","#ff69b4","#98fb98",
    "#ffa500","#7b68ee","#20b2aa","#ff4500",
    "#9370db","#3cb371","#b8860b","#4682b4",
]

# Custom waterfall colormap
_wf_cmap = LinearSegmentedColormap.from_list(
    "waterfall",
    [(0,"#000000"),(0.25,"#0a2a6e"),(0.5,"#00d4ff"),
     (0.75,"#ffaa00"),(1.0,"#ffffff")]
)

# ─────────────────────────────────────────────────────────────────────────────
# Sample FHSS message payloads for the console
# ─────────────────────────────────────────────────────────────────────────────
MSG_TEMPLATES = [
    "ACK {seq:04X} OK",
    "DATA [{seq:04X}] payload=0x{data:08X} crc=OK",
    "BEACON sync={sync} drift={drift:+.2f}Hz",
    "CTRL cmd=0x{cmd:02X} ack_req=1",
    "PING ttl={ttl} rtt={rtt:.1f}ms",
    "STATUS rssi={rssi:.0f}dBm ber={ber:.2e}",
    "SYNC epoch={epoch} offset={off:+d}us",
    "DATA [{seq:04X}] len={ln}B frag={frag}/1",
    "ACK {seq:04X} NACK retx={retx}",
    "TELEMETRY bat={bat:.2f}V temp={temp:.1f}C",
    "KEYXCHG phase=2 iv=0x{iv:016X}",
    "FLOOD bcast id=0x{id:04X} ttl={ttl}",
]

JAMMER_TEMPLATES = [
    "⚡ JAM detected on {freq:.2f} MHz — burst {dur:.0f}ms",
    "⚡ SWEEP jammer {f0:.2f}→{f1:.2f} MHz @ {rate:.0f}MHz/s",
    "⚡ TONE jammer @ {freq:.2f} MHz, pwr={pwr:.0f}dBm",
    "⚡ BARRAGE jammer covering {bw:.1f} MHz BW",
    "⚡ REPEAT jammer cloning last frame on {freq:.2f} MHz",
]

NOISE_EVENTS = [
    "⚠ High noise floor detected (+{delta:.0f}dB above baseline)",
    "⚠ Impulsive noise burst on {freq:.2f} MHz",
    "⚠ Adjacent channel bleed from {freq:.2f} MHz",
    "⚠ Phase noise spike — sync loss risk",
]

ERROR_EVENTS = [
    "✗ CRC FAIL [{seq:04X}] — packet dropped",
    "✗ DECODE ERR: BER={ber:.2e} — retransmit requested",
    "✗ HOP MISS: expected CH{exp}, got CH{got}",
    "✗ SYNC LOST — re-acquiring hopping pattern…",
    "✓ SYNC RECOVERED after {hops} hops",
]


def random_msg(eng):
    """Generate a realistic-looking received message for the console."""
    rng = random.Random()
    t   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    ch_idx = rng.randint(0, eng.num_channels - 1)
    ch  = eng.channels[ch_idx]

    roll = rng.random()

    # Decide message type based on noise/jam levels
    jam_prob   = eng.jam_level / 100.0
    noise_prob = eng.noise_level / 100.0

    if roll < jam_prob * 0.6:
        tmpl = rng.choice(JAMMER_TEMPLATES)
        msg  = tmpl.format(
            freq=ch["freq"], dur=rng.uniform(10,80),
            f0=eng.base_freq, f1=eng.base_freq + eng.total_bw,
            rate=rng.uniform(50,500), pwr=rng.uniform(-40,-10),
            bw=eng.total_bw * rng.uniform(0.3,1.0)
        )
        level = "JAM"
    elif roll < jam_prob * 0.6 + noise_prob * 0.4:
        tmpl = rng.choice(NOISE_EVENTS)
        msg  = tmpl.format(
            delta=rng.uniform(3, 15), freq=ch["freq"]
        )
        level = "NOISE"
    elif roll < jam_prob * 0.6 + noise_prob * 0.4 + (jam_prob + noise_prob) * 0.2:
        tmpl = rng.choice(ERROR_EVENTS)
        msg  = tmpl.format(
            seq=rng.randint(0, 0xFFFF),
            ber=rng.uniform(1e-5, 1e-2),
            exp=rng.randint(1, eng.num_channels),
            got=rng.randint(1, eng.num_channels),
            hops=rng.randint(2, 8)
        )
        level = "ERR"
    else:
        tmpl = rng.choice(MSG_TEMPLATES)
        msg  = tmpl.format(
            seq=rng.randint(0, 0xFFFF),
            data=rng.randint(0, 0xFFFFFFFF),
            sync=rng.randint(1000, 9999),
            drift=rng.uniform(-5, 5),
            cmd=rng.randint(0, 0xFF),
            ttl=rng.randint(1, 16),
            rtt=rng.uniform(0.5, 12.0),
            rssi=rng.uniform(-90, -40),
            ber=rng.uniform(1e-9, 1e-4),
            epoch=rng.randint(1000, 9999),
            off=rng.randint(-500, 500),
            ln=rng.choice([32, 64, 128, 256]),
            frag=1,
            bat=rng.uniform(3.2, 4.2),
            temp=rng.uniform(20.0, 55.0),
            iv=rng.randint(0, 2**63),
            id=rng.randint(0, 0xFFFF),
            retx=rng.randint(1, 4),
        )
        level = "RX"

    return t, ch["label"], ch["freq"], level, msg


# ─────────────────────────────────────────────────────────────────────────────
# Modulation Signal Generators
# ─────────────────────────────────────────────────────────────────────────────
class ModulationEngine:
    MODULATIONS = [
        "BPSK","QPSK","8PSK","16QAM","64QAM",
        "RTCM/MSK","GMSK","FSK","AM-DSB","FM",
        "OFDM","CSS (LoRa-like)","DSSS","CW"
    ]

    @staticmethod
    def spectral_shape(mod: str, freqs_norm: np.ndarray) -> np.ndarray:
        f = freqs_norm
        if mod in ("BPSK",):
            x = f * 10
            return np.sinc(x) ** 2
        elif mod in ("QPSK", "8PSK"):
            x = f * 20
            return np.sinc(x) ** 2
        elif mod in ("16QAM", "64QAM"):
            bw = 0.25
            s  = np.where(np.abs(f) < bw, 1.0,
                 np.where(np.abs(f) < bw * 1.5,
                          0.5 * (1 + np.cos(np.pi * (np.abs(f) - bw) / (0.5 * bw))),
                          0.0))
            return s
        elif mod in ("RTCM/MSK", "GMSK"):
            s  = 0.5 * np.sinc((f - 0.25) * 8) ** 2 \
               + 0.5 * np.sinc((f + 0.25) * 8) ** 2
            return s / (s.max() + 1e-12)
        elif mod == "FSK":
            s  = 0.5 * np.exp(-((f - 0.15) ** 2) / (2 * 0.04 ** 2)) \
               + 0.5 * np.exp(-((f + 0.15) ** 2) / (2 * 0.04 ** 2))
            return s / s.max()
        elif mod == "AM-DSB":
            s  = 0.6 * np.exp(-(f ** 2) / (2 * 0.01 ** 2)) \
               + 0.2 * np.exp(-((f - 0.05) ** 2) / (2 * 0.01 ** 2)) \
               + 0.2 * np.exp(-((f + 0.05) ** 2) / (2 * 0.01 ** 2))
            return s / s.max()
        elif mod == "FM":
            bw = 0.2
            s  = np.where(np.abs(f) < bw, 1.0,
                          np.exp(-((np.abs(f) - bw) ** 2) / (2 * 0.02 ** 2)))
            return s
        elif mod == "OFDM":
            bw = 0.35
            s  = np.where(np.abs(f) < bw, 1.0,
                          np.exp(-((np.abs(f) - bw) ** 2) / (2 * 0.015 ** 2)))
            return s
        elif mod == "CSS (LoRa-like)":
            bw = 0.45
            return np.where(np.abs(f) < bw, 1.0, 0.0).astype(float)
        elif mod == "DSSS":
            return np.where(np.abs(f) < 0.48, 0.9 + 0.1 * np.random.rand(len(f)), 0.0)
        elif mod == "CW":
            return np.exp(-(f ** 2) / (2 * 0.005 ** 2))
        else:
            return np.ones_like(f)


# ─────────────────────────────────────────────────────────────────────────────
# FHSS Engine
# ─────────────────────────────────────────────────────────────────────────────
class FHSSEngine:
    def __init__(self):
        self.num_channels  = 8
        self.base_freq     = 2400.0
        self.channel_bw    = 1.0
        self.hop_interval  = 10
        self.sequence_len  = 32
        self.seed          = 42
        self.pattern       = "Pseudo-Random"
        self.modulation    = "QPSK"
        self.snr_db        = 20.0
        # ── New parameters ──
        self.noise_level   = 20    # 0–100 (background noise floor raise %)
        self.jam_level     = 0     # 0–100 (jammer power %)
        self.jam_type      = "Tone"  # Tone / Sweep / Barrage / Pulse
        self._build_channels()
        self._build_sequence()

    def _build_channels(self):
        self.channels = [
            {
                "id":    i,
                "freq":  round(self.base_freq + i * self.channel_bw, 2),
                "label": f"CH{i+1}",
                "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
            }
            for i in range(self.num_channels)
        ]

    def _build_sequence(self):
        rng = random.Random(self.seed)
        if self.pattern == "Pseudo-Random":
            self.sequence = [rng.randint(0, self.num_channels - 1)
                             for _ in range(self.sequence_len)]
        elif self.pattern == "Sequential":
            self.sequence = [i % self.num_channels
                             for i in range(self.sequence_len)]
        elif self.pattern == "Interleaved":
            self.sequence = [
                (i * 2) % self.num_channels if i % 2 == 0
                else (i * 2 + 1) % self.num_channels
                for i in range(self.sequence_len)
            ]
        else:
            self.sequence = [rng.randint(0, self.num_channels - 1)
                             for _ in range(self.sequence_len)]

    def reconfigure(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._build_channels()
        self._build_sequence()

    @property
    def time_axis(self):
        return [i * self.hop_interval for i in range(self.sequence_len)]

    @property
    def freq_axis(self):
        return [self.channels[ch]["freq"] for ch in self.sequence]

    @property
    def total_bw(self):
        return self.num_channels * self.channel_bw

    @property
    def span_start(self):
        return self.base_freq - self.channel_bw * 0.5

    @property
    def span_end(self):
        return self.base_freq + self.num_channels * self.channel_bw + self.channel_bw * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Live FFT + Waterfall Canvas
# ─────────────────────────────────────────────────────────────────────────────
class SpectrumCanvas(FigureCanvas):
    N_FFT       = 512
    WATERFALL_H = 80
    NOISE_FLOOR = -90.0
    PEAK_POWER  = -20.0

    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine      = engine
        self._hop_idx    = 0
        self._tick_count = 0
        self._wf_data = np.full((self.WATERFALL_H, self.N_FFT),
                                self.NOISE_FLOOR, dtype=float)
        self.fig = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._build_axes()
        self._init_artists()

    def _build_axes(self):
        self.fig.clear()
        gs = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[2, 1.2],
            hspace=0.08,
            left=0.08, right=0.97, top=0.93, bottom=0.08
        )
        self.ax_fft = self.fig.add_subplot(gs[0])
        self.ax_wf  = self.fig.add_subplot(gs[1], sharex=self.ax_fft)

        for ax in (self.ax_fft, self.ax_wf):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        self.fig.patch.set_facecolor(BG)
        self.ax_fft.set_title("Live Spectrum  +  Waterfall", color=TEXT,
                               fontsize=10, fontweight="bold", pad=6)
        self.ax_fft.set_ylabel("Power (dBm)", color=MUTED, fontsize=8)
        self.ax_fft.set_ylim(self.NOISE_FLOOR, self.PEAK_POWER + 15)
        self.ax_fft.grid(color=BORDER, linewidth=0.4, alpha=0.6)
        self.ax_wf.set_xlabel("Frequency (MHz)", color=MUTED, fontsize=8)
        self.ax_wf.set_ylabel("Time →", color=MUTED, fontsize=7)
        self.ax_fft.tick_params(labelbottom=False)

    def _init_artists(self):
        eng  = self.engine
        freq = self._freq_axis()

        self._fft_line, = self.ax_fft.plot(
            freq, np.full(self.N_FFT, self.NOISE_FLOOR),
            color=ACCENT, linewidth=1.0, alpha=0.9, zorder=5
        )
        self._peak_line, = self.ax_fft.plot(
            freq, np.full(self.N_FFT, self.NOISE_FLOOR),
            color=ACCENT2, linewidth=0.6, alpha=0.5,
            linestyle="--", zorder=4
        )
        # Jammer overlay line
        self._jam_line, = self.ax_fft.plot(
            freq, np.full(self.N_FFT, self.NOISE_FLOOR - 1),
            color=RED, linewidth=1.2, alpha=0.0, zorder=6
        )
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)

        self._band_patches = []
        for ch in eng.channels:
            p = self.ax_fft.axvspan(
                ch["freq"] - eng.channel_bw * 0.5,
                ch["freq"] + eng.channel_bw * 0.5,
                alpha=0.07, color=ch["color"], zorder=1
            )
            self._band_patches.append(p)

        self._active_span = self.ax_fft.axvspan(
            eng.channels[0]["freq"] - eng.channel_bw * 0.5,
            eng.channels[0]["freq"] + eng.channel_bw * 0.5,
            alpha=0.25, color=ACCENT, zorder=2
        )

        self._mod_text = self.ax_fft.text(
            0.01, 0.95, f"MOD: {eng.modulation}",
            transform=self.ax_fft.transAxes,
            color=GREEN, fontsize=8, fontweight="bold",
            va="top", fontfamily="monospace"
        )
        self._ch_text = self.ax_fft.text(
            0.01, 0.86, "CH: —",
            transform=self.ax_fft.transAxes,
            color=ACCENT, fontsize=8, fontfamily="monospace"
        )
        self._snr_text = self.ax_fft.text(
            0.01, 0.77, f"SNR: {eng.snr_db:.0f} dB",
            transform=self.ax_fft.transAxes,
            color=YELLOW, fontsize=8, fontfamily="monospace"
        )
        self._noise_text = self.ax_fft.text(
            0.01, 0.68, f"NOISE: {eng.noise_level}%",
            transform=self.ax_fft.transAxes,
            color=MUTED, fontsize=8, fontfamily="monospace"
        )
        self._jam_text = self.ax_fft.text(
            0.01, 0.59, f"JAM: {eng.jam_level}%",
            transform=self.ax_fft.transAxes,
            color=RED, fontsize=8, fontfamily="monospace"
        )

        self._wf_im = self.ax_wf.imshow(
            self._wf_data,
            aspect="auto", origin="upper", cmap=_wf_cmap,
            vmin=self.NOISE_FLOOR, vmax=self.PEAK_POWER + 10,
            extent=[freq[0], freq[-1], self.WATERFALL_H, 0],
            interpolation="bilinear", zorder=2
        )
        self._set_xlim()
        self.draw()

    def _freq_axis(self):
        eng = self.engine
        return np.linspace(eng.span_start, eng.span_end, self.N_FFT)

    def _set_xlim(self):
        eng = self.engine
        self.ax_fft.set_xlim(eng.span_start, eng.span_end)
        xticks = [c["freq"] for c in eng.channels]
        self.ax_fft.set_xticks(xticks)
        self.ax_fft.set_xticklabels(
            [f"{x:.1f}" for x in xticks], fontsize=6.5, color=MUTED, rotation=30
        )

    def _compute_noise_floor(self) -> np.ndarray:
        """Elevated noise floor based on noise_level (0–100)."""
        eng       = self.engine
        # noise_level=0 → +0 dB, noise_level=100 → +30 dB lift
        lift_db   = eng.noise_level * 0.30
        base      = self.NOISE_FLOOR + lift_db
        variation = 1.8 + eng.noise_level * 0.08
        return base + np.random.randn(self.N_FFT) * variation

    def _compute_jammer(self, freq: np.ndarray) -> np.ndarray:
        """Returns jammer power spectrum in dBm (or NOISE_FLOOR-1 if off)."""
        eng = self.engine
        if eng.jam_level == 0:
            return np.full(self.N_FFT, self.NOISE_FLOOR - 5)

        # Jammer peak power: jam_level=100 → -5 dBm, jam_level=1 → -60 dBm
        jam_peak  = -60 + eng.jam_level * 0.55
        j_type    = eng.jam_type
        jam       = np.full(self.N_FFT, self.NOISE_FLOOR - 5, dtype=float)

        if j_type == "Tone":
            # Single CW tone on a random channel
            target = random.choice(eng.channels)["freq"]
            sigma  = eng.channel_bw * 0.08
            jam   += (jam_peak - (self.NOISE_FLOOR - 5)) * np.exp(
                         -((freq - target) ** 2) / (2 * sigma ** 2))

        elif j_type == "Sweep":
            # Swept tone: position cycles with time
            t_norm  = (time.time() % 2.0) / 2.0
            sweep_f = eng.span_start + t_norm * (eng.span_end - eng.span_start)
            sigma   = eng.channel_bw * 0.12
            jam    += (jam_peak - (self.NOISE_FLOOR - 5)) * np.exp(
                          -((freq - sweep_f) ** 2) / (2 * sigma ** 2))

        elif j_type == "Barrage":
            # Wideband flat jammer with some ripple
            ripple  = np.random.randn(self.N_FFT) * 3
            mask    = np.where(
                (freq >= eng.span_start + eng.channel_bw * 0.3) &
                (freq <= eng.span_end   - eng.channel_bw * 0.3),
                1.0, 0.0)
            jam    += (jam_peak - (self.NOISE_FLOOR - 5) - 8 + ripple) * mask

        elif j_type == "Pulse":
            # Random bursts on random channels
            if random.random() < 0.4:
                for _ in range(random.randint(1, 3)):
                    target = random.choice(eng.channels)["freq"]
                    sigma  = eng.channel_bw * 0.15
                    jam   += (jam_peak - (self.NOISE_FLOOR - 5)) * np.exp(
                                 -((freq - target) ** 2) / (2 * sigma ** 2))

        return jam

    def _compute_fft_frame(self, ch_idx: int) -> tuple:
        """Returns (combined_signal_dBm, jammer_dBm)."""
        eng     = self.engine
        freq    = self._freq_axis()
        ch_freq = eng.channels[ch_idx]["freq"]

        # Noise (raised by noise_level)
        noise = self._compute_noise_floor()

        # Signal spectral shape
        f_norm  = (freq - ch_freq) / (eng.channel_bw * 0.5)
        shape   = ModulationEngine.spectral_shape(eng.modulation, f_norm * 0.5)
        sig_db  = self.PEAK_POWER + 10 * np.log10(shape + 1e-12)

        snr_lin = 10 ** (eng.snr_db / 10)
        combined = 10 * np.log10(
            10 ** (noise / 10) + snr_lin * 10 ** (sig_db / 10)
        )

        from scipy.ndimage import uniform_filter1d
        combined = uniform_filter1d(combined, size=3)

        # Jammer
        jammer = self._compute_jammer(freq)

        # Merge jammer into combined
        if eng.jam_level > 0:
            merged = 10 * np.log10(
                10 ** (combined / 10) + 10 ** (jammer / 10)
            )
        else:
            merged = combined

        return merged, jammer

    def update_hop(self, hop_idx: int):
        if hop_idx < 0:
            return
        eng    = self.engine
        ch_idx = eng.sequence[hop_idx % len(eng.sequence)]
        ch     = eng.channels[ch_idx]
        freq   = self._freq_axis()

        frame, jam_frame = self._compute_fft_frame(ch_idx)

        self._peak_hold = np.maximum(self._peak_hold, frame)
        self._peak_hold = self._peak_hold * 0.97 + self.NOISE_FLOOR * 0.03

        self._fft_line.set_ydata(frame)
        self._peak_line.set_ydata(self._peak_hold)

        # Jammer overlay
        if eng.jam_level > 0:
            self._jam_line.set_ydata(jam_frame)
            self._jam_line.set_alpha(0.55)
        else:
            self._jam_line.set_alpha(0.0)

        x0 = ch["freq"] - eng.channel_bw * 0.5
        x1 = ch["freq"] + eng.channel_bw * 0.5
        self._active_span.remove()
        self._active_span = self.ax_fft.axvspan(
            x0, x1, alpha=0.25, color=ch["color"], zorder=2
        )

        self._mod_text.set_text(f"MOD: {eng.modulation}")
        self._ch_text.set_text(f"CH:  {ch['label']}  {ch['freq']:.2f} MHz")
        self._snr_text.set_text(f"SNR: {eng.snr_db:.0f} dB")
        noise_color = YELLOW if eng.noise_level > 50 else MUTED
        self._noise_text.set_text(f"NOISE: {eng.noise_level}%")
        self._noise_text.set_color(noise_color)
        jam_color = RED if eng.jam_level > 20 else (ORANGE if eng.jam_level > 0 else MUTED)
        self._jam_text.set_text(f"JAM [{eng.jam_type}]: {eng.jam_level}%")
        self._jam_text.set_color(jam_color)

        self._wf_data = np.roll(self._wf_data, -1, axis=0)
        self._wf_data[-1, :] = frame
        self._wf_im.set_data(self._wf_data)
        self._wf_im.set_extent([freq[0], freq[-1], self.WATERFALL_H, 0])

        self.draw_idle()

    def rebuild(self):
        self._wf_data   = np.full((self.WATERFALL_H, self.N_FFT),
                                  self.NOISE_FLOOR, dtype=float)
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)
        self._build_axes()
        self._init_artists()


# ─────────────────────────────────────────────────────────────────────────────
# Hop Map Canvas
# ─────────────────────────────────────────────────────────────────────────────
class HopMapCanvas(FigureCanvas):
    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine = engine
        self.fig    = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._highlight_patch = None
        self._draw_static()

    def _draw_static(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        ax  = self.ax
        eng = self.engine

        ax.set_facecolor(BG)
        self.fig.patch.set_facecolor(BG)

        times = eng.time_axis
        freqs = eng.freq_axis
        seq   = eng.sequence
        chans = eng.channels
        bw    = eng.channel_bw

        for ch in chans:
            ax.axhspan(ch["freq"] - bw * 0.45, ch["freq"] + bw * 0.45,
                       alpha=0.06, color=ch["color"])
            ax.axhline(ch["freq"], color=ch["color"], alpha=0.15,
                       linewidth=0.5, linestyle="--")

        for i in range(len(times)):
            ch   = chans[seq[i]]
            rect = mpatches.FancyBboxPatch(
                (times[i], freqs[i] - bw * 0.4),
                eng.hop_interval * 0.92, bw * 0.8,
                boxstyle="round,pad=0.01",
                linewidth=1.1, edgecolor=ch["color"],
                facecolor=ch["color"] + "35"
            )
            ax.add_patch(rect)
            ax.text(times[i] + eng.hop_interval * 0.46, freqs[i], ch["label"],
                    ha="center", va="center", fontsize=6,
                    color=ch["color"], fontweight="bold", alpha=0.85)

        ax.plot(
            [t + eng.hop_interval * 0.46 for t in times], freqs,
            color=ACCENT, linewidth=0.8, alpha=0.3, linestyle=":", zorder=5
        )

        ax.set_xlim(-eng.hop_interval * 0.5, times[-1] + eng.hop_interval * 1.5)
        freq_vals = [c["freq"] for c in chans]
        ax.set_ylim(min(freq_vals) - bw, max(freq_vals) + bw)
        ax.set_xlabel("Time (ms)", color=TEXT, fontsize=8, labelpad=6)
        ax.set_ylabel("Frequency (MHz)", color=TEXT, fontsize=8, labelpad=6)
        ax.set_title("Hop Map — Frequency vs Time", color=TEXT,
                     fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.set_yticks([c["freq"] for c in chans])
        ax.set_yticklabels([f"{c['freq']:.1f}" for c in chans], fontsize=7, color=MUTED)
        step = max(1, len(times) // 8)
        ax.set_xticks([times[i] for i in range(0, len(times), step)])
        ax.set_xticklabels([str(times[i]) for i in range(0, len(times), step)],
                           fontsize=7, color=MUTED)
        ax.grid(color=BORDER, linewidth=0.35, alpha=0.5)

        handles = [mpatches.Patch(color=c["color"],
                   label=f"{c['label']} {c['freq']} MHz") for c in chans]
        ax.legend(handles=handles, loc="upper right", fontsize=6.2, ncol=2,
                  framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

        self._highlight_patch = None
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def highlight_hop(self, hop_index: int):
        eng   = self.engine
        bw    = eng.channel_bw
        times = eng.time_axis
        freqs = eng.freq_axis

        if self._highlight_patch:
            try:
                self._highlight_patch.remove()
            except Exception:
                pass
            self._highlight_patch = None

        if 0 <= hop_index < len(times):
            rect = mpatches.FancyBboxPatch(
                (times[hop_index], freqs[hop_index] - bw * 0.48),
                eng.hop_interval * 0.92, bw * 0.96,
                boxstyle="round,pad=0.01",
                linewidth=2.2, edgecolor="#ffffff",
                facecolor="#ffffff1a", zorder=10
            )
            self.ax.add_patch(rect)
            self._highlight_patch = rect
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Channel Table
# ─────────────────────────────────────────────────────────────────────────────
class ChannelTable(QTableWidget):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Channel", "Freq (MHz)", "Hops"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(f"""
            QTableWidget {{ background:{PANEL}; color:{TEXT};
                border:1px solid {BORDER}; gridline-color:{BORDER}; font-size:11px; }}
            QTableWidget::item:alternate {{ background:{BG}; }}
            QTableWidget::item:selected  {{ background:{ACCENT}40; color:{TEXT}; }}
            QHeaderView::section {{ background:{BG}; color:{MUTED};
                border:none; border-bottom:1px solid {BORDER};
                padding:5px; font-size:10px; }}
        """)
        self.refresh()

    def refresh(self):
        eng = self.engine
        hop_counts = [0] * eng.num_channels
        for ch in eng.sequence:
            hop_counts[ch] += 1
        self.setRowCount(eng.num_channels)
        for i, ch in enumerate(eng.channels):
            color = QColor(ch["color"])
            lbl = QTableWidgetItem(ch["label"])
            lbl.setForeground(color)
            lbl.setFont(QFont("Courier New", 10, QFont.Bold))
            freq = QTableWidgetItem(f"{ch['freq']:.2f}")
            freq.setTextAlignment(Qt.AlignCenter)
            freq.setForeground(QColor(TEXT))
            hops = QTableWidgetItem(str(hop_counts[i]))
            hops.setTextAlignment(Qt.AlignCenter)
            hops.setForeground(QColor(GREEN if hop_counts[i] > 0 else MUTED))
            self.setItem(i, 0, lbl)
            self.setItem(i, 1, freq)
            self.setItem(i, 2, hops)

    def highlight_channel(self, ch_index: int):
        self.selectRow(ch_index)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Widget
# ─────────────────────────────────────────────────────────────────────────────
class SequenceWidget(QFrame):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"background:{PANEL}; border:1px solid {BORDER}; border-radius:6px;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(3)
        title = QLabel("Hopping Sequence")
        title.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:600;")
        layout.addWidget(title)
        self.seq_label = QLabel()
        self.seq_label.setWordWrap(True)
        self.seq_label.setStyleSheet(
            f"color:{TEXT}; font-family:'Courier New'; font-size:10px;")
        layout.addWidget(self.seq_label)
        self.refresh()

    def refresh(self):
        eng  = self.engine
        html = ""
        for i, ch in enumerate(eng.sequence):
            color = eng.channels[ch]["color"]
            label = eng.channels[ch]["label"]
            html += f'<span style="color:{color};font-weight:bold;">{label}</span>'
            if i < len(eng.sequence) - 1:
                html += f'<span style="color:{BORDER}"> → </span>'
        self.seq_label.setText(html)

    def highlight_hop(self, hop_index: int):
        eng  = self.engine
        html = ""
        for i, ch in enumerate(eng.sequence):
            color = eng.channels[ch]["color"]
            label = eng.channels[ch]["label"]
            if i == hop_index:
                html += (f'<span style="background:{color}40;color:{color};'
                         f'font-weight:bold;border:1px solid {color};'
                         f'border-radius:3px;padding:1px 3px;">{label}</span>')
            else:
                html += f'<span style="color:{color}55;font-weight:bold;">{label}</span>'
            if i < len(eng.sequence) - 1:
                html += f'<span style="color:{BORDER}"> → </span>'
        self.seq_label.setText(html)


# ─────────────────────────────────────────────────────────────────────────────
# Stats Bar
# ─────────────────────────────────────────────────────────────────────────────
class StatsBar(QWidget):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._labels = {}
        for key in ["Channels","Hop Rate","BW/Ch","Total BW","Seq Len","Modulation","SNR","Noise","Jammer"]:
            frame = QFrame()
            frame.setStyleSheet(
                f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 4, 8, 4)
            fl.setSpacing(1)
            k = QLabel(key)
            k.setStyleSheet(f"color:{MUTED}; font-size:9px;")
            v = QLabel("—")
            accent = RED if key == "Jammer" else (YELLOW if key == "Noise" else ACCENT)
            v.setStyleSheet(f"color:{accent}; font-size:12px; font-weight:bold;")
            fl.addWidget(k)
            fl.addWidget(v)
            layout.addWidget(frame)
            self._labels[key] = v
        layout.addStretch()
        self.refresh()

    def refresh(self):
        eng = self.engine
        self._labels["Channels"].setText(str(eng.num_channels))
        self._labels["Hop Rate"].setText(f"{1000/eng.hop_interval:.0f} h/s")
        self._labels["BW/Ch"].setText(f"{eng.channel_bw:.1f} MHz")
        self._labels["Total BW"].setText(f"{eng.total_bw:.1f} MHz")
        self._labels["Seq Len"].setText(str(eng.sequence_len))
        self._labels["Modulation"].setText(eng.modulation)
        self._labels["SNR"].setText(f"{eng.snr_db:.0f} dB")
        self._labels["Noise"].setText(f"{eng.noise_level}%")
        jam_txt = f"{eng.jam_level}% [{eng.jam_type}]" if eng.jam_level > 0 else "OFF"
        self._labels["Jammer"].setText(jam_txt)


# ─────────────────────────────────────────────────────────────────────────────
# Message Console
# ─────────────────────────────────────────────────────────────────────────────
class MessageConsole(QFrame):
    """Scrolling terminal-style console showing received/event messages."""

    MAX_LINES = 300

    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine   = engine
        self._line_count = 0
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background:{BG};
                border:1px solid {BORDER};
                border-radius:6px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid {BORDER};")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(10, 5, 10, 5)
        hdr_layout.setSpacing(8)

        dot_rx    = QLabel("●")
        dot_rx.setStyleSheet(f"color:{GREEN}; font-size:10px;")
        title_lbl = QLabel("COMM CONSOLE")
        title_lbl.setStyleSheet(
            f"color:{TEXT}; font-size:10px; font-weight:700; "
            f"font-family:'Courier New'; letter-spacing:1px;")

        self.pkt_count_lbl = QLabel("PKT: 0")
        self.pkt_count_lbl.setStyleSheet(f"color:{MUTED}; font-size:9px; font-family:'Courier New';")

        self.err_count_lbl = QLabel("ERR: 0")
        self.err_count_lbl.setStyleSheet(f"color:{RED}; font-size:9px; font-family:'Courier New';")

        self.jam_count_lbl = QLabel("JAM: 0")
        self.jam_count_lbl.setStyleSheet(f"color:{ORANGE}; font-size:9px; font-family:'Courier New';")

        btn_clear = QPushButton("CLR")
        btn_clear.setFixedWidth(36)
        btn_clear.setStyleSheet(f"""
            QPushButton {{ background:transparent; color:{MUTED}; border:1px solid {BORDER};
                border-radius:3px; font-size:9px; padding:1px 4px; }}
            QPushButton:hover {{ color:{TEXT}; border-color:{MUTED}; }}
        """)
        btn_clear.clicked.connect(self.clear)

        self.cb_pause = QCheckBox("Pause")
        self.cb_pause.setStyleSheet(f"color:{MUTED}; font-size:9px;")

        hdr_layout.addWidget(dot_rx)
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addSpacing(12)
        hdr_layout.addWidget(self.pkt_count_lbl)
        hdr_layout.addWidget(self.err_count_lbl)
        hdr_layout.addWidget(self.jam_count_lbl)
        hdr_layout.addStretch()
        hdr_layout.addWidget(self.cb_pause)
        hdr_layout.addWidget(btn_clear)
        layout.addWidget(hdr)

        # ── Text area ─────────────────────────────────────────────────────────
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Courier New", 9))
        self.text.setStyleSheet(f"""
            QTextEdit {{
                background:{BG}; color:{TEXT};
                border:none; padding:6px;
                selection-background-color:{ACCENT}40;
            }}
            QScrollBar:vertical {{ background:{BG}; width:6px; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
        """)
        layout.addWidget(self.text, 1)

        self._pkt_count = 0
        self._err_count = 0
        self._jam_count = 0

        # Seed with a few startup messages
        self._append_system("FHSS Simulation v3 — comm console ready")
        self._append_system(f"Engine: {engine.num_channels} ch · "
                            f"{engine.modulation} · seed={engine.seed}")

    # ── Formatting ────────────────────────────────────────────────────────────
    def _append_system(self, msg: str):
        self._raw_append(f"[SYS] {msg}", MUTED)

    def _raw_append(self, line: str, color: str):
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.insertText(line + "\n", fmt)

        # Trim old lines
        self._line_count += 1
        if self._line_count > self.MAX_LINES:
            cursor.movePosition(QTextCursor.Start)
            cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            self._line_count -= 1

        # Auto-scroll
        sb = self.text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def post_message(self, timestamp: str, ch_label: str, freq: float,
                     level: str, msg: str):
        if self.cb_pause.isChecked():
            return

        if level == "RX":
            color = GREEN
            prefix = f"[{timestamp}] {ch_label:>4} {freq:.2f}MHz RX"
            self._pkt_count += 1
            self.pkt_count_lbl.setText(f"PKT: {self._pkt_count}")
        elif level == "ERR":
            color = RED
            prefix = f"[{timestamp}] {ch_label:>4} {freq:.2f}MHz"
            self._err_count += 1
            self.err_count_lbl.setText(f"ERR: {self._err_count}")
        elif level == "JAM":
            color = ORANGE
            prefix = f"[{timestamp}]"
            self._jam_count += 1
            self.jam_count_lbl.setText(f"JAM: {self._jam_count}")
        elif level == "NOISE":
            color = YELLOW
            prefix = f"[{timestamp}]"
        else:
            color = MUTED
            prefix = f"[{timestamp}]"

        self._raw_append(f"{prefix} {msg}", color)

    def on_config_changed(self, engine: FHSSEngine):
        self.engine = engine
        self._append_system(
            f"Config updated → {engine.num_channels}ch · {engine.modulation} · "
            f"noise={engine.noise_level}% · jam={engine.jam_level}% [{engine.jam_type}]"
        )

    def clear(self):
        self.text.clear()
        self._line_count = 0
        self._pkt_count  = 0
        self._err_count  = 0
        self._jam_count  = 0
        self.pkt_count_lbl.setText("PKT: 0")
        self.err_count_lbl.setText("ERR: 0")
        self.jam_count_lbl.setText("JAM: 0")
        self._append_system("Console cleared")


# ─────────────────────────────────────────────────────────────────────────────
# Control Panel
# ─────────────────────────────────────────────────────────────────────────────
class ControlPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, engine: FHSSEngine):
        super().__init__("Configuration")
        self.engine = engine
        self.setStyleSheet(f"""
            QGroupBox {{ color:{TEXT}; border:1px solid {BORDER};
                border-radius:8px; font-size:11px;
                margin-top:8px; padding-top:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; color:{ACCENT}; }}
            QLabel {{ color:{MUTED}; font-size:10px; }}
            QSpinBox, QDoubleSpinBox, QComboBox {{
                background:{BG}; color:{TEXT}; border:1px solid {BORDER};
                border-radius:4px; padding:3px 5px; font-size:10px; }}
            QComboBox QAbstractItemView {{
                background:{PANEL}; color:{TEXT}; selection-background-color:{ACCENT}40; }}
        """)
        grid = QVBoxLayout(self)
        grid.setSpacing(6)

        def row(lbl, widget):
            r = QWidget()
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(lbl)
            l.setFixedWidth(115)
            h.addWidget(l)
            h.addWidget(widget, 1)
            grid.addWidget(r)
            return widget

        def slider_row(lbl, lo, hi, val, color, suffix=""):
            r = QWidget()
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(lbl)
            l.setFixedWidth(115)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            vl = QLabel(f"{val}{suffix}")
            vl.setFixedWidth(36)
            vl.setStyleSheet(f"color:{color}; font-size:10px;")
            sl.valueChanged.connect(lambda v: vl.setText(f"{v}{suffix}"))
            sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{ background:{PANEL}; height:3px; border-radius:2px; }}
                QSlider::handle:horizontal {{ background:{color}; width:12px; height:12px;
                    margin:-5px 0; border-radius:6px; }}
                QSlider::sub-page:horizontal {{ background:{color}; border-radius:2px; }}
            """)
            h.addWidget(l)
            h.addWidget(sl, 1)
            h.addWidget(vl)
            grid.addWidget(r)
            return sl

        # ── RF / Link params ──────────────────────────────────────────────────
        sep_lbl = QLabel("── RF / Link ──")
        sep_lbl.setStyleSheet(f"color:{ACCENT}; font-size:9px; font-weight:600;")
        grid.addWidget(sep_lbl)

        self.cb_mod = row("Modulation", QComboBox())
        self.cb_mod.addItems(ModulationEngine.MODULATIONS)
        idx = self.cb_mod.findText(engine.modulation)
        if idx >= 0:
            self.cb_mod.setCurrentIndex(idx)

        self.sb_channels = row("Channels", QSpinBox())
        self.sb_channels.setRange(2, 16)
        self.sb_channels.setValue(engine.num_channels)

        self.dsb_base = row("Base Freq (MHz)", QDoubleSpinBox())
        self.dsb_base.setRange(100, 6000)
        self.dsb_base.setValue(engine.base_freq)
        self.dsb_base.setSingleStep(10)

        self.dsb_bw = row("Ch BW (MHz)", QDoubleSpinBox())
        self.dsb_bw.setRange(0.1, 20)
        self.dsb_bw.setValue(engine.channel_bw)
        self.dsb_bw.setSingleStep(0.5)

        self.sb_interval = row("Hop Interval (ms)", QSpinBox())
        self.sb_interval.setRange(1, 500)
        self.sb_interval.setValue(engine.hop_interval)

        self.sb_seqlen = row("Sequence Length", QSpinBox())
        self.sb_seqlen.setRange(4, 64)
        self.sb_seqlen.setValue(engine.sequence_len)

        self.sb_seed = row("Seed", QSpinBox())
        self.sb_seed.setRange(0, 9999)
        self.sb_seed.setValue(engine.seed)

        self.cb_pattern = row("Hop Pattern", QComboBox())
        self.cb_pattern.addItems(["Pseudo-Random", "Sequential", "Interleaved"])

        self.snr_slider = slider_row("SNR (dB)", 0, 40, int(engine.snr_db), YELLOW, " dB")

        # ── Noise ─────────────────────────────────────────────────────────────
        sep_noise = QLabel("── Noise Floor ──")
        sep_noise.setStyleSheet(f"color:{YELLOW}; font-size:9px; font-weight:600;")
        grid.addWidget(sep_noise)

        self.noise_slider = slider_row("Noise Level", 0, 100, engine.noise_level, YELLOW, "%")

        # ── Jamming ───────────────────────────────────────────────────────────
        sep_jam = QLabel("── Jamming / Interference ──")
        sep_jam.setStyleSheet(f"color:{RED}; font-size:9px; font-weight:600;")
        grid.addWidget(sep_jam)

        self.jam_slider = slider_row("Jammer Power", 0, 100, engine.jam_level, RED, "%")

        self.cb_jam_type = row("Jammer Type", QComboBox())
        self.cb_jam_type.addItems(["Tone", "Sweep", "Barrage", "Pulse"])
        self.cb_jam_type.setStyleSheet(f"""
            QComboBox {{ background:{BG}; color:{RED}; border:1px solid {BORDER};
                border-radius:4px; padding:3px 5px; font-size:10px; }}
            QComboBox QAbstractItemView {{
                background:{PANEL}; color:{TEXT}; selection-background-color:{RED}40; }}
        """)

        # ── Apply ──────────────────────────────────────────────────────────────
        btn = QPushButton("Apply & Regenerate")
        btn.setStyleSheet(f"""
            QPushButton {{ background:{ACCENT}; color:#000; border:none;
                border-radius:6px; padding:6px; font-size:11px; font-weight:bold; }}
            QPushButton:hover {{ background:#33ddff; }}
        """)
        btn.clicked.connect(self._apply)
        grid.addWidget(btn)

    def _apply(self):
        self.engine.reconfigure(
            modulation    = self.cb_mod.currentText(),
            num_channels  = self.sb_channels.value(),
            base_freq     = self.dsb_base.value(),
            channel_bw    = self.dsb_bw.value(),
            hop_interval  = self.sb_interval.value(),
            sequence_len  = self.sb_seqlen.value(),
            seed          = self.sb_seed.value(),
            pattern       = self.cb_pattern.currentText(),
            snr_db        = float(self.snr_slider.value()),
            noise_level   = self.noise_slider.value(),
            jam_level     = self.jam_slider.value(),
            jam_type      = self.cb_jam_type.currentText(),
        )
        self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Animation / Transport Bar
# ─────────────────────────────────────────────────────────────────────────────
class AnimBar(QWidget):
    hop_changed = pyqtSignal(int)

    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine   = engine
        self._hop     = 0
        self._playing = False
        self._timer   = QTimer()
        self._timer.timeout.connect(self._tick)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_play  = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(85)
        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.setFixedWidth(85)
        self.btn_play.clicked.connect(self._toggle)
        self.btn_reset.clicked.connect(self._reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, engine.sequence_len - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._seek)

        self.hop_lbl = QLabel("Hop: 0 / 0")
        self.hop_lbl.setFixedWidth(95)
        self.hop_lbl.setStyleSheet(f"color:{TEXT}; font-size:10px;")

        self.speed_sb = QSpinBox()
        self.speed_sb.setRange(50, 2000)
        self.speed_sb.setValue(500)
        self.speed_sb.setSuffix(" ms")
        self.speed_sb.setFixedWidth(80)
        self.speed_sb.setStyleSheet(f"""
            QSpinBox {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};
                border-radius:4px; padding:2px 4px; font-size:10px; }}
        """)
        spd_lbl = QLabel("Speed:")
        spd_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")

        for w in (self.btn_play, self.btn_reset):
            w.setStyleSheet(f"""
                QPushButton {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};
                    border-radius:4px; padding:4px 8px; font-size:10px; }}
                QPushButton:hover {{ background:{HOVER}; }}
            """)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{PANEL}; height:4px; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:13px; height:13px;
                margin:-5px 0; border-radius:7px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}40; border-radius:2px; }}
        """)

        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_reset)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.hop_lbl)
        layout.addWidget(spd_lbl)
        layout.addWidget(self.speed_sb)

    def refresh(self, engine: FHSSEngine):
        self.engine = engine
        self._hop   = 0
        self.slider.setRange(0, engine.sequence_len - 1)
        self.slider.setValue(0)
        self._update_label()

    def _toggle(self):
        self._playing = not self._playing
        if self._playing:
            self._timer.start(self.speed_sb.value())
            self.btn_play.setText("⏸  Pause")
        else:
            self._timer.stop()
            self.btn_play.setText("▶  Play")

    def _reset(self):
        self._playing = False
        self._timer.stop()
        self.btn_play.setText("▶  Play")
        self._hop = 0
        self.slider.setValue(0)
        self.hop_changed.emit(-1)

    def _tick(self):
        self._timer.setInterval(self.speed_sb.value())
        self._hop = (self._hop + 1) % self.engine.sequence_len
        self.slider.setValue(self._hop)

    def _seek(self, value: int):
        self._hop = value
        self._update_label()
        self.hop_changed.emit(value)

    def _update_label(self):
        self.hop_lbl.setText(f"Hop: {self._hop+1} / {self.engine.sequence_len}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class FHSSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "FHSS Simulation v3  —  FFT · Waterfall · Modulation · Noise · Jamming")
        self.resize(1500, 960)
        self.engine = FHSSEngine()
        self._build_ui()
        self._apply_global_style()

        # Console message timer — fires independently of hop animation
        self._console_timer = QTimer()
        self._console_timer.timeout.connect(self._post_console_msg)
        self._console_timer.start(600)

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; }}
            QTabWidget::pane {{ border:1px solid {BORDER}; background:{PANEL}; }}
            QTabBar::tab {{ background:{BG}; color:{MUTED}; padding:6px 16px;
                border:1px solid {BORDER}; border-bottom:none; font-size:11px; }}
            QTabBar::tab:selected {{ background:{PANEL}; color:{TEXT}; border-top:2px solid {ACCENT}; }}
            QSplitter::handle {{ background:{BORDER}; }}
            QScrollBar:vertical {{ background:{BG}; width:7px; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # Header
        hdr = QLabel("⟁  FHSS Simulation v3  ·  FFT + Waterfall + Noise + Jamming + Console")
        hdr.setStyleSheet(f"""
            color:{ACCENT}; font-size:15px; font-weight:700;
            font-family:'Courier New'; letter-spacing:1.5px;
            border-bottom:1px solid {BORDER}; padding-bottom:5px;
        """)
        root.addWidget(hdr)

        # Stats
        self.stats = StatsBar(self.engine)
        root.addWidget(self.stats)

        # Outer vertical splitter: top (main) | bottom (console)
        outer_split = QSplitter(Qt.Vertical)
        outer_split.setHandleWidth(4)

        # ── Top: main H-splitter ──────────────────────────────────────────────
        main_split = QSplitter(Qt.Horizontal)
        main_split.setHandleWidth(2)

        # Left: tabs + anim bar
        self.tabs = QTabWidget()

        tab_spec = QWidget()
        tv1 = QVBoxLayout(tab_spec)
        tv1.setContentsMargins(4, 4, 4, 4)
        tv1.setSpacing(6)
        self.spec_canvas = SpectrumCanvas(self.engine)
        tv1.addWidget(self.spec_canvas, 1)
        self.tabs.addTab(tab_spec, "📡  FFT + Waterfall")

        tab_hop = QWidget()
        tv2 = QVBoxLayout(tab_hop)
        tv2.setContentsMargins(4, 4, 4, 4)
        tv2.setSpacing(6)
        self.hop_canvas = HopMapCanvas(self.engine)
        tv2.addWidget(self.hop_canvas, 1)
        self.seq_widget = SequenceWidget(self.engine)
        tv2.addWidget(self.seq_widget)
        self.tabs.addTab(tab_hop, "📶  Hop Map")

        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        lv.addWidget(self.tabs, 1)
        self.anim_bar = AnimBar(self.engine)
        self.anim_bar.hop_changed.connect(self._on_hop)
        lv.addWidget(self.anim_bar)
        main_split.addWidget(left_w)

        # Right: config + channel table (scrollable)
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_inner = QWidget()
        scroll_layout = QVBoxLayout(scroll_inner)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        self.ctrl = ControlPanel(self.engine)
        self.ctrl.changed.connect(self._on_config_changed)
        scroll_layout.addWidget(self.ctrl)

        ch_lbl = QLabel("Channel List")
        ch_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:600;")
        scroll_layout.addWidget(ch_lbl)

        self.table = ChannelTable(self.engine)
        scroll_layout.addWidget(self.table)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_inner)
        rv.addWidget(scroll, 1)
        main_split.addWidget(right_w)
        main_split.setSizes([980, 390])

        outer_split.addWidget(main_split)

        # ── Bottom: message console ───────────────────────────────────────────
        self.console = MessageConsole(self.engine)
        self.console.setMinimumHeight(140)
        outer_split.addWidget(self.console)
        outer_split.setSizes([680, 200])

        root.addWidget(outer_split, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_config_changed(self):
        self.spec_canvas.rebuild()
        self.hop_canvas._draw_static()
        self.stats.refresh()
        self.table.refresh()
        self.seq_widget.refresh()
        self.anim_bar.refresh(self.engine)
        self.console.on_config_changed(self.engine)

    def _on_hop(self, hop_index: int):
        self.spec_canvas.update_hop(hop_index)
        self.hop_canvas.highlight_hop(hop_index)
        if hop_index >= 0:
            ch_idx = self.engine.sequence[hop_index]
            self.table.highlight_channel(ch_idx)
            self.seq_widget.highlight_hop(hop_index)
            self.anim_bar._update_label()
            # Post a console message on each hop
            self._post_console_msg()
        else:
            self.table.clearSelection()
            self.seq_widget.refresh()

    def _post_console_msg(self):
        """Generate and post a random message to the console."""
        t, ch, freq, level, msg = random_msg(self.engine)
        self.console.post_message(t, ch, freq, level, msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window,        QColor(BG))
    palette.setColor(QPalette.WindowText,    QColor(TEXT))
    palette.setColor(QPalette.Base,          QColor(PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(BG))
    palette.setColor(QPalette.Text,          QColor(TEXT))
    palette.setColor(QPalette.Button,        QColor(PANEL))
    palette.setColor(QPalette.ButtonText,    QColor(TEXT))
    app.setPalette(palette)
    win = FHSSWindow()
    win.show()
    sys.exit(app.exec_())