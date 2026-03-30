"""
FHSS Simulation v2 — Live FFT + Waterfall + Modulation Selector
================================================================
Requires: PyQt5, matplotlib, numpy, scipy
Install:  pip install PyQt5 matplotlib numpy scipy
Run:      python fhss_simulation_v2.py
"""

import sys
import numpy as np
import random
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox,
    QSplitter, QFrame, QTabWidget, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

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

CHANNEL_COLORS = [
    "#00d4ff","#ff6b35","#39d353","#f0e68c",
    "#da70d6","#87ceeb","#ff69b4","#98fb98",
    "#ffa500","#7b68ee","#20b2aa","#ff4500",
    "#9370db","#3cb371","#b8860b","#4682b4",
]

# Custom waterfall colormap: black → blue → cyan → yellow → white
_wf_cmap = LinearSegmentedColormap.from_list(
    "waterfall",
    [(0,"#000000"),(0.25,"#0a2a6e"),(0.5,"#00d4ff"),
     (0.75,"#ffaa00"),(1.0,"#ffffff")]
)

# ─────────────────────────────────────────────────────────────────────────────
# Modulation Signal Generators
# ─────────────────────────────────────────────────────────────────────────────
class ModulationEngine:
    """
    Generates baseband IQ samples for various modulation schemes.
    All return complex numpy arrays of length `n_samples`.
    """
    MODULATIONS = [
        "BPSK","QPSK","8PSK","16QAM","64QAM",
        "RTCM/MSK","GMSK","FSK","AM-DSB","FM",
        "OFDM","CSS (LoRa-like)","DSSS","CW"
    ]

    @staticmethod
    def generate(mod: str, n_samples: int, fs: float, snr_db: float = 20.0) -> np.ndarray:
        """Return complex baseband IQ samples."""
        t   = np.arange(n_samples) / fs
        rng = np.random.default_rng()

        # ── noise ────────────────────────────────────────────────────
        noise_power = 10 ** (-snr_db / 10)
        noise = (rng.standard_normal(n_samples) +
                 1j * rng.standard_normal(n_samples)) * np.sqrt(noise_power / 2)

        # ── symbol helpers ───────────────────────────────────────────
        def psk_symbols(M):
            k    = int(np.log2(M))
            bits = rng.integers(0, 2, n_samples * k)
            idxs = np.packbits(bits.reshape(-1, k), axis=1,
                               bitorder='big').flatten()[:n_samples] % M
            angles = 2 * np.pi * idxs / M
            return np.exp(1j * angles)

        def qam_symbols(M):
            sqM   = int(np.sqrt(M))
            bits  = rng.integers(0, sqM, n_samples)
            bits2 = rng.integers(0, sqM, n_samples)
            levels = np.linspace(-(sqM - 1), sqM - 1, sqM)
            s = (levels[bits] + 1j * levels[bits2])
            return s / np.max(np.abs(s))

        # ── upsample helper ──────────────────────────────────────────
        sps = max(1, n_samples // max(1, n_samples // 8))  # ~8 samples/symbol

        if mod == "BPSK":
            sym   = psk_symbols(2)
            sig   = np.repeat(sym, max(1, n_samples // len(sym)))[:n_samples]

        elif mod == "QPSK":
            sym   = psk_symbols(4)
            sig   = np.repeat(sym, max(1, n_samples // len(sym)))[:n_samples]

        elif mod == "8PSK":
            sym   = psk_symbols(8)
            sig   = np.repeat(sym, max(1, n_samples // len(sym)))[:n_samples]

        elif mod == "16QAM":
            sym   = qam_symbols(16)
            sig   = np.repeat(sym, max(1, n_samples // len(sym)))[:n_samples]

        elif mod == "64QAM":
            sym   = qam_symbols(64)
            sig   = np.repeat(sym, max(1, n_samples // len(sym)))[:n_samples]

        elif mod == "RTCM/MSK":
            # Minimum Shift Keying: continuous-phase FSK h=0.5
            bits = rng.integers(0, 2, n_samples) * 2 - 1
            phase = np.cumsum(bits * np.pi / (2 * max(1, n_samples // 256)))
            sig   = np.exp(1j * phase)

        elif mod == "GMSK":
            # GMSK: smooth phase transitions (Gaussian filter on bits)
            from scipy.ndimage import gaussian_filter1d
            bits  = (rng.integers(0, 2, n_samples) * 2 - 1).astype(float)
            bits  = gaussian_filter1d(bits, sigma=max(1, n_samples // 512))
            phase = np.cumsum(bits * np.pi / (2 * max(1, n_samples // 256)))
            sig   = np.exp(1j * phase)

        elif mod == "FSK":
            bits  = rng.integers(0, 2, n_samples)
            dev   = 0.25  # normalized freq deviation
            phase = np.cumsum((bits * 2 - 1) * 2 * np.pi * dev / fs * fs / n_samples * n_samples)
            sig   = np.exp(1j * phase / n_samples * 2 * np.pi)

        elif mod == "AM-DSB":
            msg   = np.sin(2 * np.pi * 0.02 * np.arange(n_samples))
            sig   = (1 + 0.8 * msg).astype(complex)

        elif mod == "FM":
            msg   = np.sin(2 * np.pi * 0.015 * np.arange(n_samples))
            phase = np.cumsum(msg) * 0.3
            sig   = np.exp(1j * phase)

        elif mod == "OFDM":
            # Mini OFDM: 64 subcarriers, random QPSK pilots
            n_sc  = 64
            sym   = np.zeros(n_samples, dtype=complex)
            block = n_sc
            for start in range(0, n_samples - block, block):
                pilots = np.exp(1j * np.random.uniform(0, 2*np.pi, n_sc))
                sym[start:start+block] = np.fft.ifft(pilots)
            sig = sym

        elif mod == "CSS (LoRa-like)":
            # Chirp Spread Spectrum: linear chirp
            bw    = 0.5
            sig   = np.exp(1j * np.pi * bw * (np.arange(n_samples) / n_samples) ** 2 * n_samples)

        elif mod == "DSSS":
            # Direct-Sequence: data XOR'd with PN chip sequence
            data  = rng.integers(0, 2, n_samples // 11 + 1)
            chip  = rng.integers(0, 2, n_samples)
            spread = (np.repeat(data, 11)[:n_samples] ^ chip) * 2 - 1
            sig   = spread.astype(complex)

        elif mod == "CW":
            # Continuous Wave: pure carrier + slight AM keying
            key   = np.zeros(n_samples)
            for i in range(0, n_samples, n_samples // 8):
                key[i:i + n_samples // 16] = 1
            sig = key.astype(complex)

        else:
            sig = np.ones(n_samples, dtype=complex)

        # Normalize + add noise
        sig = sig / (np.max(np.abs(sig)) + 1e-12)
        return sig + noise

    @staticmethod
    def spectral_shape(mod: str, freqs_norm: np.ndarray) -> np.ndarray:
        """
        Returns a realistic *power spectral density shape* for the modulation
        on a normalized frequency axis [-0.5 … 0.5].
        Used by the live FFT renderer.
        """
        f = freqs_norm
        if mod in ("BPSK",):
            # sinc² main lobe
            x = f * 10
            return np.sinc(x) ** 2

        elif mod in ("QPSK", "8PSK"):
            x = f * 20
            return np.sinc(x) ** 2

        elif mod in ("16QAM", "64QAM"):
            # raised cosine-ish flat top
            bw = 0.25
            s  = np.where(np.abs(f) < bw, 1.0,
                 np.where(np.abs(f) < bw * 1.5,
                          0.5 * (1 + np.cos(np.pi * (np.abs(f) - bw) / (0.5 * bw))),
                          0.0))
            return s

        elif mod in ("RTCM/MSK", "GMSK"):
            # MSK: two sinc² side by side
            s  = 0.5 * np.sinc((f - 0.25) * 8) ** 2 \
               + 0.5 * np.sinc((f + 0.25) * 8) ** 2
            return s / (s.max() + 1e-12)

        elif mod == "FSK":
            s  = 0.5 * np.exp(-((f - 0.15) ** 2) / (2 * 0.04 ** 2)) \
               + 0.5 * np.exp(-((f + 0.15) ** 2) / (2 * 0.04 ** 2))
            return s / s.max()

        elif mod == "AM-DSB":
            # Carrier + two sidebands
            s  = 0.6 * np.exp(-(f ** 2) / (2 * 0.01 ** 2)) \
               + 0.2 * np.exp(-((f - 0.05) ** 2) / (2 * 0.01 ** 2)) \
               + 0.2 * np.exp(-((f + 0.05) ** 2) / (2 * 0.01 ** 2))
            return s / s.max()

        elif mod == "FM":
            # Flat occupied BW (Carson's rule)
            bw = 0.2
            s  = np.where(np.abs(f) < bw, 1.0,
                          np.exp(-((np.abs(f) - bw) ** 2) / (2 * 0.02 ** 2)))
            return s

        elif mod == "OFDM":
            # Flat rectangular spectrum
            bw = 0.35
            s  = np.where(np.abs(f) < bw, 1.0,
                          np.exp(-((np.abs(f) - bw) ** 2) / (2 * 0.015 ** 2)))
            return s

        elif mod == "CSS (LoRa-like)":
            # Wide flat chirp
            bw = 0.45
            return np.where(np.abs(f) < bw, 1.0, 0.0).astype(float)

        elif mod == "DSSS":
            # Noise-like wide flat
            return np.where(np.abs(f) < 0.48, 0.9 + 0.1 * np.random.rand(len(f)), 0.0)

        elif mod == "CW":
            # Delta-like spike
            return np.exp(-(f ** 2) / (2 * 0.005 ** 2))

        else:
            return np.ones_like(f)


# ─────────────────────────────────────────────────────────────────────────────
# FHSS Engine
# ─────────────────────────────────────────────────────────────────────────────
class FHSSEngine:
    def __init__(self):
        self.num_channels  = 8
        self.base_freq     = 2400.0   # MHz
        self.channel_bw    = 1.0      # MHz
        self.hop_interval  = 10       # ms
        self.sequence_len  = 32
        self.seed          = 42
        self.pattern       = "Pseudo-Random"
        self.modulation    = "QPSK"
        self.snr_db        = 20.0
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
    """
    Top half:   Live FFT power spectrum (dB) with modulation spectral shape.
    Bottom half: Scrolling waterfall (time-frequency intensity map).
    """
    N_FFT       = 512
    WATERFALL_H = 80          # rows kept in waterfall
    NOISE_FLOOR = -90.0       # dBm
    PEAK_POWER  = -20.0       # dBm  (signal peak)

    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine      = engine
        self._hop_idx    = 0
        self._tick_count = 0

        # Waterfall history: each row = one FFT frame
        self._wf_data = np.full((self.WATERFALL_H, self.N_FFT),
                                self.NOISE_FLOOR, dtype=float)

        self.fig = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._build_axes()
        self._init_artists()

    # ── Layout ───────────────────────────────────────────────────────────────
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

    # ── Initial artists ───────────────────────────────────────────────────────
    def _init_artists(self):
        eng  = self.engine
        freq = self._freq_axis()

        # FFT line
        self._fft_line, = self.ax_fft.plot(
            freq, np.full(self.N_FFT, self.NOISE_FLOOR),
            color=ACCENT, linewidth=1.0, alpha=0.9, zorder=5
        )
        # Peak-hold line
        self._peak_line, = self.ax_fft.plot(
            freq, np.full(self.N_FFT, self.NOISE_FLOOR),
            color=ACCENT2, linewidth=0.6, alpha=0.5,
            linestyle="--", zorder=4
        )
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)

        # Channel band shading on FFT
        self._band_patches = []
        for ch in eng.channels:
            p = self.ax_fft.axvspan(
                ch["freq"] - eng.channel_bw * 0.5,
                ch["freq"] + eng.channel_bw * 0.5,
                alpha=0.07, color=ch["color"], zorder=1
            )
            self._band_patches.append(p)

        # Active channel highlight
        self._active_span = self.ax_fft.axvspan(
            eng.channels[0]["freq"] - eng.channel_bw * 0.5,
            eng.channels[0]["freq"] + eng.channel_bw * 0.5,
            alpha=0.25, color=ACCENT, zorder=2
        )

        # Modulation label
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

        # Waterfall image
        self._wf_im = self.ax_wf.imshow(
            self._wf_data,
            aspect="auto",
            origin="upper",
            cmap=_wf_cmap,
            vmin=self.NOISE_FLOOR,
            vmax=self.PEAK_POWER + 10,
            extent=[freq[0], freq[-1], self.WATERFALL_H, 0],
            interpolation="bilinear",
            zorder=2
        )

        self._set_xlim()
        self.draw()

    # ── Helpers ───────────────────────────────────────────────────────────────
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

    def _compute_fft_frame(self, ch_idx: int) -> np.ndarray:
        """
        Generate a realistic FFT frame:
        - Noise floor with random variation
        - Modulation spectral shape centred on the active channel
        - Slight smearing for adjacent-channel bleed
        """
        eng      = self.engine
        freq     = self._freq_axis()
        ch_freq  = eng.channels[ch_idx]["freq"]

        # Noise floor
        noise = self.NOISE_FLOOR + np.random.randn(self.N_FFT) * 1.8

        # Spectral shape of modulation, normalised to channel BW
        f_norm   = (freq - ch_freq) / (eng.channel_bw * 0.5)
        shape    = ModulationEngine.spectral_shape(eng.modulation, f_norm * 0.5)

        # Power in dBm
        sig_db = self.PEAK_POWER + 10 * np.log10(shape + 1e-12)

        # SNR factor
        snr_lin = 10 ** (eng.snr_db / 10)
        combined = 10 * np.log10(
            10 ** (noise / 10) + snr_lin * 10 ** (sig_db / 10)
        )

        # Smooth slightly (simulate FFT windowing)
        from scipy.ndimage import uniform_filter1d
        combined = uniform_filter1d(combined, size=3)
        return combined

    # ── Public update ─────────────────────────────────────────────────────────
    def update_hop(self, hop_idx: int):
        if hop_idx < 0:
            return
        eng      = self.engine
        ch_idx   = eng.sequence[hop_idx % len(eng.sequence)]
        ch       = eng.channels[ch_idx]
        freq     = self._freq_axis()

        # New FFT frame
        frame = self._compute_fft_frame(ch_idx)

        # Update peak hold
        self._peak_hold = np.maximum(self._peak_hold, frame)
        # Decay peak hold slowly
        self._peak_hold = self._peak_hold * 0.97 + self.NOISE_FLOOR * 0.03

        # Update FFT line
        self._fft_line.set_ydata(frame)
        self._peak_line.set_ydata(self._peak_hold)

        # Update active channel span — remove old, add new axvspan
        x0 = ch["freq"] - eng.channel_bw * 0.5
        x1 = ch["freq"] + eng.channel_bw * 0.5
        self._active_span.remove()
        self._active_span = self.ax_fft.axvspan(
            x0, x1, alpha=0.25, color=ch["color"], zorder=2
        )

        # Labels
        self._mod_text.set_text(f"MOD: {eng.modulation}")
        self._ch_text.set_text(f"CH:  {ch['label']}  {ch['freq']:.2f} MHz")
        self._snr_text.set_text(f"SNR: {eng.snr_db:.0f} dB")

        # Push row into waterfall (scroll up)
        self._wf_data = np.roll(self._wf_data, -1, axis=0)
        self._wf_data[-1, :] = frame
        self._wf_im.set_data(self._wf_data)
        self._wf_im.set_extent([freq[0], freq[-1], self.WATERFALL_H, 0])

        self.draw_idle()

    def rebuild(self):
        """Full rebuild after engine reconfiguration."""
        self._wf_data   = np.full((self.WATERFALL_H, self.N_FFT),
                                  self.NOISE_FLOOR, dtype=float)
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)
        self._build_axes()
        self._init_artists()


# ─────────────────────────────────────────────────────────────────────────────
# Hop Map Canvas (Freq vs Time)
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
        for key in ["Channels","Hop Rate","BW/Ch","Total BW","Seq Len","Modulation","SNR"]:
            frame = QFrame()
            frame.setStyleSheet(
                f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 4, 8, 4)
            fl.setSpacing(1)
            k = QLabel(key)
            k.setStyleSheet(f"color:{MUTED}; font-size:9px;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{ACCENT}; font-size:12px; font-weight:bold;")
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

        # Modulation selector
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

        # SNR slider
        snr_row = QWidget()
        sh = QHBoxLayout(snr_row)
        sh.setContentsMargins(0, 0, 0, 0)
        snr_lbl = QLabel("SNR (dB)")
        snr_lbl.setFixedWidth(115)
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setRange(0, 40)
        self.snr_slider.setValue(int(engine.snr_db))
        self.snr_val_lbl = QLabel(f"{engine.snr_db:.0f}")
        self.snr_val_lbl.setFixedWidth(28)
        self.snr_val_lbl.setStyleSheet(f"color:{YELLOW}; font-size:10px;")
        self.snr_slider.valueChanged.connect(
            lambda v: self.snr_val_lbl.setText(str(v)))
        self.snr_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{PANEL}; height:3px; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{YELLOW}; width:12px; height:12px;
                margin:-5px 0; border-radius:6px; }}
            QSlider::sub-page:horizontal {{ background:{YELLOW}; border-radius:2px; }}
        """)
        sh.addWidget(snr_lbl)
        sh.addWidget(self.snr_slider, 1)
        sh.addWidget(self.snr_val_lbl)
        grid.addWidget(snr_row)

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
            "FHSS Simulation v2  —  FFT · Waterfall · Modulation Selector")
        self.resize(1400, 860)
        self.engine = FHSSEngine()
        self._build_ui()
        self._apply_global_style()

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
        hdr = QLabel("⟁  FHSS Simulation  ·  FFT + Waterfall + Modulation")
        hdr.setStyleSheet(f"""
            color:{ACCENT}; font-size:16px; font-weight:700;
            font-family:'Courier New'; letter-spacing:1.5px;
            border-bottom:1px solid {BORDER}; padding-bottom:5px;
        """)
        root.addWidget(hdr)

        # Stats
        self.stats = StatsBar(self.engine)
        root.addWidget(self.stats)

        # Main H splitter: left (tabs) | right (config + table)
        main_split = QSplitter(Qt.Horizontal)
        main_split.setHandleWidth(2)

        # ── Left: tabs ────────────────────────────────────────────────────────
        self.tabs = QTabWidget()

        # Tab 1: Spectrum (FFT + Waterfall)
        tab_spec = QWidget()
        tv1 = QVBoxLayout(tab_spec)
        tv1.setContentsMargins(4, 4, 4, 4)
        tv1.setSpacing(6)
        self.spec_canvas = SpectrumCanvas(self.engine)
        tv1.addWidget(self.spec_canvas, 1)
        self.tabs.addTab(tab_spec, "📡  FFT + Waterfall")

        # Tab 2: Hop Map
        tab_hop = QWidget()
        tv2 = QVBoxLayout(tab_hop)
        tv2.setContentsMargins(4, 4, 4, 4)
        tv2.setSpacing(6)
        self.hop_canvas = HopMapCanvas(self.engine)
        tv2.addWidget(self.hop_canvas, 1)
        self.seq_widget = SequenceWidget(self.engine)
        tv2.addWidget(self.seq_widget)
        self.tabs.addTab(tab_hop, "📶  Hop Map")

        # Shared anim bar below tabs
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        lv.addWidget(self.tabs, 1)
        self.anim_bar = AnimBar(self.engine)
        self.anim_bar.hop_changed.connect(self._on_hop)
        lv.addWidget(self.anim_bar)

        main_split.addWidget(left_w)

        # ── Right: config + table ─────────────────────────────────────────────
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(8)

        self.ctrl = ControlPanel(self.engine)
        self.ctrl.changed.connect(self._on_config_changed)
        rv.addWidget(self.ctrl)

        ch_lbl = QLabel("Channel List")
        ch_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; font-weight:600;")
        rv.addWidget(ch_lbl)

        self.table = ChannelTable(self.engine)
        rv.addWidget(self.table)

        main_split.addWidget(right_w)
        main_split.setSizes([940, 370])
        root.addWidget(main_split, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_config_changed(self):
        self.spec_canvas.rebuild()
        self.hop_canvas._draw_static()
        self.stats.refresh()
        self.table.refresh()
        self.seq_widget.refresh()
        self.anim_bar.refresh(self.engine)

    def _on_hop(self, hop_index: int):
        self.spec_canvas.update_hop(hop_index)
        self.hop_canvas.highlight_hop(hop_index)
        if hop_index >= 0:
            ch_idx = self.engine.sequence[hop_index]
            self.table.highlight_channel(ch_idx)
            self.seq_widget.highlight_hop(hop_index)
            self.anim_bar._update_label()
        else:
            self.table.clearSelection()
            self.seq_widget.refresh()


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