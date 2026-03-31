"""
FHSS Simulation v4
==================
5 modulations: FSK, QPSK, FM, CSS (LoRa-like), RTCM/MSK
Message TX: type a message → encoded to bits → modulated → noisy channel → demodulated → compare
Noise + Jamming controls
Requires: PyQt5 matplotlib numpy scipy
"""

import sys, numpy as np, random
from datetime import datetime
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox,
    QSplitter, QFrame, QTabWidget, QTextEdit, QLineEdit, QCheckBox,
    QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Palette ───────────────────────────────────────────────────────────────────
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

CHANNEL_COLORS = [
    "#00d4ff","#ff6b35","#39d353","#f0e68c",
    "#da70d6","#87ceeb","#ff69b4","#98fb98",
    "#ffa500","#7b68ee","#20b2aa","#ff4500",
]

_wf_cmap = LinearSegmentedColormap.from_list(
    "waterfall",
    [(0,"#000000"),(0.25,"#0a2a6e"),(0.5,"#00d4ff"),
     (0.75,"#ffaa00"),(1.0,"#ffffff")]
)

MODULATIONS = ["FSK", "QPSK", "FM", "CSS (LoRa-like)", "RTCM/MSK"]


# ═════════════════════════════════════════════════════════════════════════════
# Message codec  —  text ↔ bits ↔ bytes
# ═════════════════════════════════════════════════════════════════════════════
class MsgCodec:
    @staticmethod
    def encode(text: str) -> np.ndarray:
        """UTF-8 → bit array (MSB first)."""
        raw = text.encode("utf-8")
        bits = []
        for byte in raw:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits, dtype=np.uint8)

    @staticmethod
    def decode(bits: np.ndarray) -> str:
        """Bit array → UTF-8 string. Returns '?' for undecodable bytes."""
        bits = np.asarray(bits, dtype=np.uint8)
        # Pad to multiple of 8
        rem = len(bits) % 8
        if rem:
            bits = np.concatenate([bits, np.zeros(8 - rem, dtype=np.uint8)])
        result = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for b in bits[i:i+8]:
                byte_val = (byte_val << 1) | int(b)
            result.append(byte_val)
        try:
            return bytes(result).decode("utf-8", errors="replace")
        except Exception:
            return "".join(chr(b) if 32 <= b < 127 else "?" for b in result)

    @staticmethod
    def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """Bit error rate between two bit arrays (pads shorter one with zeros)."""
        n = max(len(tx_bits), len(rx_bits))
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        a[:len(tx_bits)] = tx_bits
        b[:len(rx_bits)] = rx_bits
        return float(np.sum(a != b)) / n


# ═════════════════════════════════════════════════════════════════════════════
# Modulator  —  bits → IQ samples
# ═════════════════════════════════════════════════════════════════════════════
class Modulator:
    FS       = 8          # samples per bit/symbol
    N_PILOT  = 32         # pilot bits prepended for synchronisation

    @classmethod
    def modulate(cls, mod: str, bits: np.ndarray) -> np.ndarray:
        """Return complex baseband IQ for the given bit sequence."""
        bits = np.asarray(bits, dtype=np.uint8)
        # Prepend known pilot pattern for sync
        pilot = np.array([1,0]*16, dtype=np.uint8)
        bits  = np.concatenate([pilot, bits])

        sps = cls.FS          # samples per symbol / bit

        if mod == "FSK":
            return cls._fsk(bits, sps)
        elif mod == "QPSK":
            return cls._qpsk(bits, sps)
        elif mod == "FM":
            return cls._fm(bits, sps)
        elif mod == "CSS (LoRa-like)":
            return cls._css(bits, sps)
        elif mod == "RTCM/MSK":
            return cls._msk(bits, sps)
        else:
            return cls._fsk(bits, sps)

    # ── FSK ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _fsk(bits, sps):
        """Binary FSK h=1 (non-coherent friendly)."""
        f0, f1 = 0.15, 0.35          # normalized frequencies
        sig = []
        for b in bits:
            f  = f1 if b else f0
            t  = np.arange(sps)
            sig.extend(np.exp(1j * 2 * np.pi * f * t))
        return np.array(sig)

    # ── QPSK ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _qpsk(bits, sps):
        """Gray-coded QPSK, 2 bits/symbol."""
        gray = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}
        # Pad to even
        if len(bits) % 2:
            bits = np.concatenate([bits, [0]])
        sig = []
        for i in range(0, len(bits), 2):
            pair  = (int(bits[i]), int(bits[i+1]))
            idx   = gray[pair]
            angle = np.pi/4 + idx * np.pi/2
            sym   = np.exp(1j * angle)
            sig.extend([sym] * sps)
        return np.array(sig)

    # ── FM ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _fm(bits, sps):
        """FM: bit stream → NRZ → frequency-integrated to phase."""
        nrz   = (bits.astype(float) * 2 - 1)
        nrz_up = np.repeat(nrz, sps)
        # Smooth transitions (Gaussian filter)
        nrz_up = gaussian_filter1d(nrz_up, sigma=sps * 0.3)
        kf     = 0.25          # frequency sensitivity
        phase  = np.cumsum(2 * np.pi * kf * nrz_up)
        return np.exp(1j * phase)

    # ── CSS (LoRa-like) ───────────────────────────────────────────────────────
    @staticmethod
    def _css(bits, sps):
        """
        Simplified CSS: each bit selects an up-chirp (1) or down-chirp (0).
        Symbol duration = sps * 8 samples (wider than other mods for spread).
        """
        sym_len = sps * 8
        sig = []
        for b in bits:
            t = np.arange(sym_len) / sym_len
            if b:
                chirp = np.exp(1j * np.pi * t**2 * sym_len)
            else:
                chirp = np.exp(-1j * np.pi * t**2 * sym_len)
            sig.extend(chirp)
        return np.array(sig)

    # ── MSK (RTCM) ───────────────────────────────────────────────────────────
    @staticmethod
    def _msk(bits, sps):
        """Minimum Shift Keying h=0.5, continuous phase."""
        nrz   = (bits.astype(float) * 2 - 1)
        nrz_up = np.repeat(nrz, sps)
        phase = np.cumsum(nrz_up * np.pi / (2 * sps))
        return np.exp(1j * phase)


# ═════════════════════════════════════════════════════════════════════════════
# Channel  —  adds noise and jamming to IQ
# ═════════════════════════════════════════════════════════════════════════════
class Channel:
    @staticmethod
    def apply(iq: np.ndarray, snr_db: float, jam_db: float) -> np.ndarray:
        """
        Add AWGN at snr_db and a jammer at jam_db (relative to signal power).
        jam_db = -inf  →  no jamming.
        Jammer model: mix of narrowband CW tones + wideband barrage noise.
        """
        sig_pwr = np.mean(np.abs(iq)**2) + 1e-12
        n = len(iq)

        # AWGN
        noise_pwr = sig_pwr * 10**(-snr_db / 10)
        noise     = (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(noise_pwr / 2)

        # Jammer
        jam = np.zeros(n, dtype=complex)
        if jam_db > -60:
            jam_pwr = sig_pwr * 10**(jam_db / 10)
            # 50% CW tone jammer + 50% barrage noise
            t   = np.arange(n)
            cw  = np.exp(1j * 2 * np.pi * 0.25 * t)          # CW at 0.25 normalized
            cw += np.exp(1j * 2 * np.pi * 0.10 * t) * 0.5    # second tone
            cw  = cw / (np.max(np.abs(cw)) + 1e-12)
            barrage = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
            jam = (0.5 * cw + 0.5 * barrage) * np.sqrt(jam_pwr)

        return iq + noise + jam


# ═════════════════════════════════════════════════════════════════════════════
# Demodulator  —  IQ → bits
# ═════════════════════════════════════════════════════════════════════════════
class Demodulator:
    FS      = Modulator.FS
    N_PILOT = Modulator.N_PILOT

    @classmethod
    def demodulate(cls, mod: str, iq: np.ndarray, n_bits: int) -> np.ndarray:
        """
        Demodulate IQ samples → bit array of length n_bits.
        Strips pilot header first.
        """
        if mod == "FSK":
            bits = cls._fsk(iq)
        elif mod == "QPSK":
            bits = cls._qpsk(iq)
        elif mod == "FM":
            bits = cls._fm(iq)
        elif mod == "CSS (LoRa-like)":
            bits = cls._css(iq)
        elif mod == "RTCM/MSK":
            bits = cls._msk(iq)
        else:
            bits = cls._fsk(iq)

        # Strip pilot
        bits = bits[cls.N_PILOT:]
        # Trim/pad to expected length
        if len(bits) >= n_bits:
            return bits[:n_bits]
        return np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])

    # ── FSK ──────────────────────────────────────────────────────────────────
    @classmethod
    def _fsk(cls, iq):
        sps = cls.FS
        n_syms = len(iq) // sps
        bits = []
        f0, f1 = 0.15, 0.35
        t = np.arange(sps)
        ref0 = np.exp(1j * 2 * np.pi * f0 * t)
        ref1 = np.exp(1j * 2 * np.pi * f1 * t)
        for i in range(n_syms):
            seg    = iq[i*sps:(i+1)*sps]
            e0     = np.abs(np.dot(seg, ref0.conj()))
            e1     = np.abs(np.dot(seg, ref1.conj()))
            bits.append(1 if e1 > e0 else 0)
        return np.array(bits, dtype=np.uint8)

    # ── QPSK ─────────────────────────────────────────────────────────────────
    @classmethod
    def _qpsk(cls, iq):
        sps = cls.FS
        n_syms = len(iq) // sps
        gray_inv = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0)}
        bits = []
        for i in range(n_syms):
            seg   = iq[i*sps:(i+1)*sps]
            sym   = np.mean(seg)
            angle = np.angle(sym) % (2*np.pi)
            idx   = int(np.round((angle - np.pi/4) / (np.pi/2))) % 4
            bits.extend(gray_inv[idx])
        return np.array(bits, dtype=np.uint8)

    # ── FM ───────────────────────────────────────────────────────────────────
    @classmethod
    def _fm(cls, iq):
        sps   = cls.FS
        phase = np.unwrap(np.angle(iq))
        inst  = np.diff(phase) / (2 * np.pi * 0.25)   # normalise by kf
        # Average over each symbol
        n_syms = (len(inst)) // sps
        bits = []
        for i in range(n_syms):
            seg = inst[i*sps:(i+1)*sps]
            bits.append(1 if np.mean(seg) > 0 else 0)
        return np.array(bits, dtype=np.uint8)

    # ── CSS ───────────────────────────────────────────────────────────────────
    @classmethod
    def _css(cls, iq):
        sps     = cls.FS
        sym_len = sps * 8
        n_syms  = len(iq) // sym_len
        t       = np.arange(sym_len) / sym_len
        up      = np.exp( 1j * np.pi * t**2 * sym_len)
        dn      = np.exp(-1j * np.pi * t**2 * sym_len)
        bits    = []
        for i in range(n_syms):
            seg  = iq[i*sym_len:(i+1)*sym_len]
            eu   = np.abs(np.dot(seg, up.conj()))
            ed   = np.abs(np.dot(seg, dn.conj()))
            bits.append(1 if eu > ed else 0)
        return np.array(bits, dtype=np.uint8)

    # ── MSK ───────────────────────────────────────────────────────────────────
    @classmethod
    def _msk(cls, iq):
        sps   = cls.FS
        phase = np.unwrap(np.angle(iq))
        diff  = np.diff(phase)
        n_syms = len(diff) // sps
        bits  = []
        for i in range(n_syms):
            seg = diff[i*sps:(i+1)*sps]
            bits.append(1 if np.mean(seg) > 0 else 0)
        return np.array(bits, dtype=np.uint8)

    @staticmethod
    def measure_snr(iq: np.ndarray) -> float:
        mag2 = np.abs(iq)**2
        mu   = np.mean(mag2)
        var  = np.var(mag2)
        if var < 1e-12:
            return 40.0
        snr_lin = mu**2 / (var + 1e-12)
        return float(np.clip(10 * np.log10(snr_lin + 1e-12), -10, 50))


# ═════════════════════════════════════════════════════════════════════════════
# FHSS Engine
# ═════════════════════════════════════════════════════════════════════════════
class FHSSEngine:
    def __init__(self):
        self.num_channels  = 8
        self.base_freq     = 2400.0
        self.channel_bw    = 1.0
        self.hop_interval  = 10
        self.sequence_len  = 32
        self.seed          = 42
        self.modulation    = "QPSK"
        self.snr_db        = 20.0
        self.jam_db        = -60.0
        self._build()

    def _build(self):
        self.channels = [
            {"id": i, "freq": round(self.base_freq + i * self.channel_bw, 2),
             "label": f"CH{i+1}", "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)]}
            for i in range(self.num_channels)
        ]
        rng = random.Random(self.seed)
        self.sequence = [rng.randint(0, self.num_channels-1) for _ in range(self.sequence_len)]

    def reconfigure(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        self._build()

    @property
    def time_axis(self): return [i * self.hop_interval for i in range(self.sequence_len)]
    @property
    def freq_axis(self): return [self.channels[ch]["freq"] for ch in self.sequence]
    @property
    def total_bw(self):  return self.num_channels * self.channel_bw
    @property
    def span_start(self): return self.base_freq - self.channel_bw * 0.5
    @property
    def span_end(self):   return self.base_freq + self.num_channels * self.channel_bw + self.channel_bw * 0.5


# ═════════════════════════════════════════════════════════════════════════════
# Spectrum Canvas (FFT + Waterfall)
# ═════════════════════════════════════════════════════════════════════════════
class SpectrumCanvas(FigureCanvas):
    N_FFT       = 512
    WATERFALL_H = 80
    NOISE_FLOOR = -90.0
    PEAK_POWER  = -20.0

    iq_ready = pyqtSignal(object, int, int)   # iq, hop_idx, ch_idx

    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine     = engine
        self._wf_data   = np.full((self.WATERFALL_H, self.N_FFT), self.NOISE_FLOOR)
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)
        self.fig        = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._build_axes()
        self._init_artists()

    def _build_axes(self):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[2, 1.2],
                               hspace=0.08, left=0.08, right=0.97, top=0.93, bottom=0.08)
        self.ax_fft = self.fig.add_subplot(gs[0])
        self.ax_wf  = self.fig.add_subplot(gs[1], sharex=self.ax_fft)
        for ax in (self.ax_fft, self.ax_wf):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        self.fig.patch.set_facecolor(BG)
        self.ax_fft.set_title("Live Spectrum + Waterfall", color=TEXT, fontsize=10, fontweight="bold", pad=6)
        self.ax_fft.set_ylabel("Power (dBm)", color=MUTED, fontsize=8)
        self.ax_fft.set_ylim(self.NOISE_FLOOR, self.PEAK_POWER + 15)
        self.ax_fft.grid(color=BORDER, linewidth=0.4, alpha=0.6)
        self.ax_wf.set_xlabel("Frequency (MHz)", color=MUTED, fontsize=8)
        self.ax_wf.set_ylabel("Time →", color=MUTED, fontsize=7)
        self.ax_fft.tick_params(labelbottom=False)

    def _init_artists(self):
        eng  = self.engine
        freq = self._freq_axis()
        self._fft_line,  = self.ax_fft.plot(freq, np.full(self.N_FFT, self.NOISE_FLOOR),
                                             color=ACCENT, lw=1.0, alpha=0.9, zorder=5)
        self._peak_line, = self.ax_fft.plot(freq, np.full(self.N_FFT, self.NOISE_FLOOR),
                                             color=ACCENT2, lw=0.6, alpha=0.5, ls="--", zorder=4)
        self._jam_line,  = self.ax_fft.plot(freq, np.full(self.N_FFT, self.NOISE_FLOOR),
                                             color=RED, lw=0.8, alpha=0.0, zorder=3)   # hidden until jam on
        for ch in eng.channels:
            self.ax_fft.axvspan(ch["freq"] - eng.channel_bw*0.5,
                                ch["freq"] + eng.channel_bw*0.5,
                                alpha=0.07, color=ch["color"], zorder=1)
        self._active_span = self.ax_fft.axvspan(
            eng.channels[0]["freq"] - eng.channel_bw*0.5,
            eng.channels[0]["freq"] + eng.channel_bw*0.5,
            alpha=0.25, color=ACCENT, zorder=2)
        self._info = self.ax_fft.text(0.01, 0.95, "", transform=self.ax_fft.transAxes,
            color=GREEN, fontsize=8, fontweight="bold", va="top", fontfamily="monospace")
        self._wf_im = self.ax_wf.imshow(self._wf_data, aspect="auto", origin="upper",
            cmap=_wf_cmap, vmin=self.NOISE_FLOOR, vmax=self.PEAK_POWER+10,
            extent=[freq[0], freq[-1], self.WATERFALL_H, 0], interpolation="bilinear", zorder=2)
        self._set_xlim()
        self.draw()

    def _freq_axis(self):
        e = self.engine
        return np.linspace(e.span_start, e.span_end, self.N_FFT)

    def _set_xlim(self):
        e = self.engine
        self.ax_fft.set_xlim(e.span_start, e.span_end)
        xt = [c["freq"] for c in e.channels]
        self.ax_fft.set_xticks(xt)
        self.ax_fft.set_xticklabels([f"{x:.1f}" for x in xt], fontsize=6.5, color=MUTED, rotation=30)

    def update_hop(self, hop_idx: int, tx_iq: np.ndarray = None):
        """
        tx_iq: if provided, use real modulated IQ for FFT display.
               Otherwise generate a spectral-shape approximation.
        """
        if hop_idx < 0: return
        eng    = self.engine
        ch_idx = eng.sequence[hop_idx % len(eng.sequence)]
        ch     = eng.channels[ch_idx]
        freq   = self._freq_axis()

        if tx_iq is not None and len(tx_iq) > 0:
            # Real IQ → FFT frame
            n   = min(len(tx_iq), self.N_FFT)
            win = np.hanning(n)
            fft = np.fft.fft(tx_iq[:n] * win, n=n)
            psd = np.abs(np.fft.fftshift(fft))**2
            psd = psd / (n * np.sum(win**2) + 1e-12)
            psd_db = 10*np.log10(psd + 1e-12) + self.PEAK_POWER + 18
            iq_freqs = np.linspace(ch["freq"] - eng.channel_bw*0.5,
                                   ch["freq"] + eng.channel_bw*0.5, n)
            frame = np.interp(freq, iq_freqs, psd_db,
                              left=self.NOISE_FLOOR, right=self.NOISE_FLOOR)
        else:
            frame = np.full(self.N_FFT, self.NOISE_FLOOR)

        # Noise floor for out-of-band bins
        mask = (freq < ch["freq"] - eng.channel_bw*0.55) | (freq > ch["freq"] + eng.channel_bw*0.55)
        frame[mask] = self.NOISE_FLOOR + np.random.randn(mask.sum()) * 1.8

        # Jammer overlay
        jam_visible = False
        if eng.jam_db > -50:
            jam_visible = True
            jam_frame = np.full(self.N_FFT, self.NOISE_FLOOR - 10)
            # CW tone spikes
            for f_tone in [0.25, 0.10]:
                tone_freq = eng.span_start + f_tone * (eng.span_end - eng.span_start)
                idx = np.argmin(np.abs(freq - tone_freq))
                jam_frame[max(0,idx-3):idx+4] = self.NOISE_FLOOR + eng.jam_db * 0.8 + 10
            # Barrage raises floor
            jam_frame += eng.jam_db * 0.15 + np.random.randn(self.N_FFT) * 2
            self._jam_line.set_ydata(jam_frame)
            self._jam_line.set_alpha(0.55)
        else:
            self._jam_line.set_alpha(0.0)

        frame = uniform_filter1d(frame, size=3)
        self._peak_hold = np.maximum(self._peak_hold, frame) * 0.97 + self.NOISE_FLOOR * 0.03
        self._fft_line.set_ydata(frame)
        self._peak_line.set_ydata(self._peak_hold)

        x0, x1 = ch["freq"] - eng.channel_bw*0.5, ch["freq"] + eng.channel_bw*0.5
        self._active_span.remove()
        self._active_span = self.ax_fft.axvspan(x0, x1, alpha=0.25, color=ch["color"], zorder=2)

        jam_str = f"  JAM={eng.jam_db:+.0f}dB" if eng.jam_db > -50 else ""
        self._info.set_text(f"MOD:{eng.modulation}  SNR={eng.snr_db:.0f}dB{jam_str}  {ch['label']} {ch['freq']:.2f}MHz")

        self._wf_data = np.roll(self._wf_data, -1, axis=0)
        self._wf_data[-1, :] = frame
        self._wf_im.set_data(self._wf_data)
        self._wf_im.set_extent([freq[0], freq[-1], self.WATERFALL_H, 0])
        self.draw_idle()

    def rebuild(self):
        self._wf_data   = np.full((self.WATERFALL_H, self.N_FFT), self.NOISE_FLOOR)
        self._peak_hold = np.full(self.N_FFT, self.NOISE_FLOOR)
        self._build_axes()
        self._init_artists()


# ═════════════════════════════════════════════════════════════════════════════
# Hop Map Canvas
# ═════════════════════════════════════════════════════════════════════════════
class HopMapCanvas(FigureCanvas):
    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine = engine
        self.fig    = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._hp = None
        self._draw()

    def _draw(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        eng = self.engine
        self.ax.set_facecolor(BG); self.fig.patch.set_facecolor(BG)
        times = eng.time_axis; freqs = eng.freq_axis
        bw    = eng.channel_bw
        for ch in eng.channels:
            self.ax.axhspan(ch["freq"]-bw*0.45, ch["freq"]+bw*0.45, alpha=0.06, color=ch["color"])
            self.ax.axhline(ch["freq"], color=ch["color"], alpha=0.15, lw=0.5, ls="--")
        for i in range(len(times)):
            ch = eng.channels[eng.sequence[i]]
            r  = mpatches.FancyBboxPatch((times[i], freqs[i]-bw*0.4), eng.hop_interval*0.92, bw*0.8,
                    boxstyle="round,pad=0.01", lw=1.1, edgecolor=ch["color"], facecolor=ch["color"]+"35")
            self.ax.add_patch(r)
            self.ax.text(times[i]+eng.hop_interval*0.46, freqs[i], ch["label"],
                         ha="center", va="center", fontsize=6, color=ch["color"], fontweight="bold")
        self.ax.plot([t+eng.hop_interval*0.46 for t in times], freqs,
                     color=ACCENT, lw=0.8, alpha=0.3, ls=":")
        fv = [c["freq"] for c in eng.channels]
        self.ax.set_xlim(-eng.hop_interval*0.5, times[-1]+eng.hop_interval*1.5)
        self.ax.set_ylim(min(fv)-bw, max(fv)+bw)
        self.ax.set_xlabel("Time (ms)", color=TEXT, fontsize=8)
        self.ax.set_ylabel("Frequency (MHz)", color=TEXT, fontsize=8)
        self.ax.set_title("Hop Map", color=TEXT, fontsize=10, fontweight="bold", pad=6)
        self.ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in self.ax.spines.values(): sp.set_edgecolor(BORDER)
        self.ax.set_yticks([c["freq"] for c in eng.channels])
        self.ax.set_yticklabels([f"{c['freq']:.1f}" for c in eng.channels], fontsize=7, color=MUTED)
        self.ax.grid(color=BORDER, lw=0.35, alpha=0.5)
        self._hp = None
        self.fig.tight_layout(pad=1.0)
        self.draw()

    def highlight_hop(self, idx: int):
        eng = self.engine
        if self._hp:
            try: self._hp.remove()
            except: pass
            self._hp = None
        if 0 <= idx < len(eng.time_axis):
            t, f = eng.time_axis[idx], eng.freq_axis[idx]
            r = mpatches.FancyBboxPatch((t, f-eng.channel_bw*0.48),
                    eng.hop_interval*0.92, eng.channel_bw*0.96,
                    boxstyle="round,pad=0.01", lw=2.2, edgecolor="#fff",
                    facecolor="#ffffff1a", zorder=10)
            self.ax.add_patch(r); self._hp = r
        self.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
# Demod Console
# ═════════════════════════════════════════════════════════════════════════════
class DemodConsole(QWidget):
    def __init__(self):
        super().__init__()
        self._paused = False
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        # Toolbar
        tb = QWidget()
        tl = QHBoxLayout(tb); tl.setContentsMargins(0,0,0,0); tl.setSpacing(6)
        title = QLabel("RX Console")
        title.setStyleSheet(f"color:{ACCENT}; font-size:11px; font-weight:bold; font-family:'Courier New';")
        tl.addWidget(title); tl.addStretch()

        self.cb_pause = QCheckBox("Pause")
        self.cb_pause.toggled.connect(lambda v: setattr(self, '_paused', v))
        self.cb_pause.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        tl.addWidget(self.cb_pause)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(52)
        btn_clear.clicked.connect(self._clear)
        btn_clear.setStyleSheet(f"""QPushButton {{background:{PANEL};color:{MUTED};
            border:1px solid {BORDER};border-radius:3px;padding:2px 6px;font-size:9px;}}
            QPushButton:hover{{color:{TEXT};}}""")
        tl.addWidget(btn_clear)
        lay.addWidget(tb)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Courier New", 9))
        self.console.setStyleSheet(f"""QTextEdit{{background:#080c10;color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:4px;}}""")
        self.console.setLineWrapMode(QTextEdit.NoWrap)
        lay.addWidget(self.console)

        self.status = QLabel("Waiting for transmissions...")
        self.status.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        lay.addWidget(self.status)

        self._banner()

    def _banner(self):
        self.console.setTextColor(QColor(MUTED))
        self.console.append("┌────────────────────────────────────────────────────────────────────┐")
        self.console.append("│  FHSS RX Console — live demodulation from simulated spectrum        │")
        self.console.append("│  TX message → modulate → channel (noise+jam) → demod → compare     │")
        self.console.append("└────────────────────────────────────────────────────────────────────┘")
        self.console.append("")

    def _clear(self):
        self.console.clear()
        self._banner()

    def _put(self, color, text, nl=False):
        self.console.setTextColor(QColor(color))
        if nl:
            self.console.append(text)
        else:
            self.console.insertPlainText(text)

    def log_transmission(self, mod: str, tx_msg: str, rx_msg: str,
                         tx_bits: np.ndarray, rx_bits: np.ndarray,
                         snr_db: float, jam_db: float,
                         snr_meas: float, hop_idx: int, ch_label: str, ch_freq: float):
        if self._paused:
            return

        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ber = MsgCodec.ber(tx_bits, rx_bits)
        ok  = tx_msg.strip() == rx_msg.strip()
        match_sym = "✓ MATCH" if ok else "✗ MISMATCH"
        match_col = GREEN if ok else RED

        self.console.append("")

        # Header
        self._put(MUTED, f"[{ts}] ")
        self._put(match_col, f"[{match_sym}] ")
        self._put(ACCENT, f"HOP#{hop_idx+1:02d} ")
        self._put(YELLOW, f"{ch_label} {ch_freq:.2f}MHz  ")
        self._put(PURPLE, f"{mod}  ")
        self._put(MUTED,  f"SNR={snr_meas:+.1f}dB (cfg:{snr_db:.0f})  ")
        if jam_db > -50:
            self._put(RED, f"JAM={jam_db:+.0f}dB  ")
        self._put(MUTED, f"BER={ber:.3f}")

        # TX message
        self.console.append("")
        self._put(MUTED, "  TX: ")
        self._put(GREEN, repr(tx_msg))

        # RX message
        self.console.append("")
        self._put(MUTED, "  RX: ")
        self._put(TEXT if ok else RED, repr(rx_msg))

        # Bit diff line (first 48 bits)
        n_show = min(48, len(tx_bits), len(rx_bits))
        if n_show > 0:
            self.console.append("")
            self._put(MUTED, "  TX bits[0:48]: ")
            self._put("#4fc3f7", "".join(str(b) for b in tx_bits[:n_show]))
            self.console.append("")
            self._put(MUTED, "  RX bits[0:48]: ")
            # Colour each bit — green if correct, red if error
            for i in range(n_show):
                tb, rb = int(tx_bits[i]), int(rx_bits[i])
                col = "#4fc3f7" if tb == rb else RED
                self._put(col, str(rb))
            if ber > 0:
                n_err = int(round(ber * max(len(tx_bits), len(rx_bits))))
                self._put(RED, f"  ← {n_err} bit errors")

        # Scroll to bottom
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Trim document if huge
        doc = self.console.document()
        if doc.lineCount() > 1200:
            cur = QTextCursor(doc)
            cur.movePosition(QTextCursor.Start)
            cur.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 60)
            cur.removeSelectedText()

        # Status bar
        jam_str = f"  JAM={jam_db:+.0f}dB" if jam_db > -50 else ""
        self.status.setText(
            f"{'✓ OK' if ok else '✗ ERR'}  BER={ber:.3f}  SNR_meas={snr_meas:+.1f}dB{jam_str}  {mod}")
        self.status.setStyleSheet(
            f"color:{'#39d353' if ok else '#f85149'}; font-size:9px;")

    def log_hop_only(self, hop_idx: int, ch_label: str, ch_freq: float,
                     mod: str, snr_db: float, jam_db: float, snr_meas: float):
        """Log a background hop (no user message queued)."""
        if self._paused: return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.console.append("")
        self._put(MUTED, f"[{ts}] ")
        self._put(BORDER, f"HOP#{hop_idx+1:02d} ")
        self._put(MUTED,  f"{ch_label} {ch_freq:.2f}MHz  {mod}  SNR={snr_meas:+.1f}dB")
        if jam_db > -50:
            self._put(RED, f"  JAM={jam_db:+.0f}dB")
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═════════════════════════════════════════════════════════════════════════════
# TX Panel (right side)
# ═════════════════════════════════════════════════════════════════════════════
class TXPanel(QGroupBox):
    transmit_requested = pyqtSignal(str, str)  # mod, message

    def __init__(self, engine: FHSSEngine):
        super().__init__("Transmit")
        self.engine = engine
        self.setStyleSheet(f"""
            QGroupBox {{color:{TEXT};border:1px solid {BORDER};border-radius:8px;
                font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title {{subcontrol-origin:margin;left:10px;color:{ACCENT};}}
            QLabel {{color:{MUTED};font-size:10px;}}
        """)
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # Modulation
        ml = QHBoxLayout()
        ml.addWidget(QLabel("Modulation"))
        self.cb_mod = QComboBox()
        self.cb_mod.addItems(MODULATIONS)
        self.cb_mod.setCurrentText("QPSK")
        self.cb_mod.setStyleSheet(f"""QComboBox{{background:{BG};color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:4px 8px;font-size:11px;}}
            QComboBox QAbstractItemView{{background:{PANEL};color:{TEXT};
            selection-background-color:{ACCENT}40;}}""")
        ml.addWidget(self.cb_mod, 1)
        lay.addLayout(ml)

        # Message input
        lay.addWidget(QLabel("Message to send"))
        self.msg_input = QLineEdit()
        self.msg_input.setPlaceholderText("Type your message here…")
        self.msg_input.setText("Hello FHSS!")
        self.msg_input.setStyleSheet(f"""QLineEdit{{background:{BG};color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:5px 8px;font-size:11px;
            font-family:'Courier New';}}
            QLineEdit:focus{{border:1px solid {ACCENT};}}""")
        self.msg_input.returnPressed.connect(self._send)
        lay.addWidget(self.msg_input)

        # Bit count info
        self.bit_info = QLabel("0 chars → 0 bits")
        self.bit_info.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        self.msg_input.textChanged.connect(self._update_bit_info)
        lay.addWidget(self.bit_info)
        self._update_bit_info(self.msg_input.text())

        # Send button
        self.btn_send = QPushButton("▶  Send on Next Hop")
        self.btn_send.setStyleSheet(f"""QPushButton{{background:{ACCENT};color:#000;
            border:none;border-radius:6px;padding:7px;font-size:11px;font-weight:bold;}}
            QPushButton:hover{{background:#33ddff;}}
            QPushButton:disabled{{background:{BORDER};color:{MUTED};}}""")
        self.btn_send.clicked.connect(self._send)
        lay.addWidget(self.btn_send)

        # Pending indicator
        self.pending_lbl = QLabel("")
        self.pending_lbl.setStyleSheet(f"color:{YELLOW}; font-size:9px; font-family:'Courier New';")
        lay.addWidget(self.pending_lbl)

        self._pending_msg = None

    def _update_bit_info(self, text):
        n_chars = len(text.encode("utf-8"))
        n_bits  = n_chars * 8
        self.bit_info.setText(f"{n_chars} chars → {n_bits} bits")

    def _send(self):
        msg = self.msg_input.text()
        if not msg:
            return
        mod = self.cb_mod.currentText()
        self._pending_msg = (mod, msg)
        self.pending_lbl.setText(f"⏳ Queued: [{mod}] {repr(msg)}")
        self.btn_send.setEnabled(False)

    def pop_pending(self):
        """Called by main window on each hop. Returns (mod, msg) or None."""
        if self._pending_msg:
            v = self._pending_msg
            self._pending_msg = None
            self.pending_lbl.setText("")
            self.btn_send.setEnabled(True)
            return v
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Noise / Jamming Panel
# ═════════════════════════════════════════════════════════════════════════════
class ChannelPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, engine: FHSSEngine):
        super().__init__("Channel")
        self.engine = engine
        self.setStyleSheet(f"""QGroupBox{{color:{TEXT};border:1px solid {BORDER};
            border-radius:8px;font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{ACCENT};}}
            QLabel{{color:{MUTED};font-size:10px;}}""")
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        def slider_row(label, lo, hi, val, suffix, color, callback):
            row = QWidget()
            rl  = QVBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(2)
            top = QWidget()
            tl  = QHBoxLayout(top); tl.setContentsMargins(0,0,0,0)
            lbl = QLabel(label)
            val_lbl = QLabel(f"{val}{suffix}")
            val_lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold;")
            tl.addWidget(lbl); tl.addStretch(); tl.addWidget(val_lbl)
            rl.addWidget(top)
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi)
            sl.setValue(val)
            sl.setStyleSheet(f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
                QSlider::handle:horizontal{{background:{color};width:14px;height:14px;
                    margin:-5px 0;border-radius:7px;}}
                QSlider::sub-page:horizontal{{background:{color}60;border-radius:2px;}}""")
            def _cb(v, lbl=val_lbl, sfx=suffix, cb=callback):
                lbl.setText(f"{v}{sfx}")
                cb(v)
            sl.valueChanged.connect(_cb)
            rl.addWidget(sl)
            lay.addWidget(row)
            return sl

        # SNR
        self.sl_snr = slider_row("Noise (SNR)", 0, 40, int(engine.snr_db), " dB", GREEN,
                                  lambda v: (setattr(engine, 'snr_db', float(v)), self.changed.emit()))

        # Jamming
        jam_init = max(-60, int(engine.jam_db))
        self.sl_jam = slider_row("Jamming power", -60, 20, jam_init, " dB", RED,
                                  lambda v: (setattr(engine, 'jam_db', float(v)), self.changed.emit()))

        # Jam type indicator
        self.jam_lbl = QLabel("Jammer: OFF")
        self.jam_lbl.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        lay.addWidget(self.jam_lbl)
        self.sl_jam.valueChanged.connect(self._update_jam_label)
        self._update_jam_label(jam_init)

    def _update_jam_label(self, v):
        if v <= -50:
            self.jam_lbl.setText("Jammer: OFF")
            self.jam_lbl.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        elif v <= -20:
            self.jam_lbl.setText("Jammer: WEAK  (CW tone + barrage)")
            self.jam_lbl.setStyleSheet(f"color:{YELLOW}; font-size:9px;")
        elif v <= 0:
            self.jam_lbl.setText("Jammer: MEDIUM  (CW tone + barrage)")
            self.jam_lbl.setStyleSheet(f"color:{ACCENT2}; font-size:9px;")
        else:
            self.jam_lbl.setText("Jammer: STRONG — expect high BER")
            self.jam_lbl.setStyleSheet(f"color:{RED}; font-size:9px; font-weight:bold;")


# ═════════════════════════════════════════════════════════════════════════════
# Hop Settings Panel
# ═════════════════════════════════════════════════════════════════════════════
class HopSettingsPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, engine: FHSSEngine):
        super().__init__("FHSS Settings")
        self.engine = engine
        self.setStyleSheet(f"""QGroupBox{{color:{TEXT};border:1px solid {BORDER};
            border-radius:8px;font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{ACCENT};}}
            QLabel{{color:{MUTED};font-size:10px;}}
            QSpinBox,QDoubleSpinBox{{background:{BG};color:{TEXT};border:1px solid {BORDER};
            border-radius:4px;padding:3px 5px;font-size:10px;}}""")
        lay = QVBoxLayout(self)
        lay.setSpacing(5)

        def row(lbl, widget):
            r  = QWidget(); rl = QHBoxLayout(r); rl.setContentsMargins(0,0,0,0)
            lb = QLabel(lbl); lb.setFixedWidth(130)
            rl.addWidget(lb); rl.addWidget(widget, 1)
            lay.addWidget(r); return widget

        self.sb_ch = row("Channels", QSpinBox())
        self.sb_ch.setRange(2, 12); self.sb_ch.setValue(engine.num_channels)

        self.dsb_bf = row("Base Freq (MHz)", QDoubleSpinBox())
        self.dsb_bf.setRange(100, 6000); self.dsb_bf.setValue(engine.base_freq); self.dsb_bf.setSingleStep(10)

        self.dsb_bw = row("Ch BW (MHz)", QDoubleSpinBox())
        self.dsb_bw.setRange(0.1, 20); self.dsb_bw.setValue(engine.channel_bw); self.dsb_bw.setSingleStep(0.5)

        self.sb_hi = row("Hop Interval (ms)", QSpinBox())
        self.sb_hi.setRange(1, 500); self.sb_hi.setValue(engine.hop_interval)

        self.sb_sl = row("Sequence Length", QSpinBox())
        self.sb_sl.setRange(4, 64); self.sb_sl.setValue(engine.sequence_len)

        self.sb_sd = row("Seed", QSpinBox())
        self.sb_sd.setRange(0, 9999); self.sb_sd.setValue(engine.seed)

        btn = QPushButton("Apply")
        btn.setStyleSheet(f"""QPushButton{{background:{PANEL};color:{ACCENT};border:1px solid {ACCENT}60;
            border-radius:5px;padding:5px;font-size:10px;font-weight:bold;}}
            QPushButton:hover{{background:{HOVER};}}""")
        btn.clicked.connect(self._apply)
        lay.addWidget(btn)

    def _apply(self):
        self.engine.reconfigure(
            num_channels  = self.sb_ch.value(),
            base_freq     = self.dsb_bf.value(),
            channel_bw    = self.dsb_bw.value(),
            hop_interval  = self.sb_hi.value(),
            sequence_len  = self.sb_sl.value(),
            seed          = self.sb_sd.value(),
        )
        self.changed.emit()


# ═════════════════════════════════════════════════════════════════════════════
# Transport Bar
# ═════════════════════════════════════════════════════════════════════════════
class AnimBar(QWidget):
    hop_changed = pyqtSignal(int)

    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine   = engine
        self._hop     = 0
        self._playing = False
        self._timer   = QTimer()
        self._timer.timeout.connect(self._tick)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self.btn_play  = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(82)
        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.setFixedWidth(82)
        self.btn_play.clicked.connect(self._toggle)
        self.btn_reset.clicked.connect(self._reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, engine.sequence_len-1)
        self.slider.valueChanged.connect(self._seek)

        self.hop_lbl = QLabel("Hop 0/0")
        self.hop_lbl.setFixedWidth(80)
        self.hop_lbl.setStyleSheet(f"color:{TEXT}; font-size:10px;")

        self.speed_sb = QSpinBox()
        self.speed_sb.setRange(100, 3000); self.speed_sb.setValue(600)
        self.speed_sb.setSuffix(" ms"); self.speed_sb.setFixedWidth(82)
        self.speed_sb.setStyleSheet(f"QSpinBox{{background:{PANEL};color:{TEXT};border:1px solid {BORDER};border-radius:4px;padding:2px 4px;font-size:10px;}}")

        btn_style = f"""QPushButton{{background:{PANEL};color:{TEXT};border:1px solid {BORDER};
            border-radius:4px;padding:4px 8px;font-size:10px;}}
            QPushButton:hover{{background:{HOVER};}}"""
        sl_style  = f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
            QSlider::handle:horizontal{{background:{ACCENT};width:13px;height:13px;margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{ACCENT}40;border-radius:2px;}}"""
        self.btn_play.setStyleSheet(btn_style)
        self.btn_reset.setStyleSheet(btn_style)
        self.slider.setStyleSheet(sl_style)

        spd_lbl = QLabel("Speed:"); spd_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(self.btn_play); lay.addWidget(self.btn_reset)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.hop_lbl)
        lay.addWidget(spd_lbl); lay.addWidget(self.speed_sb)

    def refresh(self, engine: FHSSEngine):
        self.engine = engine; self._hop = 0
        self.slider.setRange(0, engine.sequence_len-1); self.slider.setValue(0)
        self._lbl()

    def _toggle(self):
        self._playing = not self._playing
        if self._playing:
            self._timer.start(self.speed_sb.value()); self.btn_play.setText("⏸  Pause")
        else:
            self._timer.stop(); self.btn_play.setText("▶  Play")

    def _reset(self):
        self._playing = False; self._timer.stop()
        self.btn_play.setText("▶  Play"); self._hop = 0
        self.slider.setValue(0); self.hop_changed.emit(-1)

    def _tick(self):
        self._timer.setInterval(self.speed_sb.value())
        self._hop = (self._hop + 1) % self.engine.sequence_len
        self.slider.setValue(self._hop)

    def _seek(self, v):
        self._hop = v; self._lbl(); self.hop_changed.emit(v)

    def _lbl(self):
        self.hop_lbl.setText(f"Hop {self._hop+1}/{self.engine.sequence_len}")


# ═════════════════════════════════════════════════════════════════════════════
# Stats Bar
# ═════════════════════════════════════════════════════════════════════════════
class StatsBar(QWidget):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        self._lbs = {}
        for key in ["Channels","Hop Rate","Total BW","Modulation","SNR","Jamming"]:
            fr = QFrame()
            fr.setStyleSheet(f"background:{PANEL};border:1px solid {BORDER};border-radius:5px;")
            fl = QVBoxLayout(fr); fl.setContentsMargins(8,3,8,3); fl.setSpacing(1)
            k  = QLabel(key); k.setStyleSheet(f"color:{MUTED};font-size:9px;")
            v  = QLabel("—"); v.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;")
            fl.addWidget(k); fl.addWidget(v)
            lay.addWidget(fr)
            self._lbs[key] = v
        lay.addStretch()
        self.refresh()

    def refresh(self):
        e = self.engine
        self._lbs["Channels"].setText(str(e.num_channels))
        self._lbs["Hop Rate"].setText(f"{1000/e.hop_interval:.0f} h/s")
        self._lbs["Total BW"].setText(f"{e.total_bw:.1f} MHz")
        self._lbs["Modulation"].setText(e.modulation)
        self._lbs["SNR"].setText(f"{e.snr_db:.0f} dB")
        jam = f"{e.jam_db:.0f} dB" if e.jam_db > -50 else "OFF"
        self._lbs["Jamming"].setText(jam)
        self._lbs["Jamming"].setStyleSheet(
            f"color:{'#f85149' if e.jam_db > -50 else ACCENT}; font-size:12px; font-weight:bold;")


# ═════════════════════════════════════════════════════════════════════════════
# Main Window
# ═════════════════════════════════════════════════════════════════════════════
class FHSSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FHSS Simulation v4  —  TX/RX Message + Noise + Jamming")
        self.resize(1440, 880)
        self.engine        = FHSSEngine()
        self._pending_tx   = None    # (mod, msg, tx_bits, tx_iq) ready for next hop
        self._build_ui()
        self._style()

    def _style(self):
        self.setStyleSheet(f"""
            QMainWindow,QWidget{{background:{BG};color:{TEXT};}}
            QTabWidget::pane{{border:1px solid {BORDER};background:{PANEL};}}
            QTabBar::tab{{background:{BG};color:{MUTED};padding:6px 16px;
                border:1px solid {BORDER};border-bottom:none;font-size:11px;}}
            QTabBar::tab:selected{{background:{PANEL};color:{TEXT};border-top:2px solid {ACCENT};}}
            QSplitter::handle{{background:{BORDER};}}
            QScrollBar:vertical{{background:{BG};width:7px;}}
            QScrollBar::handle:vertical{{background:{BORDER};border-radius:3px;}}
            QCheckBox{{color:{MUTED};font-size:10px;}}
            QCheckBox::indicator{{width:12px;height:12px;border:1px solid {BORDER};border-radius:2px;background:{BG};}}
            QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
        """)

    def _build_ui(self):
        cw  = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setContentsMargins(10,8,10,8); root.setSpacing(7)

        hdr = QLabel("⟁  FHSS Simulation  ·  Message TX/RX  ·  Noise + Jamming")
        hdr.setStyleSheet(f"color:{ACCENT};font-size:15px;font-weight:700;"
                          f"font-family:'Courier New';letter-spacing:1.5px;"
                          f"border-bottom:1px solid {BORDER};padding-bottom:5px;")
        root.addWidget(hdr)

        self.stats = StatsBar(self.engine)
        root.addWidget(self.stats)

        # Main splitter: left (viz + console) | right (tx + channel + hop settings)
        msplit = QSplitter(Qt.Horizontal); msplit.setHandleWidth(2)

        # ── LEFT ─────────────────────────────────────────────────────────────
        lvsplit = QSplitter(Qt.Vertical); lvsplit.setHandleWidth(3)

        # Upper: tabs
        self.tabs = QTabWidget()

        t1 = QWidget(); tl1 = QVBoxLayout(t1); tl1.setContentsMargins(4,4,4,4)
        self.spec = SpectrumCanvas(self.engine)
        tl1.addWidget(self.spec)
        self.tabs.addTab(t1, "📡  FFT + Waterfall")

        t2 = QWidget(); tl2 = QVBoxLayout(t2); tl2.setContentsMargins(4,4,4,4)
        self.hop_map = HopMapCanvas(self.engine)
        tl2.addWidget(self.hop_map)
        self.tabs.addTab(t2, "📶  Hop Map")

        upper = QWidget(); uv = QVBoxLayout(upper); uv.setContentsMargins(0,0,0,0); uv.setSpacing(4)
        uv.addWidget(self.tabs, 1)
        self.anim = AnimBar(self.engine)
        self.anim.hop_changed.connect(self._on_hop)
        uv.addWidget(self.anim)
        lvsplit.addWidget(upper)

        # Lower: console
        self.console = DemodConsole()
        lvsplit.addWidget(self.console)
        lvsplit.setSizes([540, 260])
        msplit.addWidget(lvsplit)

        # ── RIGHT ─────────────────────────────────────────────────────────────
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(6,0,0,0); rv.setSpacing(8)

        self.tx_panel = TXPanel(self.engine)
        rv.addWidget(self.tx_panel)

        self.ch_panel = ChannelPanel(self.engine)
        self.ch_panel.changed.connect(self.stats.refresh)
        rv.addWidget(self.ch_panel)

        self.hop_panel = HopSettingsPanel(self.engine)
        self.hop_panel.changed.connect(self._on_config_changed)
        rv.addWidget(self.hop_panel)

        rv.addStretch()
        msplit.addWidget(right)
        msplit.setSizes([980, 380])
        root.addWidget(msplit, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_config_changed(self):
        self.spec.rebuild()
        self.hop_map._draw()
        self.stats.refresh()
        self.anim.refresh(self.engine)

    def _on_hop(self, hop_idx: int):
        if hop_idx < 0:
            self.spec.update_hop(-1)
            self.hop_map.highlight_hop(-1)
            return

        eng    = self.engine
        ch_idx = eng.sequence[hop_idx % len(eng.sequence)]
        ch     = eng.channels[ch_idx]

        # Sync modulation from TX panel to engine
        mod = self.tx_panel.cb_mod.currentText()
        eng.modulation = mod

        # Check for pending transmit
        pending = self.tx_panel.pop_pending()
        if pending:
            p_mod, p_msg = pending
            eng.modulation = p_mod
            mod = p_mod

            tx_bits = MsgCodec.encode(p_msg)
            tx_iq   = Modulator.modulate(mod, tx_bits)
            rx_iq   = Channel.apply(tx_iq, eng.snr_db, eng.jam_db)
            rx_bits = Demodulator.demodulate(mod, rx_iq, len(tx_bits))
            rx_msg  = MsgCodec.decode(rx_bits)
            snr_meas = Demodulator.measure_snr(rx_iq)

            # Update spectrum with real TX IQ
            self.spec.update_hop(hop_idx, tx_iq=rx_iq)
            self.hop_map.highlight_hop(hop_idx)

            self.console.log_transmission(
                mod=mod, tx_msg=p_msg, rx_msg=rx_msg,
                tx_bits=tx_bits, rx_bits=rx_bits,
                snr_db=eng.snr_db, jam_db=eng.jam_db,
                snr_meas=snr_meas,
                hop_idx=hop_idx, ch_label=ch["label"], ch_freq=ch["freq"]
            )
        else:
            # Background hop: generate idle carrier for spectrum
            n   = 256
            t   = np.arange(n)
            iq  = np.exp(1j * 2 * np.pi * 0.25 * t) * 0.1
            iq  = Channel.apply(iq, eng.snr_db, eng.jam_db)
            snr_meas = Demodulator.measure_snr(iq)
            self.spec.update_hop(hop_idx, tx_iq=iq)
            self.hop_map.highlight_hop(hop_idx)
            self.console.log_hop_only(hop_idx, ch["label"], ch["freq"],
                                      mod, eng.snr_db, eng.jam_db, snr_meas)

        self.anim._lbl()
        self.stats.refresh()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor(BG))
    pal.setColor(QPalette.WindowText,    QColor(TEXT))
    pal.setColor(QPalette.Base,          QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor(BG))
    pal.setColor(QPalette.Text,          QColor(TEXT))
    pal.setColor(QPalette.Button,        QColor(PANEL))
    pal.setColor(QPalette.ButtonText,    QColor(TEXT))
    app.setPalette(pal)
    win = FHSSWindow()
    win.show()
    sys.exit(app.exec_())