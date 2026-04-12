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
    QProgressBar, QButtonGroup, QScrollArea, QGridLayout
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
import matplotlib.ticker as ticker

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

# Jammer types
JAM_TYPES = ["Barrage", "Spot", "Partial-Band", "Sweep"]


# ═════════════════════════════════════════════════════════════════════════════
# Message codec
# ═════════════════════════════════════════════════════════════════════════════
class MsgCodec:
    @staticmethod
    def encode(text: str) -> np.ndarray:
        raw = text.encode("utf-8")
        bits = []
        for byte in raw:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return np.array(bits, dtype=np.uint8)

    @staticmethod
    def decode(bits: np.ndarray) -> str:
        bits = np.asarray(bits, dtype=np.uint8)
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
        n = max(len(tx_bits), len(rx_bits))
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        a[:len(tx_bits)] = tx_bits
        b[:len(rx_bits)] = rx_bits
        return float(np.sum(a != b)) / n


# ═════════════════════════════════════════════════════════════════════════════
# Modulator
# ═════════════════════════════════════════════════════════════════════════════
class Modulator:
    FS       = 8
    N_PILOT  = 32

    @classmethod
    def modulate(cls, mod: str, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        pilot = np.array([1,0]*16, dtype=np.uint8)
        bits  = np.concatenate([pilot, bits])
        sps = cls.FS

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

    @staticmethod
    def _fsk(bits, sps):
        f0, f1 = 0.15, 0.35
        sig = []
        for b in bits:
            f  = f1 if b else f0
            t  = np.arange(sps)
            sig.extend(np.exp(1j * 2 * np.pi * f * t))
        return np.array(sig)

    @staticmethod
    def _qpsk(bits, sps):
        gray = {(0,0): 0, (0,1): 1, (1,1): 2, (1,0): 3}
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

    @staticmethod
    def _fm(bits, sps):
        nrz    = (bits.astype(float) * 2 - 1)
        nrz_up = np.repeat(nrz, sps)
        nrz_up = gaussian_filter1d(nrz_up, sigma=sps * 0.3)
        kf     = 0.25
        phase  = np.cumsum(2 * np.pi * kf * nrz_up)
        return np.exp(1j * phase)

    @staticmethod
    def _css(bits, sps):
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

    @staticmethod
    def _msk(bits, sps):
        nrz    = (bits.astype(float) * 2 - 1)
        nrz_up = np.repeat(nrz, sps)
        phase  = np.cumsum(nrz_up * np.pi / (2 * sps))
        return np.exp(1j * phase)


# ═════════════════════════════════════════════════════════════════════════════
# Channel  —  now with per-channel jam targeting and jam type selection
# ═════════════════════════════════════════════════════════════════════════════
class Channel:
    @staticmethod
    def apply(iq: np.ndarray, snr_db: float, jam_db: float,
              ch_idx: int = 0,
              jammed_channels: set = None,
              jam_type: str = "Barrage",
              sweep_step: int = 0,
              num_channels: int = 8) -> np.ndarray:
        """
        Add AWGN + optional per-channel/type jamming.

        jam_type:
          "Barrage"      – wideband noise+CW hitting ALL channels (legacy behaviour)
          "Spot"         – full power only on jammed_channels set
          "Partial-Band" – jammed_channels + adjacent channels at -6 dB
          "Sweep"        – cycles through channels based on sweep_step counter
        """
        sig_pwr = np.mean(np.abs(iq)**2) + 1e-12
        n = len(iq)

        # AWGN
        noise_pwr = sig_pwr * 10**(-snr_db / 10)
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(noise_pwr / 2)

        jam = np.zeros(n, dtype=complex)
        if jam_db <= -60:
            return iq + noise

        jam_pwr_full = sig_pwr * 10**(jam_db / 10)

        # Determine effective jam power on this channel
        effective_pwr = 0.0

        if jam_type == "Barrage":
            # All channels, mix of CW + noise
            effective_pwr = jam_pwr_full

        elif jam_type == "Spot":
            if jammed_channels and ch_idx in jammed_channels:
                effective_pwr = jam_pwr_full
            # else 0

        elif jam_type == "Partial-Band":
            if jammed_channels:
                if ch_idx in jammed_channels:
                    effective_pwr = jam_pwr_full
                else:
                    # Check if adjacent to any jammed channel
                    for jc in jammed_channels:
                        if abs(ch_idx - jc) == 1:
                            effective_pwr = jam_pwr_full * 0.25  # -6 dB
                            break

        elif jam_type == "Sweep":
            # Sweep jammer cycles through channels round-robin
            target = sweep_step % num_channels
            if ch_idx == target:
                effective_pwr = jam_pwr_full
            elif abs(ch_idx - target) == 1:
                effective_pwr = jam_pwr_full * 0.3

        if effective_pwr > 0:
            t   = np.arange(n)
            cw  = np.exp(1j * 2 * np.pi * 0.25 * t)
            cw += np.exp(1j * 2 * np.pi * 0.10 * t) * 0.5
            cw  = cw / (np.max(np.abs(cw)) + 1e-12)
            barrage = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
            jam = (0.5 * cw + 0.5 * barrage) * np.sqrt(effective_pwr)

        return iq + noise + jam


# ═════════════════════════════════════════════════════════════════════════════
# Demodulator
# ═════════════════════════════════════════════════════════════════════════════
class Demodulator:
    FS      = Modulator.FS
    N_PILOT = Modulator.N_PILOT

    @classmethod
    def demodulate(cls, mod: str, iq: np.ndarray, n_bits: int) -> np.ndarray:
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

        bits = bits[cls.N_PILOT:]
        if len(bits) >= n_bits:
            return bits[:n_bits]
        return np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])

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
            seg = iq[i*sps:(i+1)*sps]
            e0  = np.abs(np.dot(seg, ref0.conj()))
            e1  = np.abs(np.dot(seg, ref1.conj()))
            bits.append(1 if e1 > e0 else 0)
        return np.array(bits, dtype=np.uint8)

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

    @classmethod
    def _fm(cls, iq):
        sps   = cls.FS
        phase = np.unwrap(np.angle(iq))
        inst  = np.diff(phase) / (2 * np.pi * 0.25)
        n_syms = (len(inst)) // sps
        bits = []
        for i in range(n_syms):
            seg = inst[i*sps:(i+1)*sps]
            bits.append(1 if np.mean(seg) > 0 else 0)
        return np.array(bits, dtype=np.uint8)

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
        self.jam_type      = "Barrage"
        self.jammed_channels = set()   # empty = no specific targets
        self._sweep_step   = 0
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

    def advance_sweep(self):
        self._sweep_step += 1

    def is_channel_jammed(self, ch_idx: int) -> bool:
        """Return True if this channel will receive jamming on the current hop."""
        if self.jam_db <= -60:
            return False
        jt = self.jam_type
        if jt == "Barrage":
            return True
        elif jt == "Spot":
            return ch_idx in self.jammed_channels
        elif jt == "Partial-Band":
            if ch_idx in self.jammed_channels:
                return True
            return any(abs(ch_idx - jc) == 1 for jc in self.jammed_channels)
        elif jt == "Sweep":
            target = self._sweep_step % self.num_channels
            return abs(ch_idx - target) <= 1
        return False

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

    iq_ready = pyqtSignal(object, int, int)

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
                                             color=RED, lw=0.8, alpha=0.0, zorder=3)
        for ch in eng.channels:
            self.ax_fft.axvspan(ch["freq"] - eng.channel_bw*0.5,
                                ch["freq"] + eng.channel_bw*0.5,
                                alpha=0.07, color=ch["color"], zorder=1)
        self._active_span = self.ax_fft.axvspan(
            eng.channels[0]["freq"] - eng.channel_bw*0.5,
            eng.channels[0]["freq"] + eng.channel_bw*0.5,
            alpha=0.25, color=ACCENT, zorder=2)

        # Jammed channel highlight spans (one per channel, toggled by alpha)
        self._jam_spans = []
        for ch in eng.channels:
            sp = self.ax_fft.axvspan(
                ch["freq"] - eng.channel_bw*0.5,
                ch["freq"] + eng.channel_bw*0.5,
                alpha=0.0, color=RED, zorder=2)
            self._jam_spans.append(sp)

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
        if hop_idx < 0: return
        eng    = self.engine
        ch_idx = eng.sequence[hop_idx % len(eng.sequence)]
        ch     = eng.channels[ch_idx]
        freq   = self._freq_axis()

        if tx_iq is not None and len(tx_iq) > 0:
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

        mask = (freq < ch["freq"] - eng.channel_bw*0.55) | (freq > ch["freq"] + eng.channel_bw*0.55)
        frame[mask] = self.NOISE_FLOOR + np.random.randn(mask.sum()) * 1.8

        # Jammer overlay in spectrum
        if eng.jam_db > -50:
            jam_frame = np.full(self.N_FFT, self.NOISE_FLOOR - 10)
            for f_tone in [0.25, 0.10]:
                tone_freq = eng.span_start + f_tone * (eng.span_end - eng.span_start)
                idx = np.argmin(np.abs(freq - tone_freq))
                # Only show jam spike on actually-jammed channels
                if eng.is_channel_jammed(ch_idx):
                    jam_frame[max(0,idx-3):idx+4] = self.NOISE_FLOOR + eng.jam_db * 0.8 + 10
            jam_frame += eng.jam_db * 0.08 + np.random.randn(self.N_FFT) * 1.5
            self._jam_line.set_ydata(jam_frame)
            self._jam_line.set_alpha(0.55 if eng.is_channel_jammed(ch_idx) else 0.15)
        else:
            self._jam_line.set_alpha(0.0)

        # Update jammed channel highlight spans
        for i, sp in enumerate(self._jam_spans):
            sp.set_alpha(0.18 if (eng.jam_db > -50 and eng.is_channel_jammed(i)) else 0.0)

        frame = uniform_filter1d(frame, size=3)
        self._peak_hold = np.maximum(self._peak_hold, frame) * 0.97 + self.NOISE_FLOOR * 0.03
        self._fft_line.set_ydata(frame)
        self._peak_line.set_ydata(self._peak_hold)

        x0, x1 = ch["freq"] - eng.channel_bw*0.5, ch["freq"] + eng.channel_bw*0.5
        self._active_span.remove()
        self._active_span = self.ax_fft.axvspan(x0, x1, alpha=0.25, color=ch["color"], zorder=2)

        jam_str = f"  JAM={eng.jam_db:+.0f}dB [{eng.jam_type}]" if eng.jam_db > -50 else ""
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
# NEW: Jamming Analysis Canvas
# ═════════════════════════════════════════════════════════════════════════════
class JammingAnalysisCanvas(FigureCanvas):
    """
    Two subplots:
      Top:    Per-channel jam exposure bar chart (colour = jammed/partial/clean)
      Bottom: BER-vs-hop timeline for the active channel, updated live
    """
    MAX_HISTORY = 64   # hops to keep in timeline

    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine  = engine
        self.fig     = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)

        # Per-channel accumulated jamming power (normalized 0-1)
        self._jam_exposure  = np.zeros(engine.num_channels)
        # BER history per hop
        self._ber_history   = []
        self._hop_history   = []
        self._ch_history    = []
        self._snr_history   = []

        self._build_axes()

    def _build_axes(self):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1.4, 1],
                               width_ratios=[1.6, 1],
                               hspace=0.42, wspace=0.35,
                               left=0.10, right=0.97, top=0.92, bottom=0.10)
        self.ax_bar   = self.fig.add_subplot(gs[0, 0])   # per-channel jam bar
        self.ax_heat  = self.fig.add_subplot(gs[0, 1])   # jam-type legend + stats
        self.ax_ber   = self.fig.add_subplot(gs[1, :])   # BER/SNR timeline

        for ax in (self.ax_bar, self.ax_heat, self.ax_ber):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        self.fig.patch.set_facecolor(BG)

        self.ax_bar.set_title("Channel Jam Exposure", color=TEXT, fontsize=9, fontweight="bold", pad=4)
        self.ax_ber.set_title("BER / SNR Timeline (per hop)", color=TEXT, fontsize=9, fontweight="bold", pad=4)
        self.ax_heat.set_title("Jam Status", color=TEXT, fontsize=9, fontweight="bold", pad=4)

        self._redraw_all()

    def _redraw_all(self):
        self._redraw_bar()
        self._redraw_timeline()
        self._redraw_status()
        self.draw_idle()

    def _redraw_bar(self):
        eng = self.engine
        ax  = self.ax_bar
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        n = eng.num_channels
        xs = np.arange(n)
        colors = []
        for i in range(n):
            if eng.jam_db <= -60:
                colors.append(MUTED + "60")
            elif eng.is_channel_jammed(i):
                colors.append(RED)
            else:
                colors.append(GREEN + "90")

        # Jam exposure: blend recent hops
        exposure = self._jam_exposure[:n] if len(self._jam_exposure) >= n else np.zeros(n)
        bars = ax.bar(xs, exposure, color=colors, width=0.7, zorder=3)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"CH{i+1}" for i in range(n)], fontsize=7, color=MUTED)
        ax.set_ylabel("Jam Exposure", color=MUTED, fontsize=7.5)
        ax.grid(axis="y", color=BORDER, lw=0.4, alpha=0.6)
        ax.set_title("Channel Jam Exposure", color=TEXT, fontsize=9, fontweight="bold", pad=4)

        # Mark sweep position if applicable
        if eng.jam_type == "Sweep" and eng.jam_db > -60:
            target = eng._sweep_step % n
            ax.axvline(target, color=YELLOW, lw=1.5, ls="--", alpha=0.7, zorder=5)
            ax.text(target + 0.1, 0.92, "sweep→", color=YELLOW, fontsize=7)

    def _redraw_timeline(self):
        eng = self.engine
        ax  = self.ax_ber
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title("BER / SNR Timeline (per hop)", color=TEXT, fontsize=9, fontweight="bold", pad=4)

        if len(self._ber_history) < 2:
            ax.text(0.5, 0.5, "Waiting for hops…", ha="center", va="center",
                    color=MUTED, fontsize=9, transform=ax.transAxes)
            return

        hops = self._hop_history[-self.MAX_HISTORY:]
        bers = self._ber_history[-self.MAX_HISTORY:]
        snrs = self._snr_history[-self.MAX_HISTORY:]
        chs  = self._ch_history[-self.MAX_HISTORY:]

        ax2 = ax.twinx()
        ax2.set_facecolor(PANEL)
        ax2.tick_params(colors=MUTED, labelsize=7)

        # BER bars coloured by channel
        bar_colors = [CHANNEL_COLORS[c % len(CHANNEL_COLORS)] for c in chs]
        ax.bar(hops, bers, color=bar_colors, alpha=0.7, width=0.75, zorder=3)

        # SNR line
        ax2.plot(hops, snrs, color=YELLOW, lw=1.2, alpha=0.85, zorder=4)
        ax2.set_ylabel("SNR meas (dB)", color=YELLOW, fontsize=7)
        ax2.tick_params(axis='y', colors=YELLOW)

        ax.set_ylabel("BER", color=MUTED, fontsize=7.5)
        ax.set_xlabel("Hop #", color=MUTED, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.grid(color=BORDER, lw=0.35, alpha=0.5)

        # Threshold line at BER=0.1
        ax.axhline(0.1, color=RED, lw=0.8, ls="--", alpha=0.5)
        ax.text(hops[0], 0.12, "BER=0.1 threshold", color=RED, fontsize=7, alpha=0.7)

    def _redraw_status(self):
        eng = self.engine
        ax  = self.ax_heat
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title("Jam Status", color=TEXT, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")

        lines = [
            ("Type",    eng.jam_type,                       ACCENT),
            ("Power",   f"{eng.jam_db:.0f} dB" if eng.jam_db > -60 else "OFF",
                        RED if eng.jam_db > -50 else MUTED),
            ("SNR cfg", f"{eng.snr_db:.0f} dB",             GREEN),
            ("Targets", self._target_str(),                  YELLOW),
            ("Hops",    str(len(self._hop_history)),          MUTED),
        ]
        for row, (k, v, col) in enumerate(lines):
            y = 0.88 - row * 0.18
            ax.text(0.05, y, k + ":", color=MUTED, fontsize=8, transform=ax.transAxes,
                    va="center")
            ax.text(0.50, y, v, color=col, fontsize=8.5, fontweight="bold",
                    transform=ax.transAxes, va="center")

        # Channel grid dots: green=clean, red=jammed
        n = eng.num_channels
        cols_per_row = 4
        for i in range(n):
            row_i = i // cols_per_row
            col_i = i % cols_per_row
            x = 0.05 + col_i * 0.23
            y = 0.14 - row_i * 0.12
            col = RED if (eng.jam_db > -60 and eng.is_channel_jammed(i)) else GREEN + "80"
            ax.add_patch(mpatches.Circle((x, y), 0.04, transform=ax.transAxes,
                                          color=col, zorder=3, clip_on=False))
            ax.text(x + 0.06, y, f"CH{i+1}", color=MUTED, fontsize=7,
                    transform=ax.transAxes, va="center")

    def _target_str(self) -> str:
        eng = self.engine
        if eng.jam_db <= -60:
            return "None"
        if eng.jam_type == "Barrage":
            return "All channels"
        if eng.jam_type == "Sweep":
            return f"Sweeping (step {eng._sweep_step % eng.num_channels})"
        if not eng.jammed_channels:
            return "None selected"
        return ", ".join(f"CH{c+1}" for c in sorted(eng.jammed_channels))

    def update_hop(self, hop_idx: int, ch_idx: int, ber: float, snr_meas: float):
        """Called on every hop to record data and refresh plots."""
        eng = self.engine
        n   = eng.num_channels

        # Resize exposure array if channels changed
        if len(self._jam_exposure) != n:
            self._jam_exposure = np.zeros(n)

        # Update jam exposure: EWMA with decay
        for i in range(n):
            hit = 1.0 if eng.is_channel_jammed(i) else 0.0
            self._jam_exposure[i] = 0.85 * self._jam_exposure[i] + 0.15 * hit

        # Record history
        self._ber_history.append(ber)
        self._snr_history.append(snr_meas)
        self._hop_history.append(hop_idx)
        self._ch_history.append(ch_idx)

        eng.advance_sweep()
        self._redraw_all()

    def reset_stats(self):
        eng = self.engine
        self._jam_exposure = np.zeros(eng.num_channels)
        self._ber_history  = []
        self._snr_history  = []
        self._hop_history  = []
        self._ch_history   = []
        self._redraw_all()

    def rebuild(self):
        self.reset_stats()
        self._build_axes()


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
                         snr_meas: float, hop_idx: int, ch_label: str, ch_freq: float,
                         jam_type: str = "Barrage", ch_jammed: bool = False):
        if self._paused:
            return

        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ber = MsgCodec.ber(tx_bits, rx_bits)
        ok  = tx_msg.strip() == rx_msg.strip()
        match_sym = "✓ MATCH" if ok else "✗ MISMATCH"
        match_col = GREEN if ok else RED

        self.console.append("")
        self._put(MUTED, f"[{ts}] ")
        self._put(match_col, f"[{match_sym}] ")
        self._put(ACCENT, f"HOP#{hop_idx+1:02d} ")
        self._put(YELLOW, f"{ch_label} {ch_freq:.2f}MHz  ")
        self._put(PURPLE, f"{mod}  ")
        self._put(MUTED,  f"SNR={snr_meas:+.1f}dB (cfg:{snr_db:.0f})  ")
        if jam_db > -50:
            jam_col = RED if ch_jammed else YELLOW
            self._put(jam_col, f"JAM={jam_db:+.0f}dB [{jam_type}]{'⚡' if ch_jammed else '○'}  ")
        self._put(MUTED, f"BER={ber:.3f}")

        self.console.append("")
        self._put(MUTED, "  TX: ")
        self._put(GREEN, repr(tx_msg))

        self.console.append("")
        self._put(MUTED, "  RX: ")
        self._put(TEXT if ok else RED, repr(rx_msg))

        n_show = min(48, len(tx_bits), len(rx_bits))
        if n_show > 0:
            self.console.append("")
            self._put(MUTED, "  TX bits[0:48]: ")
            self._put("#4fc3f7", "".join(str(b) for b in tx_bits[:n_show]))
            self.console.append("")
            self._put(MUTED, "  RX bits[0:48]: ")
            for i in range(n_show):
                tb, rb = int(tx_bits[i]), int(rx_bits[i])
                col = "#4fc3f7" if tb == rb else RED
                self._put(col, str(rb))
            if ber > 0:
                n_err = int(round(ber * max(len(tx_bits), len(rx_bits))))
                self._put(RED, f"  ← {n_err} bit errors")

        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

        doc = self.console.document()
        if doc.lineCount() > 1200:
            cur = QTextCursor(doc)
            cur.movePosition(QTextCursor.Start)
            cur.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 60)
            cur.removeSelectedText()

        jam_str = f"  JAM={jam_db:+.0f}dB [{jam_type}]{'⚡' if ch_jammed else ''}" if jam_db > -50 else ""
        self.status.setText(
            f"{'✓ OK' if ok else '✗ ERR'}  BER={ber:.3f}  SNR_meas={snr_meas:+.1f}dB{jam_str}  {mod}")
        self.status.setStyleSheet(
            f"color:{'#39d353' if ok else '#f85149'}; font-size:9px;")

    def log_hop_only(self, hop_idx: int, ch_label: str, ch_freq: float,
                     mod: str, snr_db: float, jam_db: float, snr_meas: float,
                     jam_type: str = "Barrage", ch_jammed: bool = False):
        if self._paused: return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.console.append("")
        self._put(MUTED, f"[{ts}] ")
        self._put(BORDER, f"HOP#{hop_idx+1:02d} ")
        self._put(MUTED,  f"{ch_label} {ch_freq:.2f}MHz  {mod}  SNR={snr_meas:+.1f}dB")
        if jam_db > -50:
            jam_col = RED if ch_jammed else YELLOW
            self._put(jam_col, f"  JAM={jam_db:+.0f}dB [{jam_type}]{'⚡' if ch_jammed else '○'}")
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())


# ═════════════════════════════════════════════════════════════════════════════
# TX Panel
# ═════════════════════════════════════════════════════════════════════════════
class TXPanel(QGroupBox):
    transmit_requested = pyqtSignal(str, str)

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

        self.bit_info = QLabel("0 chars → 0 bits")
        self.bit_info.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        self.msg_input.textChanged.connect(self._update_bit_info)
        lay.addWidget(self.bit_info)
        self._update_bit_info(self.msg_input.text())

        self.btn_send = QPushButton("▶  Send on Next Hop")
        self.btn_send.setStyleSheet(f"""QPushButton{{background:{ACCENT};color:#000;
            border:none;border-radius:6px;padding:7px;font-size:11px;font-weight:bold;}}
            QPushButton:hover{{background:#33ddff;}}
            QPushButton:disabled{{background:{BORDER};color:{MUTED};}}""")
        self.btn_send.clicked.connect(self._send)
        lay.addWidget(self.btn_send)

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
        if self._pending_msg:
            v = self._pending_msg
            self._pending_msg = None
            self.pending_lbl.setText("")
            self.btn_send.setEnabled(True)
            return v
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Channel Panel  —  now with jam type selector + per-channel targeting
# ═════════════════════════════════════════════════════════════════════════════
class ChannelPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, engine: FHSSEngine):
        super().__init__("Channel")
        self.engine = engine
        self._ch_btns = []
        self.setStyleSheet(f"""QGroupBox{{color:{TEXT};border:1px solid {BORDER};
            border-radius:8px;font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{ACCENT};}}
            QLabel{{color:{MUTED};font-size:10px;}}""")
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

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

        self.sl_snr = slider_row("Noise (SNR)", 0, 40, int(engine.snr_db), " dB", GREEN,
                                  lambda v: (setattr(engine, 'snr_db', float(v)), self.changed.emit()))
        self.sl_jam = slider_row("Jamming power", -60, 20, max(-60, int(engine.jam_db)), " dB", RED,
                                  lambda v: (setattr(engine, 'jam_db', float(v)), self._on_jam_change()))

        # Jam type selector
        jt_row = QWidget()
        jt_lay = QHBoxLayout(jt_row); jt_lay.setContentsMargins(0,0,0,0); jt_lay.setSpacing(6)
        jt_lay.addWidget(QLabel("Jam type"))
        self.cb_jam_type = QComboBox()
        self.cb_jam_type.addItems(JAM_TYPES)
        self.cb_jam_type.setCurrentText(engine.jam_type)
        self.cb_jam_type.setStyleSheet(f"""QComboBox{{background:{BG};color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:3px 6px;font-size:10px;}}
            QComboBox QAbstractItemView{{background:{PANEL};color:{TEXT};
            selection-background-color:{ACCENT}40;}}""")
        self.cb_jam_type.currentTextChanged.connect(self._on_jam_type_change)
        jt_lay.addWidget(self.cb_jam_type, 1)
        lay.addWidget(jt_row)

        # Per-channel target selector (shown only for Spot/Partial-Band)
        self.ch_target_group = QGroupBox("Target Channels")
        self.ch_target_group.setStyleSheet(f"""QGroupBox{{color:{YELLOW};border:1px solid {BORDER}40;
            border-radius:5px;font-size:9px;margin-top:5px;padding-top:4px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;}}""")
        self._ch_grid_lay = QGridLayout(self.ch_target_group)
        self._ch_grid_lay.setContentsMargins(4,4,4,4); self._ch_grid_lay.setSpacing(3)
        lay.addWidget(self.ch_target_group)

        self._rebuild_ch_buttons()

        # Jam type description
        self.jam_desc = QLabel("")
        self.jam_desc.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        self.jam_desc.setWordWrap(True)
        lay.addWidget(self.jam_desc)

        self._on_jam_type_change(engine.jam_type)

    def _rebuild_ch_buttons(self):
        # Clear existing buttons
        for btn in self._ch_btns:
            self._ch_grid_lay.removeWidget(btn)
            btn.deleteLater()
        self._ch_btns = []

        eng = self.engine
        cols = 4
        for i in range(eng.num_channels):
            btn = QPushButton(f"CH{i+1}")
            btn.setCheckable(True)
            btn.setChecked(i in eng.jammed_channels)
            btn.setFixedHeight(22)
            ch_col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            btn.setStyleSheet(f"""QPushButton{{background:{BG};color:{MUTED};
                border:1px solid {BORDER};border-radius:3px;font-size:9px;}}
                QPushButton:checked{{background:{RED}30;color:{RED};border-color:{RED};}}
                QPushButton:hover{{background:{HOVER};}}""")
            btn.toggled.connect(lambda checked, idx=i: self._toggle_channel(idx, checked))
            self._ch_grid_lay.addWidget(btn, i // cols, i % cols)
            self._ch_btns.append(btn)

    def _toggle_channel(self, idx: int, checked: bool):
        if checked:
            self.engine.jammed_channels.add(idx)
        else:
            self.engine.jammed_channels.discard(idx)
        self.changed.emit()

    def _on_jam_change(self):
        self.changed.emit()
        self._update_jam_desc()

    def _on_jam_type_change(self, jam_type: str = None):
        if jam_type is None:
            jam_type = self.cb_jam_type.currentText()
        self.engine.jam_type = jam_type
        show_targets = jam_type in ("Spot", "Partial-Band")
        self.ch_target_group.setVisible(show_targets)
        self._update_jam_desc()
        self.changed.emit()

    def _update_jam_desc(self):
        jt = self.engine.jam_type
        jdb = self.engine.jam_db
        on = jdb > -50
        descriptions = {
            "Barrage":      "Wideband noise + CW tones hit ALL channels simultaneously.",
            "Spot":         "Full power only on selected target channels. Others unaffected.",
            "Partial-Band": "Full power on targets + −6 dB spill onto adjacent channels.",
            "Sweep":        "Jammer rotates channel-by-channel each hop (round-robin sweep).",
        }
        base = descriptions.get(jt, "")
        if not on:
            self.jam_desc.setText("Jammer: OFF  (raise power to activate)")
            self.jam_desc.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        elif jdb > 0:
            self.jam_desc.setText(f"⚡ STRONG JAM  [{jt}]  — {base}")
            self.jam_desc.setStyleSheet(f"color:{RED}; font-size:9px; font-weight:bold;")
        elif jdb > -20:
            self.jam_desc.setText(f"[{jt}]  {base}")
            self.jam_desc.setStyleSheet(f"color:{ACCENT2}; font-size:9px;")
        else:
            self.jam_desc.setText(f"[{jt}]  {base}")
            self.jam_desc.setStyleSheet(f"color:{YELLOW}; font-size:9px;")

    def refresh_channels(self, engine: FHSSEngine):
        """Called after FHSS reconfigure to rebuild channel buttons."""
        self.engine = engine
        self._rebuild_ch_buttons()


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
        for key in ["Channels","Hop Rate","Total BW","Modulation","SNR","Jamming","Jam Type"]:
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
        self._lbs["Jam Type"].setText(e.jam_type if e.jam_db > -50 else "—")
        self._lbs["Jam Type"].setStyleSheet(
            f"color:{'#ff6b35' if e.jam_db > -50 else MUTED}; font-size:12px; font-weight:bold;")


# ═════════════════════════════════════════════════════════════════════════════
# Main Window
# ═════════════════════════════════════════════════════════════════════════════
class FHSSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FHSS Simulation v4  —  Week 3: Jam Model + Channel Targeting + Analysis")
        self.resize(1480, 900)
        self.engine        = FHSSEngine()
        self._pending_tx   = None
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

        hdr = QLabel("⟁  FHSS Simulation  ·  Message TX/RX  ·  Noise + Jamming  ·  Week 3")
        hdr.setStyleSheet(f"color:{ACCENT};font-size:15px;font-weight:700;"
                          f"font-family:'Courier New';letter-spacing:1.5px;"
                          f"border-bottom:1px solid {BORDER};padding-bottom:5px;")
        root.addWidget(hdr)

        self.stats = StatsBar(self.engine)
        root.addWidget(self.stats)

        msplit = QSplitter(Qt.Horizontal); msplit.setHandleWidth(2)

        # ── LEFT ─────────────────────────────────────────────────────────────
        lvsplit = QSplitter(Qt.Vertical); lvsplit.setHandleWidth(3)

        self.tabs = QTabWidget()

        t1 = QWidget(); tl1 = QVBoxLayout(t1); tl1.setContentsMargins(4,4,4,4)
        self.spec = SpectrumCanvas(self.engine)
        tl1.addWidget(self.spec)
        self.tabs.addTab(t1, "📡  FFT + Waterfall")

        t2 = QWidget(); tl2 = QVBoxLayout(t2); tl2.setContentsMargins(4,4,4,4)
        self.hop_map = HopMapCanvas(self.engine)
        tl2.addWidget(self.hop_map)
        self.tabs.addTab(t2, "📶  Hop Map")

        # NEW: Jamming Analysis tab
        t3 = QWidget(); tl3 = QVBoxLayout(t3); tl3.setContentsMargins(4,4,4,4)
        self.jam_analysis = JammingAnalysisCanvas(self.engine)
        tl3.addWidget(self.jam_analysis)
        self.tabs.addTab(t3, "⚡  Jamming Analysis")

        upper = QWidget(); uv = QVBoxLayout(upper); uv.setContentsMargins(0,0,0,0); uv.setSpacing(4)
        uv.addWidget(self.tabs, 1)
        self.anim = AnimBar(self.engine)
        self.anim.hop_changed.connect(self._on_hop)
        uv.addWidget(self.anim)
        lvsplit.addWidget(upper)

        self.console = DemodConsole()
        lvsplit.addWidget(self.console)
        lvsplit.setSizes([560, 260])
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
        msplit.setSizes([1000, 400])
        root.addWidget(msplit, 1)

    def _on_config_changed(self):
        self.spec.rebuild()
        self.hop_map._draw()
        self.jam_analysis.rebuild()
        self.stats.refresh()
        self.anim.refresh(self.engine)
        self.ch_panel.refresh_channels(self.engine)

    def _on_hop(self, hop_idx: int):
        if hop_idx < 0:
            self.spec.update_hop(-1)
            self.hop_map.highlight_hop(-1)
            return

        eng    = self.engine
        ch_idx = eng.sequence[hop_idx % len(eng.sequence)]
        ch     = eng.channels[ch_idx]
        ch_jammed = eng.is_channel_jammed(ch_idx)

        mod = self.tx_panel.cb_mod.currentText()
        eng.modulation = mod

        pending = self.tx_panel.pop_pending()
        if pending:
            p_mod, p_msg = pending
            eng.modulation = p_mod
            mod = p_mod

            tx_bits = MsgCodec.encode(p_msg)
            tx_iq   = Modulator.modulate(mod, tx_bits)
            rx_iq   = Channel.apply(tx_iq, eng.snr_db, eng.jam_db,
                                    ch_idx=ch_idx,
                                    jammed_channels=eng.jammed_channels,
                                    jam_type=eng.jam_type,
                                    sweep_step=eng._sweep_step,
                                    num_channels=eng.num_channels)
            rx_bits  = Demodulator.demodulate(mod, rx_iq, len(tx_bits))
            rx_msg   = MsgCodec.decode(rx_bits)
            snr_meas = Demodulator.measure_snr(rx_iq)
            ber      = MsgCodec.ber(tx_bits, rx_bits)

            self.spec.update_hop(hop_idx, tx_iq=rx_iq)
            self.hop_map.highlight_hop(hop_idx)

            self.console.log_transmission(
                mod=mod, tx_msg=p_msg, rx_msg=rx_msg,
                tx_bits=tx_bits, rx_bits=rx_bits,
                snr_db=eng.snr_db, jam_db=eng.jam_db,
                snr_meas=snr_meas,
                hop_idx=hop_idx, ch_label=ch["label"], ch_freq=ch["freq"],
                jam_type=eng.jam_type, ch_jammed=ch_jammed,
            )
        else:
            n   = 256
            t   = np.arange(n)
            iq  = np.exp(1j * 2 * np.pi * 0.25 * t) * 0.1
            rx_iq = Channel.apply(iq, eng.snr_db, eng.jam_db,
                                  ch_idx=ch_idx,
                                  jammed_channels=eng.jammed_channels,
                                  jam_type=eng.jam_type,
                                  sweep_step=eng._sweep_step,
                                  num_channels=eng.num_channels)
            snr_meas = Demodulator.measure_snr(rx_iq)
            ber = 0.0

            self.spec.update_hop(hop_idx, tx_iq=rx_iq)
            self.hop_map.highlight_hop(hop_idx)
            self.console.log_hop_only(hop_idx, ch["label"], ch["freq"],
                                      mod, eng.snr_db, eng.jam_db, snr_meas,
                                      jam_type=eng.jam_type, ch_jammed=ch_jammed)

        # Always update jamming analysis
        self.jam_analysis.update_hop(hop_idx, ch_idx, ber, snr_meas)

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