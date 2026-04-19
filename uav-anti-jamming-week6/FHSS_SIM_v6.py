"""
FHSS Simulation v6
==================
Week 5 additions:
  • Adaptive FHSS — channel avoidance using energy-detector occupancy map
      - "Avoid Jammed"   : skip any channel flagged occupied, pick random clean channel
      - "Best Channel"   : always hop to the lowest-energy (cleanest) available channel
      - Falls back to random if all channels are flagged occupied
  • Parallel simulation — Standard (fixed sequence) and Adaptive run side-by-side every hop
  • Performance Comparison tab (4 plots):
      - BER vs hop index:  Standard (red) vs Adaptive (teal)
      - Jammed-hop fraction timeline (rolling 8-hop window)
      - Channel-usage heatmap: Standard vs Adaptive  (which channels each mode picks)
      - Summary bar chart: mean BER, jammed-hop %, avoided-hop % for both modes

Weeks 1-4 retained:
  • 5 modulations: FSK, QPSK, FM, CSS (LoRa-like), RTCM/MSK
  • Message TX/RX pipeline, BER/SNR, 4 jammer models, per-channel targeting
  • Jamming Analysis tab, Energy/Sensing tab (energy detection, Pd/Pfa, heatmap)

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
TEAL    = "#20d9a0"
ORANGE  = "#f0a030"

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

# Custom colourmap for occupancy heatmap: dark=free, bright red=occupied
_occ_cmap = LinearSegmentedColormap.from_list(
    "occupancy",
    [(0.0, "#0d1117"), (0.35, "#0a2a6e"), (0.65, "#ff6b35"), (1.0, "#f85149")]
)

MODULATIONS = ["FSK", "QPSK", "FM", "CSS (LoRa-like)", "RTCM/MSK"]
JAM_TYPES   = ["Barrage", "Spot", "Partial-Band", "Sweep"]


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
# Channel
# ═════════════════════════════════════════════════════════════════════════════
class Channel:
    @staticmethod
    def apply(iq: np.ndarray, snr_db: float, jam_db: float,
              ch_idx: int = 0,
              jammed_channels: set = None,
              jam_type: str = "Barrage",
              sweep_step: int = 0,
              num_channels: int = 8) -> np.ndarray:
        sig_pwr = np.mean(np.abs(iq)**2) + 1e-12
        n = len(iq)

        noise_pwr = sig_pwr * 10**(-snr_db / 10)
        noise = (np.random.randn(n) + 1j * np.random.randn(n)) * np.sqrt(noise_pwr / 2)

        jam = np.zeros(n, dtype=complex)
        if jam_db <= -60:
            return iq + noise

        jam_pwr_full = sig_pwr * 10**(jam_db / 10)
        effective_pwr = 0.0

        if jam_type == "Barrage":
            effective_pwr = jam_pwr_full
        elif jam_type == "Spot":
            if jammed_channels and ch_idx in jammed_channels:
                effective_pwr = jam_pwr_full
        elif jam_type == "Partial-Band":
            if jammed_channels:
                if ch_idx in jammed_channels:
                    effective_pwr = jam_pwr_full
                else:
                    for jc in jammed_channels:
                        if abs(ch_idx - jc) == 1:
                            effective_pwr = jam_pwr_full * 0.25
                            break
        elif jam_type == "Sweep":
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
# ★ NEW: Energy Detector  (Week 4 core)
# ═════════════════════════════════════════════════════════════════════════════
class EnergyDetector:
    """
    Spectrum sensing via energy detection.

    For each channel we compute the normalised received energy (mean |IQ|²),
    compare it to a threshold, and declare the channel occupied or free.

    Threshold strategies:
      "Fixed dB"  – user-set absolute dB level
      "Adaptive"  – per-channel EWMA noise floor + margin (CFAR-lite)
      "Neyman-Pearson" – threshold set to achieve target Pfa under H0 (noise only)
    """

    # EWMA smoothing for adaptive noise floor
    ALPHA_NOISE = 0.05   # slow learning rate for noise floor
    NP_PFA      = 0.05   # target false-alarm probability for N-P mode

    def __init__(self, num_channels: int):
        self.num_channels    = num_channels
        self.threshold_mode  = "Adaptive"  # "Fixed dB" | "Adaptive" | "Neyman-Pearson"
        self.fixed_threshold = -10.0       # dB relative to noise floor (Fixed dB mode)
        self.adaptive_margin = 6.0         # dB above estimated noise floor (Adaptive mode)

        # Per-channel noise floor estimates (initialised high, will decay)
        self._noise_floor_db = np.full(num_channels, -5.0)
        # Per-channel variance of energy (for N-P threshold)
        self._energy_var     = np.full(num_channels, 1.0)
        # Current per-channel energy in dB
        self.energy_db       = np.full(num_channels, -30.0)
        # Computed thresholds (dB)
        self.thresholds_db   = np.zeros(num_channels)
        # Occupancy decision: True = occupied
        self.occupied        = np.zeros(num_channels, dtype=bool)

    def reset(self, num_channels: int):
        self.num_channels    = num_channels
        self._noise_floor_db = np.full(num_channels, -5.0)
        self._energy_var     = np.full(num_channels, 1.0)
        self.energy_db       = np.full(num_channels, -30.0)
        self.thresholds_db   = np.zeros(num_channels)
        self.occupied        = np.zeros(num_channels, dtype=bool)

    def sense(self, ch_idx: int, rx_iq: np.ndarray,
              ground_truth_jammed: bool = False) -> dict:
        """
        Perform energy detection on one channel's IQ samples.

        Returns a dict with:
          energy_db      – measured channel energy (dBW normalised)
          threshold_db   – decision threshold used
          occupied       – bool decision
          noise_floor_db – current noise floor estimate
        """
        # ── Energy computation ────────────────────────────────────────────
        energy_lin = float(np.mean(np.abs(rx_iq) ** 2))
        energy_db  = 10.0 * np.log10(energy_lin + 1e-12)
        self.energy_db[ch_idx] = energy_db

        # ── Noise floor update (EWMA on quiet channels only) ──────────────
        # Update noise floor estimate only when we believe channel is free
        # (bootstrap: always update during first few calls)
        prev_decision = self.occupied[ch_idx]
        if not prev_decision:
            alpha = self.ALPHA_NOISE
            self._noise_floor_db[ch_idx] = (
                (1 - alpha) * self._noise_floor_db[ch_idx] + alpha * energy_db
            )
            # Update variance estimate (for N-P)
            e_var = float((energy_lin - 10 ** (self._noise_floor_db[ch_idx] / 10)) ** 2)
            self._energy_var[ch_idx] = (
                (1 - alpha) * self._energy_var[ch_idx] + alpha * max(e_var, 1e-12)
            )

        # ── Threshold computation ─────────────────────────────────────────
        nf = self._noise_floor_db[ch_idx]

        if self.threshold_mode == "Fixed dB":
            threshold_db = self.fixed_threshold
        elif self.threshold_mode == "Adaptive":
            threshold_db = nf + self.adaptive_margin
        elif self.threshold_mode == "Neyman-Pearson":
            # Under H0 (noise only), energy ~ noise floor ± sigma.
            # Set threshold so that P(E > thr | H0) = NP_PFA
            # Using Gaussian approx: thr = nf + z_{1-Pfa} * sigma_dB
            sigma_lin = float(np.sqrt(self._energy_var[ch_idx]))
            sigma_db  = 10.0 * np.log10(max(sigma_lin, 1e-12) + 10 ** (nf / 10)) - nf
            sigma_db  = max(abs(sigma_db), 0.5)
            import math
            z = -math.log(self.NP_PFA + 1e-12)   # crude approximation
            threshold_db = nf + z * sigma_db
        else:
            threshold_db = nf + self.adaptive_margin

        self.thresholds_db[ch_idx] = threshold_db

        # ── Decision ─────────────────────────────────────────────────────
        occupied = energy_db > threshold_db
        self.occupied[ch_idx] = occupied

        return {
            "energy_db":      energy_db,
            "threshold_db":   threshold_db,
            "occupied":       occupied,
            "noise_floor_db": nf,
            "ground_truth":   ground_truth_jammed,
        }


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW Week 5: Adaptive FHSS Engine
# ═════════════════════════════════════════════════════════════════════════════
class AdaptiveFHSSEngine:
    """
    Runs alongside the standard FHSSEngine.  On each hop it uses the
    EnergyDetector's occupancy map to intelligently choose a channel,
    rather than following the fixed pseudo-random sequence.

    Strategies
    ----------
    "Avoid Jammed"  – exclude occupied channels, pick uniformly from clean ones.
    "Best Channel"  – pick the channel with the lowest measured energy_db.

    If ALL channels are flagged occupied the engine falls back to the
    lowest-energy channel (graceful degradation).
    """

    STRATEGIES = ["Avoid Jammed", "Best Channel"]

    def __init__(self, engine: "FHSSEngine", detector: "EnergyDetector"):
        self.engine    = engine
        self.detector  = detector
        self.strategy  = "Avoid Jammed"
        self.enabled   = True
        self._rng      = random.Random(99)   # independent seed so choices differ from standard

        # History of channels chosen by adaptive engine (for heatmap)
        self.channel_history: list[int] = []
        # Per-hop decision metadata
        self.last_ch_idx   = 0
        self.last_avoided  = False   # True if this hop actually avoided a jammed channel

    def choose_channel(self) -> int:
        """Return the channel index to use for this hop."""
        n        = self.engine.num_channels
        occupied = self.detector.occupied[:n]
        energies = self.detector.energy_db[:n]

        clean_indices = [i for i in range(n) if not occupied[i]]
        all_jammed    = len(clean_indices) == 0

        if self.strategy == "Avoid Jammed":
            if all_jammed:
                # Graceful degradation: pick lowest energy even among jammed
                ch = int(np.argmin(energies[:n]))
            else:
                ch = self._rng.choice(clean_indices)

        elif self.strategy == "Best Channel":
            if all_jammed:
                ch = int(np.argmin(energies[:n]))
            else:
                # Best among clean channels only
                clean_energies = [(energies[i], i) for i in clean_indices]
                ch = min(clean_energies)[1]
        else:
            ch = self._rng.randint(0, n - 1)

        # Record whether we avoided something this hop
        std_ch = self.engine.sequence[
            len(self.channel_history) % self.engine.sequence_len
        ]
        self.last_avoided = (occupied[std_ch] and ch != std_ch)
        self.last_ch_idx  = ch
        self.channel_history.append(ch)
        return ch

    def reset(self):
        self.channel_history = []
        self.last_avoided    = False
        self.last_ch_idx     = 0
        self._rng            = random.Random(99)


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW Week 5: Performance Tracker
# ═════════════════════════════════════════════════════════════════════════════
class PerformanceTracker:
    """
    Accumulates per-hop statistics for both Standard and Adaptive FHSS.
    Stores enough history for all four comparison plots.
    """
    MAX_HOPS = 200

    def __init__(self):
        self.reset()

    def reset(self):
        # Per-hop lists  (truncated at MAX_HOPS)
        self.hops: list[int]       = []

        # Standard FHSS
        self.std_ber:      list[float] = []
        self.std_snr:      list[float] = []
        self.std_jammed:   list[bool]  = []   # was the std channel actually jammed?
        self.std_ch:       list[int]   = []   # which channel was used

        # Adaptive FHSS
        self.adp_ber:      list[float] = []
        self.adp_snr:      list[float] = []
        self.adp_jammed:   list[bool]  = []
        self.adp_ch:       list[int]   = []
        self.adp_avoided:  list[bool]  = []   # did adaptive actually dodge a jammed channel?

    def record(self, hop_idx: int,
               std_ber: float, std_snr: float, std_ch: int, std_jammed: bool,
               adp_ber: float, adp_snr: float, adp_ch: int, adp_jammed: bool,
               adp_avoided: bool):
        if len(self.hops) >= self.MAX_HOPS:
            # Slide window
            for lst in (self.hops, self.std_ber, self.std_snr, self.std_jammed,
                        self.std_ch, self.adp_ber, self.adp_snr, self.adp_jammed,
                        self.adp_ch, self.adp_avoided):
                lst.pop(0)

        self.hops.append(hop_idx)
        self.std_ber.append(std_ber);   self.std_snr.append(std_snr)
        self.std_jammed.append(std_jammed); self.std_ch.append(std_ch)
        self.adp_ber.append(adp_ber);   self.adp_snr.append(adp_snr)
        self.adp_jammed.append(adp_jammed); self.adp_ch.append(adp_ch)
        self.adp_avoided.append(adp_avoided)

    # ── Aggregate stats ──────────────────────────────────────────────────────
    @property
    def n(self): return len(self.hops)

    def mean_ber(self, mode="std") -> float:
        data = self.std_ber if mode == "std" else self.adp_ber
        return float(np.mean(data)) if data else 0.0

    def jammed_frac(self, mode="std") -> float:
        data = self.std_jammed if mode == "std" else self.adp_jammed
        return float(np.mean(data)) if data else 0.0

    def avoided_frac(self) -> float:
        return float(np.mean(self.adp_avoided)) if self.adp_avoided else 0.0

    def rolling_jammed(self, mode="std", window=8) -> list:
        data = self.std_jammed if mode == "std" else self.adp_jammed
        out  = []
        for i in range(len(data)):
            sl  = data[max(0, i - window + 1): i + 1]
            out.append(float(np.mean([1 if x else 0 for x in sl])))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW Week 5: Performance Comparison Canvas
# ═════════════════════════════════════════════════════════════════════════════
class PerformanceComparisonCanvas(FigureCanvas):
    """
    Four-panel comparison between Standard FHSS and Adaptive FHSS:

    [top-left]    BER vs hop  —  both modes overlaid as line+marker
    [top-right]   Channel-usage heatmap: rows=channels, cols=hops
                  top half = Standard, bottom half = Adaptive
    [bot-left]    Rolling jammed-hop fraction (8-hop window) — both modes
    [bot-right]   Summary bar chart: mean BER / jammed% / avoided%
    """

    def __init__(self, tracker: "PerformanceTracker",
                 engine: "FHSSEngine", parent=None):
        self.tracker = tracker
        self.engine  = engine
        self.fig     = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._build_axes()

    def _build_axes(self):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1, 1],
                               width_ratios=[1.6, 1.4],
                               hspace=0.48, wspace=0.32,
                               left=0.08, right=0.97, top=0.93, bottom=0.09)
        self.ax_ber    = self.fig.add_subplot(gs[0, 0])
        self.ax_hmap   = self.fig.add_subplot(gs[0, 1])
        self.ax_jam    = self.fig.add_subplot(gs[1, 0])
        self.ax_summ   = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_ber, self.ax_hmap, self.ax_jam, self.ax_summ):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        self.fig.patch.set_facecolor(BG)
        self._redraw_all()

    def _ax_style(self, ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

    # ── BER comparison plot ──────────────────────────────────────────────────
    def _draw_ber(self):
        ax = self.ax_ber; ax.clear(); self._ax_style(ax)
        ax.set_title("BER vs Hop  —  Standard vs Adaptive", color=TEXT,
                     fontsize=9, fontweight="bold", pad=4)
        t = self.tracker
        if t.n < 2:
            ax.text(0.5, 0.5, "Waiting for hops…", ha="center", va="center",
                    color=MUTED, fontsize=9, transform=ax.transAxes)
            return

        hops = t.hops
        ax.plot(hops, t.std_ber, color=RED,  lw=1.2, alpha=0.85,
                marker=".", markersize=3, label="Standard")
        ax.plot(hops, t.adp_ber, color=TEAL, lw=1.2, alpha=0.85,
                marker=".", markersize=3, label="Adaptive")

        # Shade jammed hops for standard
        for i, jm in enumerate(t.std_jammed):
            if jm:
                ax.axvspan(hops[i] - 0.4, hops[i] + 0.4,
                           alpha=0.08, color=RED, zorder=0)

        ax.axhline(0.1, color=YELLOW, lw=0.7, ls="--", alpha=0.6)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("BER", color=MUTED, fontsize=7.5)
        ax.set_xlabel("Hop #", color=MUTED, fontsize=7.5)
        ax.grid(color=BORDER, lw=0.35, alpha=0.5)
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")

        # Annotation: mean BER delta
        delta = t.mean_ber("std") - t.mean_ber("adp")
        sign  = "+" if delta >= 0 else ""
        color = GREEN if delta >= 0 else ACCENT2
        ax.text(0.02, 0.93,
                f"Adaptive improvement: {sign}{delta:.3f}",
                transform=ax.transAxes, color=color, fontsize=7.5,
                fontweight="bold", va="top")

    # ── Channel usage heatmap ────────────────────────────────────────────────
    def _draw_heatmap(self):
        ax = self.ax_hmap; ax.clear(); self._ax_style(ax)
        ax.set_title("Channel Usage  (top=Std, bot=Adp)", color=TEXT,
                     fontsize=9, fontweight="bold", pad=4)
        t  = self.tracker
        n  = self.engine.num_channels
        H  = min(t.n, 60)   # show last 60 hops

        if H < 2:
            ax.text(0.5, 0.5, "Waiting…", ha="center", va="center",
                    color=MUTED, fontsize=9, transform=ax.transAxes)
            return

        std_chs = t.std_ch[-H:]
        adp_chs = t.adp_ch[-H:]

        # Build usage matrices  [channel × hop]
        std_mat = np.zeros((n, H))
        adp_mat = np.zeros((n, H))
        for col, ch in enumerate(std_chs): std_mat[ch, col] = 1.0
        for col, ch in enumerate(adp_chs): adp_mat[ch, col] = 1.0

        # Stack vertically: standard on top, adaptive on bottom, separated by gap row
        gap     = np.zeros((1, H))
        combined = np.vstack([std_mat, gap, adp_mat])

        _cmap_usage = LinearSegmentedColormap.from_list(
            "usage", [(0, PANEL), (1, ACCENT)])
        ax.imshow(combined, aspect="auto", origin="upper",
                  cmap=_cmap_usage, vmin=0, vmax=1,
                  interpolation="nearest", zorder=2)

        # Y-axis labels
        yticks = list(range(n)) + [n + 0.5] + list(range(n + 1, 2 * n + 1))
        ylabels= [f"S{i+1}" for i in range(n)] + ["─"] + [f"A{i+1}" for i in range(n)]
        ax.set_yticks(range(2 * n + 1))
        ax.set_yticklabels(
            [f"S{i+1}" for i in range(n)] + [""] + [f"A{i+1}" for i in range(n)],
            fontsize=6, color=MUTED)

        # Horizontal divider
        ax.axhline(n - 0.5, color=BORDER, lw=1.5)
        ax.axhline(n + 0.5, color=BORDER, lw=1.5)

        # Colour-code avoided hops in adaptive rows
        for col, (ch, avoided) in enumerate(zip(adp_chs, t.adp_avoided[-H:])):
            if avoided:
                ax.add_patch(mpatches.Rectangle(
                    (col - 0.5, n + 1 + ch - 0.5), 1, 1,
                    linewidth=1.2, edgecolor=GREEN, facecolor="none", zorder=5))

        ax.set_xlabel("Hop (recent →)", color=MUTED, fontsize=7)
        ax.set_xticks([])

        # Legend patches
        p1 = mpatches.Patch(color=ACCENT,  label="Channel used")
        p2 = mpatches.Patch(color="none",  label="Avoided (green border)",
                            edgecolor=GREEN, linewidth=1.2)
        ax.legend(handles=[p1, p2], fontsize=6.5, facecolor=PANEL,
                  edgecolor=BORDER, labelcolor=TEXT, loc="lower right")

    # ── Rolling jammed fraction ──────────────────────────────────────────────
    def _draw_jam_frac(self):
        ax = self.ax_jam; ax.clear(); self._ax_style(ax)
        ax.set_title("Rolling Jammed-Hop Fraction  (window=8)", color=TEXT,
                     fontsize=9, fontweight="bold", pad=4)
        t = self.tracker
        if t.n < 2:
            ax.text(0.5, 0.5, "Waiting for hops…", ha="center", va="center",
                    color=MUTED, fontsize=9, transform=ax.transAxes)
            return

        hops    = t.hops
        std_jf  = t.rolling_jammed("std",  window=8)
        adp_jf  = t.rolling_jammed("adp",  window=8)

        ax.fill_between(hops, std_jf, alpha=0.18, color=RED,  step="mid")
        ax.fill_between(hops, adp_jf, alpha=0.18, color=TEAL, step="mid")
        ax.step(hops, std_jf, color=RED,  lw=1.3, where="mid", label="Standard")
        ax.step(hops, adp_jf, color=TEAL, lw=1.3, where="mid", label="Adaptive")

        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("Jammed fraction", color=MUTED, fontsize=7.5)
        ax.set_xlabel("Hop #",           color=MUTED, fontsize=7.5)
        ax.grid(color=BORDER, lw=0.35, alpha=0.5)
        ax.axhline(0.5, color=YELLOW, lw=0.7, ls="--", alpha=0.5)
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")

    # ── Summary bar chart ────────────────────────────────────────────────────
    def _draw_summary(self):
        ax = self.ax_summ; ax.clear(); self._ax_style(ax)
        ax.set_title("Performance Summary", color=TEXT,
                     fontsize=9, fontweight="bold", pad=4)
        t = self.tracker
        if t.n < 2:
            ax.text(0.5, 0.5, "Waiting…", ha="center", va="center",
                    color=MUTED, fontsize=9, transform=ax.transAxes)
            return

        metrics = ["Mean BER", "Jammed %", "Avoided %"]
        std_vals = [
            t.mean_ber("std"),
            t.jammed_frac("std"),
            0.0,              # N/A for standard
        ]
        adp_vals = [
            t.mean_ber("adp"),
            t.jammed_frac("adp"),
            t.avoided_frac(),
        ]

        x     = np.arange(len(metrics))
        width = 0.32
        bars_s = ax.bar(x - width/2, std_vals, width, color=RED,  alpha=0.80,
                        label="Standard", zorder=3)
        bars_a = ax.bar(x + width/2, adp_vals, width, color=TEAL, alpha=0.80,
                        label="Adaptive", zorder=3)

        # Value labels on bars
        for bar in bars_s:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7, color=RED)
        for bar in bars_a:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7, color=TEAL)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=8, color=MUTED)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Value (0–1)", color=MUTED, fontsize=7.5)
        ax.grid(axis="y", color=BORDER, lw=0.35, alpha=0.5)
        ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")

        # Gain annotation
        ber_gain = (t.mean_ber("std") - t.mean_ber("adp"))
        jam_gain = (t.jammed_frac("std") - t.jammed_frac("adp"))
        lines = []
        if abs(ber_gain) > 0.001:
            lines.append(f"BER Δ  {'+' if ber_gain>0 else ''}{ber_gain:.3f}")
        if abs(jam_gain) > 0.01:
            lines.append(f"Jam Δ  {'+' if jam_gain>0 else ''}{jam_gain:.2f}")
        if lines:
            ax.text(0.03, 0.97, "\n".join(lines),
                    transform=ax.transAxes, color=GREEN,
                    fontsize=7.5, va="top", fontfamily="monospace")

    def _redraw_all(self):
        self._draw_ber()
        self._draw_heatmap()
        self._draw_jam_frac()
        self._draw_summary()
        self.draw_idle()

    def update(self):
        """Called every hop to refresh all four plots."""
        self._redraw_all()

    def rebuild(self):
        self._build_axes()


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
        self.jammed_channels = set()
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

        if eng.jam_db > -50:
            jam_frame = np.full(self.N_FFT, self.NOISE_FLOOR - 10)
            for f_tone in [0.25, 0.10]:
                tone_freq = eng.span_start + f_tone * (eng.span_end - eng.span_start)
                idx = np.argmin(np.abs(freq - tone_freq))
                if eng.is_channel_jammed(ch_idx):
                    jam_frame[max(0,idx-3):idx+4] = self.NOISE_FLOOR + eng.jam_db * 0.8 + 10
            jam_frame += eng.jam_db * 0.08 + np.random.randn(self.N_FFT) * 1.5
            self._jam_line.set_ydata(jam_frame)
            self._jam_line.set_alpha(0.55 if eng.is_channel_jammed(ch_idx) else 0.15)
        else:
            self._jam_line.set_alpha(0.0)

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
# Jamming Analysis Canvas  (Week 3)
# ═════════════════════════════════════════════════════════════════════════════
class JammingAnalysisCanvas(FigureCanvas):
    MAX_HISTORY = 64

    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine  = engine
        self.fig     = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._jam_exposure  = np.zeros(engine.num_channels)
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
        self.ax_bar   = self.fig.add_subplot(gs[0, 0])
        self.ax_heat  = self.fig.add_subplot(gs[0, 1])
        self.ax_ber   = self.fig.add_subplot(gs[1, :])
        for ax in (self.ax_bar, self.ax_heat, self.ax_ber):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        self.fig.patch.set_facecolor(BG)
        self._redraw_all()

    def _redraw_all(self):
        self._redraw_bar()
        self._redraw_timeline()
        self._redraw_status()
        self.draw_idle()

    def _redraw_bar(self):
        eng = self.engine; ax = self.ax_bar
        ax.clear(); ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        n = eng.num_channels; xs = np.arange(n)
        colors = []
        for i in range(n):
            if eng.jam_db <= -60:
                colors.append(MUTED + "60")
            elif eng.is_channel_jammed(i):
                colors.append(RED)
            else:
                colors.append(GREEN + "90")
        exposure = self._jam_exposure[:n] if len(self._jam_exposure) >= n else np.zeros(n)
        ax.bar(xs, exposure, color=colors, width=0.7, zorder=3)
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"CH{i+1}" for i in range(n)], fontsize=7, color=MUTED)
        ax.set_ylabel("Jam Exposure", color=MUTED, fontsize=7.5)
        ax.grid(axis="y", color=BORDER, lw=0.4, alpha=0.6)
        ax.set_title("Channel Jam Exposure", color=TEXT, fontsize=9, fontweight="bold", pad=4)
        if eng.jam_type == "Sweep" and eng.jam_db > -60:
            target = eng._sweep_step % n
            ax.axvline(target, color=YELLOW, lw=1.5, ls="--", alpha=0.7, zorder=5)
            ax.text(target + 0.1, 0.92, "sweep→", color=YELLOW, fontsize=7)

    def _redraw_timeline(self):
        eng = self.engine; ax = self.ax_ber
        ax.clear(); ax.set_facecolor(PANEL)
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
        ax2  = ax.twinx()
        ax2.set_facecolor(PANEL)
        ax2.tick_params(colors=MUTED, labelsize=7)
        bar_colors = [CHANNEL_COLORS[c % len(CHANNEL_COLORS)] for c in chs]
        ax.bar(hops, bers, color=bar_colors, alpha=0.7, width=0.75, zorder=3)
        ax2.plot(hops, snrs, color=YELLOW, lw=1.2, alpha=0.85, zorder=4)
        ax2.set_ylabel("SNR meas (dB)", color=YELLOW, fontsize=7)
        ax2.tick_params(axis='y', colors=YELLOW)
        ax.set_ylabel("BER", color=MUTED, fontsize=7.5)
        ax.set_xlabel("Hop #", color=MUTED, fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.grid(color=BORDER, lw=0.35, alpha=0.5)
        ax.axhline(0.1, color=RED, lw=0.8, ls="--", alpha=0.5)
        ax.text(hops[0], 0.12, "BER=0.1 threshold", color=RED, fontsize=7, alpha=0.7)

    def _redraw_status(self):
        eng = self.engine; ax = self.ax_heat
        ax.clear(); ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title("Jam Status", color=TEXT, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")
        lines = [
            ("Type",    eng.jam_type,                          ACCENT),
            ("Power",   f"{eng.jam_db:.0f} dB" if eng.jam_db > -60 else "OFF",
                        RED if eng.jam_db > -50 else MUTED),
            ("SNR cfg", f"{eng.snr_db:.0f} dB",                GREEN),
            ("Targets", self._target_str(),                    YELLOW),
            ("Hops",    str(len(self._hop_history)),            MUTED),
        ]
        for row, (k, v, col) in enumerate(lines):
            y = 0.88 - row * 0.18
            ax.text(0.05, y, k + ":", color=MUTED, fontsize=8, transform=ax.transAxes, va="center")
            ax.text(0.50, y, v, color=col, fontsize=8.5, fontweight="bold",
                    transform=ax.transAxes, va="center")
        n = eng.num_channels; cols_per_row = 4
        for i in range(n):
            row_i = i // cols_per_row; col_i = i % cols_per_row
            x = 0.05 + col_i * 0.23; y = 0.14 - row_i * 0.12
            col = RED if (eng.jam_db > -60 and eng.is_channel_jammed(i)) else GREEN + "80"
            ax.add_patch(mpatches.Circle((x, y), 0.04, transform=ax.transAxes,
                                          color=col, zorder=3, clip_on=False))
            ax.text(x + 0.06, y, f"CH{i+1}", color=MUTED, fontsize=7,
                    transform=ax.transAxes, va="center")

    def _target_str(self) -> str:
        eng = self.engine
        if eng.jam_db <= -60: return "None"
        if eng.jam_type == "Barrage": return "All channels"
        if eng.jam_type == "Sweep": return f"Sweeping (step {eng._sweep_step % eng.num_channels})"
        if not eng.jammed_channels: return "None selected"
        return ", ".join(f"CH{c+1}" for c in sorted(eng.jammed_channels))

    def update_hop(self, hop_idx: int, ch_idx: int, ber: float, snr_meas: float):
        eng = self.engine; n = eng.num_channels
        if len(self._jam_exposure) != n:
            self._jam_exposure = np.zeros(n)
        for i in range(n):
            hit = 1.0 if eng.is_channel_jammed(i) else 0.0
            self._jam_exposure[i] = 0.85 * self._jam_exposure[i] + 0.15 * hit
        self._ber_history.append(ber)
        self._snr_history.append(snr_meas)
        self._hop_history.append(hop_idx)
        self._ch_history.append(ch_idx)
        eng.advance_sweep()
        self._redraw_all()

    def reset_stats(self):
        eng = self.engine
        self._jam_exposure = np.zeros(eng.num_channels)
        self._ber_history = []; self._snr_history = []
        self._hop_history = []; self._ch_history  = []
        self._redraw_all()

    def rebuild(self):
        self.reset_stats()
        self._build_axes()


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW: Channel Energy Analysis Canvas  (Week 4)
# ═════════════════════════════════════════════════════════════════════════════
class EnergyAnalysisCanvas(FigureCanvas):
    """
    Three-panel energy / occupancy display:

    Top-left:   Per-channel energy bar chart with per-channel threshold overlay
                and noise floor markers.  Bars coloured by occupancy decision.

    Top-right:  Detection performance counters:
                  True Positive (TP), False Alarm (FA), Miss (Miss), TN
                  + running Pd = TP/(TP+Miss), Pfa = FA/(FA+TN)

    Bottom:     Occupancy heatmap — rows = channels, columns = hop history.
                Colour encodes measured energy (normalised); white border = occupied.
    """
    MAX_HISTORY = 80   # hops in heatmap

    def __init__(self, engine: FHSSEngine, detector: EnergyDetector, parent=None):
        self.engine   = engine
        self.detector = detector
        self.fig      = Figure(facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)

        n = engine.num_channels
        # Rolling history arrays  [channel × hop]
        self._energy_history  = np.full((n, self.MAX_HISTORY), np.nan)   # dB
        self._occ_history     = np.zeros((n, self.MAX_HISTORY), dtype=bool)
        self._gt_history      = np.zeros((n, self.MAX_HISTORY), dtype=bool)
        self._hop_labels      = []   # hop index strings for x-axis

        # Detection counters
        self._tp = self._fp = self._fn = self._tn = 0
        self._hop_count = 0

        self._build_axes()

    def _build_axes(self):
        self.fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1.3, 1.0],
                               width_ratios=[1.8, 1.0],
                               hspace=0.45, wspace=0.32,
                               left=0.09, right=0.97, top=0.93, bottom=0.09)
        self.ax_energy = self.fig.add_subplot(gs[0, 0])
        self.ax_stats  = self.fig.add_subplot(gs[0, 1])
        self.ax_heatmap= self.fig.add_subplot(gs[1, :])
        for ax in (self.ax_energy, self.ax_stats, self.ax_heatmap):
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        self.fig.patch.set_facecolor(BG)
        self._redraw_all()

    def _redraw_all(self):
        self._draw_energy_bars()
        self._draw_stats_panel()
        self._draw_heatmap()
        self.draw_idle()

    # ── Energy bars ──────────────────────────────────────────────────────────
    def _draw_energy_bars(self):
        ax  = self.ax_energy
        det = self.detector
        eng = self.engine
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title("Channel Energy (dB) vs Threshold", color=TEXT, fontsize=9,
                     fontweight="bold", pad=4)

        n  = eng.num_channels
        xs = np.arange(n)

        # Energy values (clip for display)
        energies   = np.clip(det.energy_db[:n], -40, 10)
        thresholds = det.thresholds_db[:n]
        noise_fl   = det._noise_floor_db[:n]
        occupied   = det.occupied[:n]

        bar_colors = [RED if occupied[i] else TEAL for i in range(n)]
        bars = ax.bar(xs, energies - noise_fl,   # show relative to noise floor
                      bottom=noise_fl,
                      color=bar_colors, width=0.65, alpha=0.85, zorder=3)

        # Threshold line (per-channel dots connected)
        ax.plot(xs, thresholds, color=YELLOW, lw=1.2, ls="--",
                marker="o", markersize=4, zorder=5, label="Threshold")

        # Noise floor line
        ax.plot(xs, noise_fl, color=MUTED, lw=0.8, ls=":",
                marker="s", markersize=3, zorder=4, label="Noise floor")

        ax.set_xlim(-0.5, n - 0.5)
        y_min = np.min(noise_fl) - 5
        y_max = max(np.max(energies), np.max(thresholds)) + 4
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"CH{i+1}" for i in range(n)], fontsize=7, color=MUTED)
        ax.set_ylabel("Energy (dB)", color=MUTED, fontsize=7.5)
        ax.grid(axis="y", color=BORDER, lw=0.35, alpha=0.5, zorder=0)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, loc="upper right")

        # Occupancy labels
        for i in range(n):
            label = "OCC" if occupied[i] else "free"
            col   = RED if occupied[i] else TEAL + "cc"
            ax.text(i, y_max - 1.5, label, ha="center", fontsize=6.5,
                    color=col, fontweight="bold" if occupied[i] else "normal")

    # ── Detection stats panel ─────────────────────────────────────────────────
    def _draw_stats_panel(self):
        ax = self.ax_stats
        ax.clear()
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.axis("off")
        ax.set_title("Detection Performance", color=TEXT, fontsize=9,
                     fontweight="bold", pad=4)

        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        total = tp + fp + fn + tn + 1e-9
        pd    = tp / (tp + fn + 1e-9)
        pfa   = fp / (fp + tn + 1e-9)
        acc   = (tp + tn) / total
        det   = self.detector

        rows = [
            ("Mode",    det.threshold_mode,                    ACCENT),
            ("Margin",  f"+{det.adaptive_margin:.1f} dB",      YELLOW if det.threshold_mode=="Adaptive" else MUTED),
            ("",        "",                                    MUTED),
            ("TP",      str(tp),                               GREEN),
            ("FP (FA)", str(fp),                               YELLOW),
            ("FN (Miss)",str(fn),                              ACCENT2),
            ("TN",      str(tn),                               MUTED),
            ("",        "",                                    MUTED),
            ("Pd",      f"{pd:.3f}",                           GREEN if pd > 0.8 else YELLOW),
            ("Pfa",     f"{pfa:.3f}",                          GREEN if pfa < 0.1 else RED),
            ("Accuracy",f"{acc:.3f}",                          TEAL),
            ("Hops",    str(self._hop_count),                  MUTED),
        ]

        for row_i, (k, v, col) in enumerate(rows):
            if not k:
                continue
            y = 0.97 - row_i * 0.083
            ax.text(0.04, y, k + ":", color=MUTED, fontsize=8,
                    transform=ax.transAxes, va="top")
            ax.text(0.52, y, v, color=col, fontsize=8.5, fontweight="bold",
                    transform=ax.transAxes, va="top")

    # ── Occupancy heatmap ─────────────────────────────────────────────────────
    def _draw_heatmap(self):
        ax  = self.ax_heatmap
        eng = self.engine
        n   = eng.num_channels
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.set_title("Occupancy Heatmap  (channel × hop history)", color=TEXT,
                     fontsize=9, fontweight="bold", pad=4)

        # Resize history arrays if channel count changed
        if self._energy_history.shape[0] != n:
            self._energy_history = np.full((n, self.MAX_HISTORY), np.nan)
            self._occ_history    = np.zeros((n, self.MAX_HISTORY), dtype=bool)
            self._gt_history     = np.zeros((n, self.MAX_HISTORY), dtype=bool)

        # Normalise energy for display (0–1 scale per row)
        disp = self._energy_history[:n, :]
        row_min = np.nanmin(disp, axis=1, keepdims=True)
        row_max = np.nanmax(disp, axis=1, keepdims=True)
        norm = (disp - row_min) / np.where(row_max - row_min > 0.1, row_max - row_min, 1.0)
        norm = np.nan_to_num(norm, nan=0.0)

        im = ax.imshow(norm, aspect="auto", origin="upper",
                       cmap=_occ_cmap, vmin=0, vmax=1,
                       interpolation="nearest", zorder=2)

        # Overlay white dots where detector said "occupied"
        occ = self._occ_history[:n, :]
        gt  = self._gt_history[:n, :]
        ch_ids, hop_ids = np.where(occ)
        if len(ch_ids) > 0:
            ax.scatter(hop_ids, ch_ids, s=5, c=YELLOW, alpha=0.6,
                       marker="s", zorder=4, label="Detected occ.")

        # Ground-truth jammed overlay (cross markers)
        gch, ghop = np.where(gt)
        if len(gch) > 0:
            ax.scatter(ghop, gch, s=8, c=RED, alpha=0.45,
                       marker="x", linewidths=0.8, zorder=5, label="True jam (GT)")

        ax.set_xlim(-0.5, self.MAX_HISTORY - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels([f"CH{i+1}" for i in range(n)], fontsize=7, color=MUTED)
        ax.set_xlabel("Hop index (newest →)", color=MUTED, fontsize=7.5)

        if len(ch_ids) > 0 or len(gch) > 0:
            ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER,
                      labelcolor=TEXT, loc="upper left", markerscale=1.8)

        # Divider line at current hop
        hop_ptr = self._hop_count % self.MAX_HISTORY
        ax.axvline(hop_ptr, color=ACCENT, lw=1.0, alpha=0.5, ls="--")

    # ── Public update called each hop ─────────────────────────────────────────
    def update_hop(self, hop_idx: int,
                   sense_results: list,   # list of dicts, one per channel
                   ground_truth: list):   # list of bool, one per channel
        """
        sense_results: output of EnergyDetector.sense() for every channel
        ground_truth:  True = channel is actually jammed this hop
        """
        eng = self.engine
        n   = eng.num_channels
        col = self._hop_count % self.MAX_HISTORY

        # Resize if needed
        if self._energy_history.shape[0] != n:
            self._energy_history = np.full((n, self.MAX_HISTORY), np.nan)
            self._occ_history    = np.zeros((n, self.MAX_HISTORY), dtype=bool)
            self._gt_history     = np.zeros((n, self.MAX_HISTORY), dtype=bool)

        for i, res in enumerate(sense_results):
            e_db  = res["energy_db"]
            occ   = res["occupied"]
            gt    = ground_truth[i]

            self._energy_history[i, col] = e_db
            self._occ_history[i, col]    = occ
            self._gt_history[i, col]     = gt

            # Confusion matrix update
            if gt and occ:
                self._tp += 1
            elif (not gt) and occ:
                self._fp += 1
            elif gt and (not occ):
                self._fn += 1
            else:
                self._tn += 1

        self._hop_count += 1
        self._redraw_all()

    def reset(self):
        eng = self.engine
        n   = eng.num_channels
        self._energy_history = np.full((n, self.MAX_HISTORY), np.nan)
        self._occ_history    = np.zeros((n, self.MAX_HISTORY), dtype=bool)
        self._gt_history     = np.zeros((n, self.MAX_HISTORY), dtype=bool)
        self._tp = self._fp = self._fn = self._tn = 0
        self._hop_count = 0
        self._redraw_all()

    def rebuild(self):
        self.reset()
        self._build_axes()


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW: Sensing Settings Panel  (Week 4)
# ═════════════════════════════════════════════════════════════════════════════
class SensingPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, detector: EnergyDetector):
        super().__init__("Spectrum Sensing")
        self.detector = detector
        self.setStyleSheet(f"""QGroupBox{{color:{TEXT};border:1px solid {BORDER};
            border-radius:8px;font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{TEAL};}}
            QLabel{{color:{MUTED};font-size:10px;}}""")
        lay = QVBoxLayout(self)
        lay.setSpacing(7)

        # Threshold mode
        mode_row = QWidget(); ml = QHBoxLayout(mode_row)
        ml.setContentsMargins(0,0,0,0); ml.setSpacing(6)
        ml.addWidget(QLabel("Threshold mode"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Adaptive", "Fixed dB", "Neyman-Pearson"])
        self.cb_mode.setCurrentText(detector.threshold_mode)
        self.cb_mode.setStyleSheet(f"""QComboBox{{background:{BG};color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:3px 6px;font-size:10px;}}
            QComboBox QAbstractItemView{{background:{PANEL};color:{TEXT};
            selection-background-color:{ACCENT}40;}}""")
        self.cb_mode.currentTextChanged.connect(self._on_mode_change)
        ml.addWidget(self.cb_mode, 1)
        lay.addWidget(mode_row)

        # Adaptive margin slider
        self._margin_widget = QWidget()
        mml = QVBoxLayout(self._margin_widget); mml.setContentsMargins(0,0,0,0); mml.setSpacing(2)
        margin_top = QWidget(); mt_lay = QHBoxLayout(margin_top)
        mt_lay.setContentsMargins(0,0,0,0)
        mt_lay.addWidget(QLabel("Adaptive margin"))
        self.margin_lbl = QLabel(f"+{detector.adaptive_margin:.0f} dB")
        self.margin_lbl.setStyleSheet(f"color:{YELLOW}; font-size:11px; font-weight:bold;")
        mt_lay.addStretch(); mt_lay.addWidget(self.margin_lbl)
        mml.addWidget(margin_top)
        self.sl_margin = QSlider(Qt.Horizontal)
        self.sl_margin.setRange(1, 20)
        self.sl_margin.setValue(int(detector.adaptive_margin))
        self.sl_margin.setStyleSheet(f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
            QSlider::handle:horizontal{{background:{YELLOW};width:14px;height:14px;margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{YELLOW}60;border-radius:2px;}}""")
        self.sl_margin.valueChanged.connect(self._on_margin_change)
        mml.addWidget(self.sl_margin)
        lay.addWidget(self._margin_widget)

        # Fixed threshold slider
        self._fixed_widget = QWidget()
        ffl = QVBoxLayout(self._fixed_widget); ffl.setContentsMargins(0,0,0,0); ffl.setSpacing(2)
        fixed_top = QWidget(); ft_lay = QHBoxLayout(fixed_top)
        ft_lay.setContentsMargins(0,0,0,0)
        ft_lay.addWidget(QLabel("Fixed threshold"))
        self.fixed_lbl = QLabel(f"{detector.fixed_threshold:.0f} dB")
        self.fixed_lbl.setStyleSheet(f"color:{ACCENT}; font-size:11px; font-weight:bold;")
        ft_lay.addStretch(); ft_lay.addWidget(self.fixed_lbl)
        ffl.addWidget(fixed_top)
        self.sl_fixed = QSlider(Qt.Horizontal)
        self.sl_fixed.setRange(-30, 10)
        self.sl_fixed.setValue(int(detector.fixed_threshold))
        self.sl_fixed.setStyleSheet(f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
            QSlider::handle:horizontal{{background:{ACCENT};width:14px;height:14px;margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{ACCENT}60;border-radius:2px;}}""")
        self.sl_fixed.valueChanged.connect(self._on_fixed_change)
        ffl.addWidget(self.sl_fixed)
        lay.addWidget(self._fixed_widget)

        # Description label
        self.desc_lbl = QLabel("")
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        lay.addWidget(self.desc_lbl)

        # Reset button
        btn_reset = QPushButton("Reset Detector")
        btn_reset.setStyleSheet(f"""QPushButton{{background:{PANEL};color:{TEAL};border:1px solid {TEAL}60;
            border-radius:5px;padding:4px;font-size:10px;font-weight:bold;}}
            QPushButton:hover{{background:{HOVER};}}""")
        btn_reset.clicked.connect(lambda: self.changed.emit())
        lay.addWidget(btn_reset)

        self._on_mode_change(detector.threshold_mode)

    def _on_mode_change(self, mode: str):
        self.detector.threshold_mode = mode
        self._margin_widget.setVisible(mode == "Adaptive")
        self._fixed_widget.setVisible(mode == "Fixed dB")
        descriptions = {
            "Adaptive":        "Threshold = per-channel EWMA noise floor + margin (CFAR-lite).  Adapts to changing noise conditions.",
            "Fixed dB":        "Threshold is a fixed absolute dB value applied to all channels.  Simple but not adaptive.",
            "Neyman-Pearson":  "Threshold set to achieve target Pfa ≈ 5% under H0 (noise-only).  Gaussian approximation.",
        }
        self.desc_lbl.setText(descriptions.get(mode, ""))
        self.changed.emit()

    def _on_margin_change(self, v: int):
        self.detector.adaptive_margin = float(v)
        self.margin_lbl.setText(f"+{v} dB")
        self.changed.emit()

    def _on_fixed_change(self, v: int):
        self.detector.fixed_threshold = float(v)
        self.fixed_lbl.setText(f"{v} dB")
        self.changed.emit()


# ═════════════════════════════════════════════════════════════════════════════
# ★ NEW Week 5: Adaptive FHSS Control Panel
# ═════════════════════════════════════════════════════════════════════════════
class AdaptivePanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, adaptive_engine: "AdaptiveFHSSEngine"):
        super().__init__("Adaptive FHSS")
        self.adaptive = adaptive_engine
        self.setStyleSheet(f"""QGroupBox{{color:{TEXT};border:1px solid {BORDER};
            border-radius:8px;font-size:11px;margin-top:8px;padding-top:6px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:10px;color:{GREEN};}}
            QLabel{{color:{MUTED};font-size:10px;}}""")
        lay = QVBoxLayout(self); lay.setSpacing(7)

        # Enable toggle
        en_row = QWidget(); el = QHBoxLayout(en_row)
        el.setContentsMargins(0, 0, 0, 0)
        self.cb_enable = QCheckBox("Enable Adaptive Mode")
        self.cb_enable.setChecked(adaptive_engine.enabled)
        self.cb_enable.setStyleSheet(f"color:{GREEN}; font-size:10px; font-weight:bold;")
        self.cb_enable.toggled.connect(self._on_toggle)
        el.addWidget(self.cb_enable); el.addStretch()
        lay.addWidget(en_row)

        # Strategy selector
        st_row = QWidget(); sl = QHBoxLayout(st_row)
        sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(6)
        sl.addWidget(QLabel("Strategy"))
        self.cb_strat = QComboBox()
        self.cb_strat.addItems(AdaptiveFHSSEngine.STRATEGIES)
        self.cb_strat.setCurrentText(adaptive_engine.strategy)
        self.cb_strat.setStyleSheet(f"""QComboBox{{background:{BG};color:{TEXT};
            border:1px solid {BORDER};border-radius:4px;padding:3px 6px;font-size:10px;}}
            QComboBox QAbstractItemView{{background:{PANEL};color:{TEXT};
            selection-background-color:{ACCENT}40;}}""")
        self.cb_strat.currentTextChanged.connect(self._on_strategy)
        sl.addWidget(self.cb_strat, 1)
        lay.addWidget(st_row)

        # Description
        self.desc = QLabel("")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        lay.addWidget(self.desc)

        # Live stats readout
        self.stats_lbl = QLabel("Hops: 0   Avoided: 0   Avoided %: 0.0%")
        self.stats_lbl.setStyleSheet(
            f"color:{TEAL}; font-size:9px; font-family:'Courier New';")
        lay.addWidget(self.stats_lbl)

        self._update_desc()

    def _on_toggle(self, v: bool):
        self.adaptive.enabled = v
        self.changed.emit()

    def _on_strategy(self, s: str):
        self.adaptive.strategy = s
        self._update_desc()
        self.changed.emit()

    def _update_desc(self):
        descs = {
            "Avoid Jammed":  "Skips channels flagged occupied by the energy detector. "
                             "Picks uniformly at random from clean channels. "
                             "Falls back to lowest-energy if all are flagged.",
            "Best Channel":  "Always hops to the channel with the lowest measured energy "
                             "(cleanest reception). Ignores all jammed channels first, "
                             "then picks the best of the remaining ones.",
        }
        self.desc.setText(descs.get(self.adaptive.strategy, ""))

    def refresh(self, tracker: "PerformanceTracker"):
        n       = tracker.n
        avoided = sum(tracker.adp_avoided)
        pct     = 100.0 * avoided / n if n > 0 else 0.0
        self.stats_lbl.setText(
            f"Hops: {n}   Avoided: {avoided}   Avoided %: {pct:.1f}%")


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
        self.console.append("┌─────────────────────────────────────────────────────────────────────────┐")
        self.console.append("│  FHSS RX Console v5  —  live demodulation + energy detection output      │")
        self.console.append("│  TX → modulate → channel (noise+jam) → demod → compare + sense          │")
        self.console.append("└─────────────────────────────────────────────────────────────────────────┘")
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

    def log_transmission(self, mod, tx_msg, rx_msg, tx_bits, rx_bits,
                         snr_db, jam_db, snr_meas, hop_idx, ch_label, ch_freq,
                         jam_type="Barrage", ch_jammed=False,
                         sense_result: dict = None):
        if self._paused: return
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ber = MsgCodec.ber(tx_bits, rx_bits)
        ok  = tx_msg.strip() == rx_msg.strip()
        match_col = GREEN if ok else RED

        self.console.append("")
        self._put(MUTED,       f"[{ts}] ")
        self._put(match_col,   f"[{'✓ MATCH' if ok else '✗ MISMATCH'}] ")
        self._put(ACCENT,      f"HOP#{hop_idx+1:02d} ")
        self._put(YELLOW,      f"{ch_label} {ch_freq:.2f}MHz  ")
        self._put(PURPLE,      f"{mod}  ")
        self._put(MUTED,       f"SNR={snr_meas:+.1f}dB (cfg:{snr_db:.0f})  ")
        if jam_db > -50:
            self._put(RED if ch_jammed else YELLOW,
                      f"JAM={jam_db:+.0f}dB [{jam_type}]{'⚡' if ch_jammed else '○'}  ")
        self._put(MUTED, f"BER={ber:.3f}")

        if sense_result:
            self.console.append("")
            occ_col = RED if sense_result["occupied"] else TEAL
            occ_str = "OCCUPIED" if sense_result["occupied"] else "free"
            self._put(MUTED,   "  Sense: ")
            self._put(occ_col, f"{occ_str}  ")
            self._put(MUTED,   f"E={sense_result['energy_db']:+.1f}dB  thr={sense_result['threshold_db']:+.1f}dB  ")
            self._put(MUTED,   f"NF={sense_result['noise_floor_db']:+.1f}dB")

        self.console.append("")
        self._put(MUTED, "  TX: "); self._put(GREEN,              repr(tx_msg))
        self.console.append("")
        self._put(MUTED, "  RX: "); self._put(TEXT if ok else RED, repr(rx_msg))

        n_show = min(48, len(tx_bits), len(rx_bits))
        if n_show > 0:
            self.console.append("")
            self._put(MUTED, "  TX bits[0:48]: ")
            self._put("#4fc3f7", "".join(str(b) for b in tx_bits[:n_show]))
            self.console.append("")
            self._put(MUTED, "  RX bits[0:48]: ")
            for i in range(n_show):
                tb, rb = int(tx_bits[i]), int(rx_bits[i])
                self._put("#4fc3f7" if tb == rb else RED, str(rb))
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
        self.status.setStyleSheet(f"color:{'#39d353' if ok else '#f85149'}; font-size:9px;")

    def log_adaptive(self, hop_idx: int, ch_label: str, ch_freq: float,
                     ber: float, snr_meas: float, avoided: bool, strategy: str):
        """Log adaptive FHSS channel selection decision."""
        if self._paused: return
        self.console.append("")
        self._put(GREEN,  f"  ↳ ADAPTIVE ")
        self._put(TEAL,   f"{ch_label} {ch_freq:.2f}MHz  ")
        self._put(MUTED,  f"[{strategy}]  ")
        self._put(MUTED,  f"SNR={snr_meas:+.1f}dB  BER={ber:.3f}  ")
        if avoided:
            self._put(GREEN, "✓ AVOIDED JAMMED CH")
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def log_hop_only(self, hop_idx, ch_label, ch_freq, mod, snr_db, jam_db, snr_meas,
                     jam_type="Barrage", ch_jammed=False, sense_result: dict = None):
        if self._paused: return
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.console.append("")
        self._put(MUTED,  f"[{ts}] ")
        self._put(BORDER, f"HOP#{hop_idx+1:02d} ")
        self._put(MUTED,  f"{ch_label} {ch_freq:.2f}MHz  {mod}  SNR={snr_meas:+.1f}dB")
        if jam_db > -50:
            self._put(RED if ch_jammed else YELLOW,
                      f"  JAM={jam_db:+.0f}dB [{jam_type}]{'⚡' if ch_jammed else '○'}")
        if sense_result:
            occ_col = RED if sense_result["occupied"] else TEAL
            self._put(occ_col,
                      f"  {'OCCUPIED' if sense_result['occupied'] else 'free'}"
                      f"  E={sense_result['energy_db']:+.1f}dB"
                      f"  thr={sense_result['threshold_db']:+.1f}dB")
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
        lay = QVBoxLayout(self); lay.setSpacing(8)

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
            font-family:'Courier New';}} QLineEdit:focus{{border:1px solid {ACCENT};}}""")
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
        self.bit_info.setText(f"{n_chars} chars → {n_chars*8} bits")

    def _send(self):
        msg = self.msg_input.text()
        if not msg: return
        mod = self.cb_mod.currentText()
        self._pending_msg = (mod, msg)
        self.pending_lbl.setText(f"⏳ Queued: [{mod}] {repr(msg)}")
        self.btn_send.setEnabled(False)

    def pop_pending(self):
        if self._pending_msg:
            v = self._pending_msg; self._pending_msg = None
            self.pending_lbl.setText(""); self.btn_send.setEnabled(True)
            return v
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Channel Panel
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
        lay = QVBoxLayout(self); lay.setSpacing(8)

        def slider_row(label, lo, hi, val, suffix, color, callback):
            row = QWidget(); rl = QVBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(2)
            top = QWidget(); tl = QHBoxLayout(top); tl.setContentsMargins(0,0,0,0)
            lbl = QLabel(label)
            val_lbl = QLabel(f"{val}{suffix}")
            val_lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold;")
            tl.addWidget(lbl); tl.addStretch(); tl.addWidget(val_lbl)
            rl.addWidget(top)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(val)
            sl.setStyleSheet(f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
                QSlider::handle:horizontal{{background:{color};width:14px;height:14px;
                    margin:-5px 0;border-radius:7px;}}
                QSlider::sub-page:horizontal{{background:{color}60;border-radius:2px;}}""")
            def _cb(v, lbl=val_lbl, sfx=suffix, cb=callback):
                lbl.setText(f"{v}{sfx}"); cb(v)
            sl.valueChanged.connect(_cb)
            rl.addWidget(sl); lay.addWidget(row)
            return sl

        self.sl_snr = slider_row("Noise (SNR)", 0, 40, int(engine.snr_db), " dB", GREEN,
                                  lambda v: (setattr(engine, 'snr_db', float(v)), self.changed.emit()))
        self.sl_jam = slider_row("Jamming power", -60, 20, max(-60, int(engine.jam_db)), " dB", RED,
                                  lambda v: (setattr(engine, 'jam_db', float(v)), self._on_jam_change()))

        jt_row = QWidget(); jt_lay = QHBoxLayout(jt_row)
        jt_lay.setContentsMargins(0,0,0,0); jt_lay.setSpacing(6)
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

        self.ch_target_group = QGroupBox("Target Channels")
        self.ch_target_group.setStyleSheet(f"""QGroupBox{{color:{YELLOW};border:1px solid {BORDER}40;
            border-radius:5px;font-size:9px;margin-top:5px;padding-top:4px;}}
            QGroupBox::title{{subcontrol-origin:margin;left:8px;}}""")
        self._ch_grid_lay = QGridLayout(self.ch_target_group)
        self._ch_grid_lay.setContentsMargins(4,4,4,4); self._ch_grid_lay.setSpacing(3)
        lay.addWidget(self.ch_target_group)

        self._rebuild_ch_buttons()

        self.jam_desc = QLabel("")
        self.jam_desc.setStyleSheet(f"color:{MUTED}; font-size:9px;")
        self.jam_desc.setWordWrap(True)
        lay.addWidget(self.jam_desc)
        self._on_jam_type_change(engine.jam_type)

    def _rebuild_ch_buttons(self):
        for btn in self._ch_btns:
            self._ch_grid_lay.removeWidget(btn); btn.deleteLater()
        self._ch_btns = []
        eng = self.engine; cols = 4
        for i in range(eng.num_channels):
            btn = QPushButton(f"CH{i+1}")
            btn.setCheckable(True); btn.setChecked(i in eng.jammed_channels)
            btn.setFixedHeight(22)
            btn.setStyleSheet(f"""QPushButton{{background:{BG};color:{MUTED};
                border:1px solid {BORDER};border-radius:3px;font-size:9px;}}
                QPushButton:checked{{background:{RED}30;color:{RED};border-color:{RED};}}
                QPushButton:hover{{background:{HOVER};}}""")
            btn.toggled.connect(lambda checked, idx=i: self._toggle_channel(idx, checked))
            self._ch_grid_lay.addWidget(btn, i // cols, i % cols)
            self._ch_btns.append(btn)

    def _toggle_channel(self, idx, checked):
        if checked: self.engine.jammed_channels.add(idx)
        else:        self.engine.jammed_channels.discard(idx)
        self.changed.emit()

    def _on_jam_change(self):
        self.changed.emit(); self._update_jam_desc()

    def _on_jam_type_change(self, jam_type=None):
        if jam_type is None: jam_type = self.cb_jam_type.currentText()
        self.engine.jam_type = jam_type
        self.ch_target_group.setVisible(jam_type in ("Spot", "Partial-Band"))
        self._update_jam_desc(); self.changed.emit()

    def _update_jam_desc(self):
        jt = self.engine.jam_type; jdb = self.engine.jam_db; on = jdb > -50
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
        else:
            self.jam_desc.setText(f"[{jt}]  {base}")
            self.jam_desc.setStyleSheet(f"color:{YELLOW}; font-size:9px;")

    def refresh_channels(self, engine):
        self.engine = engine; self._rebuild_ch_buttons()


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
        lay = QVBoxLayout(self); lay.setSpacing(5)

        def row(lbl, widget):
            r = QWidget(); rl = QHBoxLayout(r); rl.setContentsMargins(0,0,0,0)
            lb = QLabel(lbl); lb.setFixedWidth(130)
            rl.addWidget(lb); rl.addWidget(widget, 1)
            lay.addWidget(r); return widget

        self.sb_ch   = row("Channels",          QSpinBox());       self.sb_ch.setRange(2,12);     self.sb_ch.setValue(engine.num_channels)
        self.dsb_bf  = row("Base Freq (MHz)",   QDoubleSpinBox()); self.dsb_bf.setRange(100,6000);self.dsb_bf.setValue(engine.base_freq); self.dsb_bf.setSingleStep(10)
        self.dsb_bw  = row("Ch BW (MHz)",       QDoubleSpinBox()); self.dsb_bw.setRange(0.1,20);  self.dsb_bw.setValue(engine.channel_bw); self.dsb_bw.setSingleStep(0.5)
        self.sb_hi   = row("Hop Interval (ms)", QSpinBox());       self.sb_hi.setRange(1,500);    self.sb_hi.setValue(engine.hop_interval)
        self.sb_sl   = row("Sequence Length",   QSpinBox());       self.sb_sl.setRange(4,64);     self.sb_sl.setValue(engine.sequence_len)
        self.sb_sd   = row("Seed",              QSpinBox());       self.sb_sd.setRange(0,9999);   self.sb_sd.setValue(engine.seed)

        btn = QPushButton("Apply")
        btn.setStyleSheet(f"""QPushButton{{background:{PANEL};color:{ACCENT};border:1px solid {ACCENT}60;
            border-radius:5px;padding:5px;font-size:10px;font-weight:bold;}}
            QPushButton:hover{{background:{HOVER};}}""")
        btn.clicked.connect(self._apply)
        lay.addWidget(btn)

    def _apply(self):
        self.engine.reconfigure(
            num_channels = self.sb_ch.value(),
            base_freq    = self.dsb_bf.value(),
            channel_bw   = self.dsb_bw.value(),
            hop_interval = self.sb_hi.value(),
            sequence_len = self.sb_sl.value(),
            seed         = self.sb_sd.value(),
        )
        self.changed.emit()


# ═════════════════════════════════════════════════════════════════════════════
# Transport Bar
# ═════════════════════════════════════════════════════════════════════════════
class AnimBar(QWidget):
    hop_changed = pyqtSignal(int)

    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine   = engine; self._hop = 0; self._playing = False
        self._timer   = QTimer(); self._timer.timeout.connect(self._tick)
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)

        self.btn_play  = QPushButton("▶  Play");  self.btn_play.setFixedWidth(82)
        self.btn_reset = QPushButton("⏮  Reset"); self.btn_reset.setFixedWidth(82)
        self.btn_play.clicked.connect(self._toggle)
        self.btn_reset.clicked.connect(self._reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, engine.sequence_len-1)
        self.slider.valueChanged.connect(self._seek)

        self.hop_lbl = QLabel("Hop 0/0"); self.hop_lbl.setFixedWidth(80)
        self.hop_lbl.setStyleSheet(f"color:{TEXT}; font-size:10px;")

        self.speed_sb = QSpinBox()
        self.speed_sb.setRange(100, 3000); self.speed_sb.setValue(600)
        self.speed_sb.setSuffix(" ms"); self.speed_sb.setFixedWidth(82)
        self.speed_sb.setStyleSheet(f"QSpinBox{{background:{PANEL};color:{TEXT};border:1px solid {BORDER};border-radius:4px;padding:2px 4px;font-size:10px;}}")

        bs = f"""QPushButton{{background:{PANEL};color:{TEXT};border:1px solid {BORDER};
            border-radius:4px;padding:4px 8px;font-size:10px;}}
            QPushButton:hover{{background:{HOVER};}}"""
        ss = f"""QSlider::groove:horizontal{{background:{PANEL};height:4px;border-radius:2px;}}
            QSlider::handle:horizontal{{background:{ACCENT};width:13px;height:13px;margin:-5px 0;border-radius:7px;}}
            QSlider::sub-page:horizontal{{background:{ACCENT}40;border-radius:2px;}}"""
        self.btn_play.setStyleSheet(bs); self.btn_reset.setStyleSheet(bs)
        self.slider.setStyleSheet(ss)

        spd_lbl = QLabel("Speed:"); spd_lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        lay.addWidget(self.btn_play); lay.addWidget(self.btn_reset)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.hop_lbl); lay.addWidget(spd_lbl); lay.addWidget(self.speed_sb)

    def refresh(self, engine):
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
        for key in ["Channels","Hop Rate","Total BW","Modulation","SNR","Jamming","Jam Type","Sensing","Adaptive"]:
            fr = QFrame()
            fr.setStyleSheet(f"background:{PANEL};border:1px solid {BORDER};border-radius:5px;")
            fl = QVBoxLayout(fr); fl.setContentsMargins(8,3,8,3); fl.setSpacing(1)
            k = QLabel(key); k.setStyleSheet(f"color:{MUTED};font-size:9px;")
            v = QLabel("—"); v.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;")
            fl.addWidget(k); fl.addWidget(v)
            lay.addWidget(fr); self._lbs[key] = v
        lay.addStretch()
        self.refresh()

    def refresh(self, detector: "EnergyDetector" = None,
                adaptive: "AdaptiveFHSSEngine" = None):
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
        if detector:
            n_occ = int(np.sum(detector.occupied[:e.num_channels]))
            self._lbs["Sensing"].setText(f"{n_occ}/{e.num_channels} occ")
            self._lbs["Sensing"].setStyleSheet(
                f"color:{RED if n_occ > 0 else TEAL}; font-size:12px; font-weight:bold;")
        if adaptive:
            txt = adaptive.strategy[:6] if adaptive.enabled else "OFF"
            col = GREEN if adaptive.enabled else MUTED
            self._lbs["Adaptive"].setText(txt)
            self._lbs["Adaptive"].setStyleSheet(
                f"color:{col}; font-size:12px; font-weight:bold;")


# ═════════════════════════════════════════════════════════════════════════════
# Main Window
# ═════════════════════════════════════════════════════════════════════════════
class FHSSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "FHSS Simulation v6  —  Week 5: Adaptive FHSS · Channel Avoidance · Performance Comparison")
        self.resize(1560, 940)
        self.engine        = FHSSEngine()
        self.detector      = EnergyDetector(self.engine.num_channels)
        self.adaptive_eng  = AdaptiveFHSSEngine(self.engine, self.detector)
        self.perf_tracker  = PerformanceTracker()
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
        cw   = QWidget(); self.setCentralWidget(cw)
        root = QVBoxLayout(cw); root.setContentsMargins(10,8,10,8); root.setSpacing(7)

        hdr = QLabel("⟁  FHSS Simulation v6  ·  TX/RX  ·  Jamming  ·  Sensing  ·  Adaptive Hopping  ·  Week 5")
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

        t3 = QWidget(); tl3 = QVBoxLayout(t3); tl3.setContentsMargins(4,4,4,4)
        self.jam_analysis = JammingAnalysisCanvas(self.engine)
        tl3.addWidget(self.jam_analysis)
        self.tabs.addTab(t3, "⚡  Jamming Analysis")

        # ★ NEW Week 4 tab
        t4 = QWidget(); tl4 = QVBoxLayout(t4); tl4.setContentsMargins(4,4,4,4)
        self.energy_analysis = EnergyAnalysisCanvas(self.engine, self.detector)
        tl4.addWidget(self.energy_analysis)
        self.tabs.addTab(t4, "🔍  Energy / Sensing")

        # ★ NEW Week 5 tab
        t5 = QWidget(); tl5 = QVBoxLayout(t5); tl5.setContentsMargins(4,4,4,4)
        self.perf_canvas = PerformanceComparisonCanvas(self.perf_tracker, self.engine)
        tl5.addWidget(self.perf_canvas)
        self.tabs.addTab(t5, "📊  Performance")

        upper = QWidget(); uv = QVBoxLayout(upper)
        uv.setContentsMargins(0,0,0,0); uv.setSpacing(4)
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
        right = QWidget(); rv = QVBoxLayout(right)
        rv.setContentsMargins(6,0,0,0); rv.setSpacing(8)

        self.tx_panel = TXPanel(self.engine)
        rv.addWidget(self.tx_panel)

        self.ch_panel = ChannelPanel(self.engine)
        self.ch_panel.changed.connect(lambda: self.stats.refresh(self.detector, self.adaptive_eng))
        rv.addWidget(self.ch_panel)

        # ★ NEW: Adaptive FHSS panel
        self.adaptive_panel = AdaptivePanel(self.adaptive_eng)
        self.adaptive_panel.changed.connect(lambda: self.stats.refresh(self.detector, self.adaptive_eng))
        rv.addWidget(self.adaptive_panel)

        # ★ NEW: Sensing settings panel
        self.sensing_panel = SensingPanel(self.detector)
        self.sensing_panel.changed.connect(self._on_sensing_changed)
        rv.addWidget(self.sensing_panel)

        self.hop_panel = HopSettingsPanel(self.engine)
        self.hop_panel.changed.connect(self._on_config_changed)
        rv.addWidget(self.hop_panel)

        rv.addStretch()
        msplit.addWidget(right)
        msplit.setSizes([1040, 420])
        root.addWidget(msplit, 1)

    def _on_sensing_changed(self):
        self.energy_analysis._redraw_all()

    def _on_config_changed(self):
        n = self.engine.num_channels
        self.detector.reset(n)
        self.adaptive_eng.reset()
        self.perf_tracker.reset()
        self.spec.rebuild()
        self.hop_map._draw()
        self.jam_analysis.rebuild()
        self.energy_analysis.rebuild()
        self.perf_canvas.rebuild()
        self.stats.refresh(self.detector, self.adaptive_eng)
        self.anim.refresh(self.engine)
        self.ch_panel.refresh_channels(self.engine)

    def _on_hop(self, hop_idx: int):
        if hop_idx < 0:
            self.spec.update_hop(-1)
            self.hop_map.highlight_hop(-1)
            return

        eng    = self.engine
        mod    = self.tx_panel.cb_mod.currentText()
        eng.modulation = mod

        # ── 1. Probe ALL channels for sensing (shared between both modes) ─────
        probe_len = 256
        all_rx_iqs = []
        for ci in range(eng.num_channels):
            t    = np.arange(probe_len)
            base = np.exp(1j * 2 * np.pi * 0.25 * t) * 0.1
            rx   = Channel.apply(base, eng.snr_db, eng.jam_db,
                                 ch_idx=ci,
                                 jammed_channels=eng.jammed_channels,
                                 jam_type=eng.jam_type,
                                 sweep_step=eng._sweep_step,
                                 num_channels=eng.num_channels)
            all_rx_iqs.append(rx)

        ground_truth  = [eng.is_channel_jammed(ci) for ci in range(eng.num_channels)]
        sense_results = [
            self.detector.sense(ci, all_rx_iqs[ci], ground_truth[ci])
            for ci in range(eng.num_channels)
        ]

        # ── 2. Standard FHSS channel (fixed sequence) ─────────────────────────
        std_ch_idx  = eng.sequence[hop_idx % len(eng.sequence)]
        std_ch      = eng.channels[std_ch_idx]
        std_jammed  = eng.is_channel_jammed(std_ch_idx)
        active_sense= sense_results[std_ch_idx]

        # ── 3. Adaptive FHSS channel (detector-guided) ───────────────────────
        if self.adaptive_eng.enabled:
            adp_ch_idx = self.adaptive_eng.choose_channel()
        else:
            adp_ch_idx = std_ch_idx   # mirror standard when disabled
            self.adaptive_eng.channel_history.append(adp_ch_idx)
            self.adaptive_eng.last_ch_idx  = adp_ch_idx
            self.adaptive_eng.last_avoided = False

        adp_jammed  = eng.is_channel_jammed(adp_ch_idx)
        adp_avoided = self.adaptive_eng.last_avoided

        # ── 4. Simulate reception on BOTH channels ────────────────────────────
        def simulate_channel(ch_i, tx_iq_arg=None):
            """Return (rx_iq, ber, snr_meas, rx_msg) for a given channel."""
            if tx_iq_arg is not None:
                rx = Channel.apply(tx_iq_arg, eng.snr_db, eng.jam_db,
                                   ch_idx=ch_i,
                                   jammed_channels=eng.jammed_channels,
                                   jam_type=eng.jam_type,
                                   sweep_step=eng._sweep_step,
                                   num_channels=eng.num_channels)
                return rx
            return all_rx_iqs[ch_i]

        pending = self.tx_panel.pop_pending()
        if pending:
            p_mod, p_msg = pending
            eng.modulation = p_mod; mod = p_mod
            tx_bits = MsgCodec.encode(p_msg)
            tx_iq   = Modulator.modulate(mod, tx_bits)

            # Standard path
            std_rx_iq  = simulate_channel(std_ch_idx, tx_iq)
            std_rx_bits= Demodulator.demodulate(mod, std_rx_iq, len(tx_bits))
            std_rx_msg = MsgCodec.decode(std_rx_bits)
            std_snr    = Demodulator.measure_snr(std_rx_iq)
            std_ber    = MsgCodec.ber(tx_bits, std_rx_bits)

            # Adaptive path
            adp_rx_iq  = simulate_channel(adp_ch_idx, tx_iq)
            adp_rx_bits= Demodulator.demodulate(mod, adp_rx_iq, len(tx_bits))
            adp_snr    = Demodulator.measure_snr(adp_rx_iq)
            adp_ber    = MsgCodec.ber(tx_bits, adp_rx_bits)

            self.spec.update_hop(hop_idx, tx_iq=std_rx_iq)
            self.hop_map.highlight_hop(hop_idx)
            self.console.log_transmission(
                mod=mod, tx_msg=p_msg, rx_msg=std_rx_msg,
                tx_bits=tx_bits, rx_bits=std_rx_bits,
                snr_db=eng.snr_db, jam_db=eng.jam_db,
                snr_meas=std_snr,
                hop_idx=hop_idx, ch_label=std_ch["label"], ch_freq=std_ch["freq"],
                jam_type=eng.jam_type, ch_jammed=std_jammed,
                sense_result=active_sense,
            )
            # Log adaptive decision if different channel
            if adp_ch_idx != std_ch_idx:
                adp_ch = eng.channels[adp_ch_idx]
                self.console.log_adaptive(hop_idx, adp_ch["label"], adp_ch["freq"],
                                          adp_ber, adp_snr, adp_avoided,
                                          self.adaptive_eng.strategy)
        else:
            std_rx_iq = all_rx_iqs[std_ch_idx]
            adp_rx_iq = all_rx_iqs[adp_ch_idx]
            std_snr   = Demodulator.measure_snr(std_rx_iq)
            adp_snr   = Demodulator.measure_snr(adp_rx_iq)
            std_ber   = 0.0
            adp_ber   = 0.0

            self.spec.update_hop(hop_idx, tx_iq=std_rx_iq)
            self.hop_map.highlight_hop(hop_idx)
            self.console.log_hop_only(hop_idx, std_ch["label"], std_ch["freq"],
                                      mod, eng.snr_db, eng.jam_db, std_snr,
                                      jam_type=eng.jam_type, ch_jammed=std_jammed,
                                      sense_result=active_sense)
            if adp_ch_idx != std_ch_idx:
                adp_ch = eng.channels[adp_ch_idx]
                self.console.log_adaptive(hop_idx, adp_ch["label"], adp_ch["freq"],
                                          adp_ber, adp_snr, adp_avoided,
                                          self.adaptive_eng.strategy)

        # ── 5. Update all analysis canvases ──────────────────────────────────
        self.jam_analysis.update_hop(hop_idx, std_ch_idx, std_ber, std_snr)
        self.energy_analysis.update_hop(hop_idx, sense_results, ground_truth)

        self.perf_tracker.record(
            hop_idx=hop_idx,
            std_ber=std_ber,   std_snr=std_snr,
            std_ch=std_ch_idx, std_jammed=std_jammed,
            adp_ber=adp_ber,   adp_snr=adp_snr,
            adp_ch=adp_ch_idx, adp_jammed=adp_jammed,
            adp_avoided=adp_avoided,
        )
        self.perf_canvas.update()
        self.adaptive_panel.refresh(self.perf_tracker)

        self.anim._lbl()
        self.stats.refresh(self.detector, self.adaptive_eng)


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