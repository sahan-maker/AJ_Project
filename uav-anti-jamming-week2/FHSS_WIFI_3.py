"""
Frequency Hopping Spread Spectrum (FHSS) Simulation
Visualizes frequency vs time hopping pattern with channel list and hopping sequence.
Requires: PyQt5, matplotlib, numpy
Install: pip install PyQt5 matplotlib numpy
"""

import sys
import numpy as np
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QComboBox,
    QSplitter, QFrame, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#00d4ff"
ACCENT2  = "#ff6b35"
GREEN    = "#39d353"
MUTED    = "#8b949e"
TEXT     = "#e6edf3"
HOVER    = "#1f2937"

CHANNEL_COLORS = [
    "#00d4ff", "#ff6b35", "#39d353", "#f0e68c",
    "#da70d6", "#87ceeb", "#ff69b4", "#98fb98",
    "#ffa500", "#7b68ee", "#20b2aa", "#ff4500",
    "#9370db", "#3cb371", "#b8860b", "#4682b4",
]


# ── FHSS Engine ────────────────────────────────────────────────────────────────
class FHSSEngine:
    def __init__(self):
        self.num_channels   = 8
        self.base_freq      = 2400.0   # MHz
        self.channel_bw     = 1.0      # MHz
        self.hop_interval   = 10       # ms
        self.sequence_len   = 32
        self.seed           = 42
        self.pattern        = "Pseudo-Random"
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
            half = self.num_channels // 2
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


# ── Matplotlib Canvas ───────────────────────────────────────────────────────────
class FHSSCanvas(FigureCanvas):
    def __init__(self, engine: FHSSEngine, parent=None):
        self.engine = engine
        self.fig = Figure(figsize=(10, 5), facecolor=BG)
        super().__init__(self.fig)
        self.setParent(parent)
        self._current_hop = -1
        self._draw_static()

    def _draw_static(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        ax = self.ax
        ax.set_facecolor(BG)
        self.fig.patch.set_facecolor(BG)

        eng    = self.engine
        times  = eng.time_axis
        freqs  = eng.freq_axis
        seq    = eng.sequence
        chans  = eng.channels
        bw     = eng.channel_bw

        # Draw channel bands (faint horizontal stripes)
        for ch in chans:
            ax.axhspan(ch["freq"] - bw * 0.45, ch["freq"] + bw * 0.45,
                       alpha=0.06, color=ch["color"])
            ax.axhline(ch["freq"], color=ch["color"], alpha=0.15,
                       linewidth=0.5, linestyle="--")

        # Draw hop segments
        for i in range(len(times)):
            x0 = times[i]
            x1 = times[i] + eng.hop_interval * 0.92
            yc = freqs[i]
            ch = chans[seq[i]]

            # Filled rectangle per hop
            rect = mpatches.FancyBboxPatch(
                (x0, yc - bw * 0.4), eng.hop_interval * 0.92, bw * 0.8,
                boxstyle="round,pad=0.01",
                linewidth=1.2,
                edgecolor=ch["color"],
                facecolor=ch["color"] + "40",
            )
            ax.add_patch(rect)

            # Channel label inside box
            ax.text(x0 + eng.hop_interval * 0.46, yc, ch["label"],
                    ha="center", va="center", fontsize=6.5,
                    color=ch["color"], fontweight="bold", alpha=0.9)

        # Connect hops with a trajectory line
        ax.plot(
            [t + eng.hop_interval * 0.46 for t in times],
            freqs,
            color=ACCENT, linewidth=1, alpha=0.35,
            linestyle=":", zorder=5
        )

        # Axes styling
        ax.set_xlim(-eng.hop_interval * 0.5,
                    times[-1] + eng.hop_interval * 1.5)
        freq_vals = [c["freq"] for c in chans]
        margin    = bw * 0.8
        ax.set_ylim(min(freq_vals) - margin, max(freq_vals) + margin)

        ax.set_xlabel("Time  (ms)", color=TEXT, fontsize=9, labelpad=8)
        ax.set_ylabel("Frequency  (MHz)", color=TEXT, fontsize=9, labelpad=8)
        ax.set_title("FHSS — Frequency vs Time", color=TEXT,
                     fontsize=11, fontweight="bold", pad=12)

        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

        ax.set_yticks([c["freq"] for c in chans])
        ax.set_yticklabels([f"{c['freq']:.1f}" for c in chans],
                           fontsize=7.5, color=MUTED)

        # X ticks every 5 hops
        step = max(1, len(times) // 8)
        ax.set_xticks([times[i] for i in range(0, len(times), step)])
        ax.set_xticklabels(
            [str(times[i]) for i in range(0, len(times), step)],
            fontsize=7.5, color=MUTED
        )

        ax.grid(axis="both", color=BORDER, linewidth=0.4, alpha=0.5)

        # Legend
        handles = [
            mpatches.Patch(color=c["color"], label=f"{c['label']} {c['freq']} MHz")
            for c in chans
        ]
        legend = ax.legend(
            handles=handles, loc="upper right",
            fontsize=6.8, ncol=2,
            framealpha=0.25, facecolor=PANEL,
            edgecolor=BORDER, labelcolor=TEXT,
        )

        self._current_hop = -1
        self._highlight_patch = None
        self.fig.tight_layout(pad=1.2)
        self.draw()

    def highlight_hop(self, hop_index: int):
        """Redraw current-hop highlight without full redraw."""
        eng  = self.engine
        ax   = self.ax
        bw   = eng.channel_bw
        times = eng.time_axis
        freqs = eng.freq_axis

        # Remove previous highlight
        if self._highlight_patch:
            try:
                self._highlight_patch.remove()
            except Exception:
                pass
            self._highlight_patch = None

        if 0 <= hop_index < len(times):
            i   = hop_index
            x0  = times[i]
            yc  = freqs[i]
            rect = mpatches.FancyBboxPatch(
                (x0, yc - bw * 0.48), eng.hop_interval * 0.92, bw * 0.96,
                boxstyle="round,pad=0.01",
                linewidth=2.5,
                edgecolor="#ffffff",
                facecolor="#ffffff22",
                zorder=10,
            )
            ax.add_patch(rect)
            self._highlight_patch = rect

        self._current_hop = hop_index
        self.draw_idle()


# ── Channel Table ───────────────────────────────────────────────────────────────
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
        self._style()
        self.refresh()

    def _style(self):
        self.setStyleSheet(f"""
            QTableWidget {{
                background: {PANEL}; color: {TEXT};
                border: 1px solid {BORDER}; gridline-color: {BORDER};
                font-size: 12px;
            }}
            QTableWidget::item:alternate {{ background: {BG}; }}
            QTableWidget::item:selected  {{ background: {ACCENT}40; color: {TEXT}; }}
            QHeaderView::section {{
                background: {BG}; color: {MUTED};
                border: none; border-bottom: 1px solid {BORDER};
                padding: 6px; font-size: 11px;
            }}
        """)

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
            if hop_counts[i] > 0:
                hops.setForeground(QColor(GREEN))
            else:
                hops.setForeground(QColor(MUTED))

            self.setItem(i, 0, lbl)
            self.setItem(i, 1, freq)
            self.setItem(i, 2, hops)

    def highlight_channel(self, ch_index: int):
        self.selectRow(ch_index)


# ── Sequence Display ────────────────────────────────────────────────────────────
class SequenceWidget(QFrame):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("Hopping Sequence")
        title.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:600;")
        layout.addWidget(title)

        self.seq_label = QLabel()
        self.seq_label.setWordWrap(True)
        self.seq_label.setStyleSheet(f"color:{TEXT}; font-family:'Courier New'; font-size:11px;")
        layout.addWidget(self.seq_label)

        self.setStyleSheet(f"background:{PANEL}; border:1px solid {BORDER}; border-radius:6px;")
        self.refresh()

    def refresh(self):
        eng  = self.engine
        html = ""
        for i, ch in enumerate(eng.sequence):
            color = eng.channels[ch]["color"]
            label = eng.channels[ch]["label"]
            html += f'<span style="color:{color}; font-weight:bold;">{label}</span>'
            if i < len(eng.sequence) - 1:
                html += f'<span style="color:{BORDER}"> → </span>'
        self.seq_label.setText(html)

    def highlight_hop(self, hop_index: int):
        # Rebuild with highlighted position
        eng  = self.engine
        html = ""
        for i, ch in enumerate(eng.sequence):
            color = eng.channels[ch]["color"]
            label = eng.channels[ch]["label"]
            if i == hop_index:
                html += (f'<span style="background:{color}40;'
                         f'color:{color}; font-weight:bold;'
                         f'border:1px solid {color}; border-radius:3px;'
                         f'padding:1px 3px;">{label}</span>')
            else:
                html += f'<span style="color:{color}66; font-weight:bold;">{label}</span>'
            if i < len(eng.sequence) - 1:
                html += f'<span style="color:{BORDER}"> → </span>'
        self.seq_label.setText(html)


# ── Stats Bar ──────────────────────────────────────────────────────────────────
class StatsBar(QWidget):
    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine = engine
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self._labels = {}
        for key in ["Channels", "Hop Rate", "BW/Channel", "Total BW", "Seq Len"]:
            frame = QFrame()
            frame.setStyleSheet(f"background:{PANEL}; border:1px solid {BORDER}; border-radius:6px;")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(10, 6, 10, 6)
            fl.setSpacing(2)
            k = QLabel(key)
            k.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            v = QLabel("—")
            v.setStyleSheet(f"color:{ACCENT}; font-size:13px; font-weight:bold;")
            fl.addWidget(k)
            fl.addWidget(v)
            layout.addWidget(frame)
            self._labels[key] = v
        layout.addStretch()
        self.refresh()

    def refresh(self):
        eng = self.engine
        hop_rate = 1000 / eng.hop_interval  # hops/sec
        total_bw = eng.num_channels * eng.channel_bw
        self._labels["Channels"].setText(str(eng.num_channels))
        self._labels["Hop Rate"].setText(f"{hop_rate:.0f} h/s")
        self._labels["BW/Channel"].setText(f"{eng.channel_bw:.1f} MHz")
        self._labels["Total BW"].setText(f"{total_bw:.1f} MHz")
        self._labels["Seq Len"].setText(str(eng.sequence_len))


# ── Control Panel ───────────────────────────────────────────────────────────────
class ControlPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, engine: FHSSEngine):
        super().__init__("Configuration")
        self.engine = engine
        self.setStyleSheet(f"""
            QGroupBox {{
                color:{TEXT}; border:1px solid {BORDER};
                border-radius:8px; font-size:12px;
                margin-top:8px; padding-top:6px;
            }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; color:{ACCENT}; }}
            QLabel {{ color:{MUTED}; font-size:11px; }}
            QSpinBox, QDoubleSpinBox, QComboBox {{
                background:{BG}; color:{TEXT}; border:1px solid {BORDER};
                border-radius:4px; padding:3px 6px; font-size:11px;
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width:16px; background:{PANEL};
            }}
        """)

        grid = QVBoxLayout(self)
        grid.setSpacing(8)

        def row(lbl, widget):
            r = QWidget()
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(lbl)
            l.setFixedWidth(110)
            h.addWidget(l)
            h.addWidget(widget, 1)
            grid.addWidget(r)
            return widget

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

        self.cb_pattern = row("Pattern", QComboBox())
        self.cb_pattern.addItems(["Pseudo-Random", "Sequential", "Interleaved"])

        btn = QPushButton("Apply & Regenerate")
        btn.setStyleSheet(f"""
            QPushButton {{
                background:{ACCENT}; color:#000; border:none;
                border-radius:6px; padding:7px; font-size:12px; font-weight:bold;
            }}
            QPushButton:hover {{ background:#33ddff; }}
        """)
        btn.clicked.connect(self._apply)
        grid.addWidget(btn)

    def _apply(self):
        self.engine.reconfigure(
            num_channels  = self.sb_channels.value(),
            base_freq     = self.dsb_base.value(),
            channel_bw    = self.dsb_bw.value(),
            hop_interval  = self.sb_interval.value(),
            sequence_len  = self.sb_seqlen.value(),
            seed          = self.sb_seed.value(),
            pattern       = self.cb_pattern.currentText(),
        )
        self.changed.emit()


# ── Animation Controls ──────────────────────────────────────────────────────────
class AnimBar(QWidget):
    hop_changed = pyqtSignal(int)

    def __init__(self, engine: FHSSEngine):
        super().__init__()
        self.engine      = engine
        self._hop        = 0
        self._playing    = False
        self._timer      = QTimer()
        self._timer.timeout.connect(self._tick)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(90)
        self.btn_play.clicked.connect(self._toggle)

        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.setFixedWidth(90)
        self.btn_reset.clicked.connect(self._reset)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, engine.sequence_len - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._seek)

        self.hop_lbl = QLabel("Hop: 0 / 0")
        self.hop_lbl.setFixedWidth(100)
        self.hop_lbl.setStyleSheet(f"color:{TEXT}; font-size:11px;")

        self.speed_lbl = QLabel("Speed:")
        self.speed_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.speed_sb = QSpinBox()
        self.speed_sb.setRange(50, 2000)
        self.speed_sb.setValue(600)
        self.speed_sb.setSuffix(" ms")
        self.speed_sb.setFixedWidth(90)
        self.speed_sb.setStyleSheet(f"""
            QSpinBox {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};
                        border-radius:4px; padding:2px 4px; font-size:11px; }}
        """)

        for w in [self.btn_play, self.btn_reset]:
            w.setStyleSheet(f"""
                QPushButton {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER};
                               border-radius:5px; padding:5px 10px; font-size:11px; }}
                QPushButton:hover {{ background:{HOVER}; }}
            """)

        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{PANEL}; height:4px; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{ACCENT}; width:14px; height:14px;
                                          margin:-5px 0; border-radius:7px; }}
            QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
        """)

        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_reset)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.hop_lbl)
        layout.addWidget(self.speed_lbl)
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


# ── Main Window ─────────────────────────────────────────────────────────────────
class FHSSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FHSS Simulation  —  Frequency Hopping Spread Spectrum")
        self.resize(1280, 780)
        self.engine = FHSSEngine()
        self._build_ui()
        self._apply_global_style()

    def _apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; }}
            QSplitter::handle {{ background:{BORDER}; }}
            QScrollBar:vertical {{ background:{BG}; width:8px; border-radius:4px; }}
            QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:4px; }}
        """)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # ── Header ───────────────────────────────────────────────────────────
        header = QLabel("⟁  FHSS Simulation")
        header.setStyleSheet(f"""
            color:{ACCENT}; font-size:18px; font-weight:700;
            font-family:'Courier New'; letter-spacing:2px;
            border-bottom:1px solid {BORDER}; padding-bottom:6px;
        """)
        root.addWidget(header)

        # ── Stats ─────────────────────────────────────────────────────────────
        self.stats = StatsBar(self.engine)
        root.addWidget(self.stats)

        # ── Main splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        # Left: canvas + anim bar + sequence
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(8)

        self.canvas = FHSSCanvas(self.engine)
        lv.addWidget(self.canvas, 4)

        self.anim_bar = AnimBar(self.engine)
        self.anim_bar.hop_changed.connect(self._on_hop)
        lv.addWidget(self.anim_bar)

        self.seq_widget = SequenceWidget(self.engine)
        lv.addWidget(self.seq_widget, 1)

        splitter.addWidget(left)

        # Right: controls + channel table
        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.setSpacing(8)

        self.ctrl = ControlPanel(self.engine)
        self.ctrl.changed.connect(self._on_config_changed)
        rv.addWidget(self.ctrl)

        tbl_lbl = QLabel("Channel List")
        tbl_lbl.setStyleSheet(f"color:{MUTED}; font-size:11px; font-weight:600;")
        rv.addWidget(tbl_lbl)

        self.table = ChannelTable(self.engine)
        rv.addWidget(self.table)

        splitter.addWidget(right)
        splitter.setSizes([820, 360])
        root.addWidget(splitter, 1)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_config_changed(self):
        self.canvas._draw_static()
        self.stats.refresh()
        self.table.refresh()
        self.seq_widget.refresh()
        self.anim_bar.refresh(self.engine)

    def _on_hop(self, hop_index: int):
        self.canvas.highlight_hop(hop_index)
        if hop_index >= 0:
            ch_idx = self.engine.sequence[hop_index]
            self.table.highlight_channel(ch_idx)
            self.seq_widget.highlight_hop(hop_index)
            self.anim_bar._update_label()
        else:
            self.table.clearSelection()
            self.seq_widget.refresh()


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.Base,            QColor(PANEL))
    palette.setColor(QPalette.AlternateBase,   QColor(BG))
    palette.setColor(QPalette.Text,            QColor(TEXT))
    palette.setColor(QPalette.Button,          QColor(PANEL))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT))
    app.setPalette(palette)

    win = FHSSWindow()
    win.show()
    sys.exit(app.exec_())
