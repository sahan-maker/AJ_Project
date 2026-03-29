# Week 1: Multi-Signal RF Simulator (UAV Anti‑Jamming Research)
# Supports: Sine, BPSK, QPSK, Chirp, RTCM-like digital burst
# GUI built with PyQt5

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RFSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi‑Signal RF Environment Simulator")
        self.resize(1100, 750)

        layout = QVBoxLayout()

        # ------------------ Controls ------------------
        control_group = QGroupBox("Simulation Controls")
        control_layout = QGridLayout()

        # Signal type
        self.signal_type = QComboBox()
        self.signal_type.addItems([
            "Sine",
            "BPSK",
            "QPSK",
            "Chirp",
            "RTCM-like"
        ])

        # Frequency
        self.freq = QDoubleSpinBox()
        self.freq.setRange(1, 5000)
        self.freq.setValue(100)
        self.freq.setSuffix(" Hz")

        # Sampling rate
        self.fs = QSpinBox()
        self.fs.setRange(1000, 100000)
        self.fs.setValue(10000)

        # Duration
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 5)
        self.duration.setValue(1)
        self.duration.setSuffix(" s")

        # Noise level
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0, 2)
        self.noise.setSingleStep(0.05)
        self.noise.setValue(0.1)

        # Jammer amplitude
        self.jammer = QDoubleSpinBox()
        self.jammer.setRange(0, 3)
        self.jammer.setValue(0.5)

        control_layout.addWidget(QLabel("Signal Type"), 0, 0)
        control_layout.addWidget(self.signal_type, 0, 1)

        control_layout.addWidget(QLabel("Carrier Frequency"), 1, 0)
        control_layout.addWidget(self.freq, 1, 1)

        control_layout.addWidget(QLabel("Sampling Rate"), 2, 0)
        control_layout.addWidget(self.fs, 2, 1)

        control_layout.addWidget(QLabel("Duration"), 3, 0)
        control_layout.addWidget(self.duration, 3, 1)

        control_layout.addWidget(QLabel("Noise Level"), 4, 0)
        control_layout.addWidget(self.noise, 4, 1)

        control_layout.addWidget(QLabel("Jammer Amplitude"), 5, 0)
        control_layout.addWidget(self.jammer, 5, 1)

        control_group.setLayout(control_layout)

        # ------------------ Buttons ------------------
        button_layout = QHBoxLayout()

        self.btn_generate = QPushButton("Generate Signal")
        self.btn_noise = QPushButton("Add Noise")
        self.btn_jam = QPushButton("Add Jammer")
        self.btn_fft = QPushButton("Show Spectrum")
        self.btn_reset = QPushButton("Reset")

        self.btn_generate.clicked.connect(self.generate_signal)
        self.btn_noise.clicked.connect(self.add_noise)
        self.btn_jam.clicked.connect(self.add_jammer)
        self.btn_fft.clicked.connect(self.show_fft)
        self.btn_reset.clicked.connect(self.reset_signal)

        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_noise)
        button_layout.addWidget(self.btn_jam)
        button_layout.addWidget(self.btn_fft)
        button_layout.addWidget(self.btn_reset)

        # ------------------ Plot ------------------
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(control_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        self.t = None
        self.signal = None

    # ------------------ Signal Generators ------------------
    def generate_signal(self):
        fs = self.fs.value()
        duration = self.duration.value()
        f = self.freq.value()

        self.t = np.arange(0, duration, 1/fs)
        sig_type = self.signal_type.currentText()

        if sig_type == "Sine":
            self.signal = np.sin(2*np.pi*f*self.t)

        elif sig_type == "BPSK":
            bits = np.random.choice([-1, 1], size=len(self.t)//50)
            symbols = np.repeat(bits, 50)
            carrier = np.sin(2*np.pi*f*self.t[:len(symbols)])
            self.signal = symbols * carrier

        elif sig_type == "QPSK":
            symbols = np.random.randint(0, 4, size=len(self.t)//50)
            phase = symbols * (np.pi/2)
            phase = np.repeat(phase, 50)
            self.signal = np.cos(2*np.pi*f*self.t[:len(phase)] + phase)

        elif sig_type == "Chirp":
            f1 = f
            f2 = f*5
            self.signal = np.sin(2*np.pi*(f1 + (f2-f1)*self.t/self.t[-1]) * self.t)

        elif sig_type == "RTCM-like":
            # burst digital packet simulation
            packet = np.random.choice([-1, 1], size=len(self.t)//200)
            burst = np.repeat(packet, 200)
            carrier = np.sin(2*np.pi*f*self.t[:len(burst)])
            self.signal = burst * carrier

        self.plot_signal("Generated Signal")

    def add_noise(self):
        if self.signal is None:
            return
        noise = np.random.normal(0, self.noise.value(), len(self.signal))
        self.signal = self.signal + noise
        self.plot_signal("Signal + Noise")

    def add_jammer(self):
        if self.signal is None:
            return
        jammer_freq = self.freq.value() * 1.8
        jammer = self.jammer.value() * np.sin(2*np.pi*jammer_freq*self.t[:len(self.signal)])
        self.signal = self.signal + jammer
        self.plot_signal("Signal + Jammer")

    def show_fft(self):
        if self.signal is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        fft = np.fft.fft(self.signal)
        freq = np.fft.fftfreq(len(fft), d=1/self.fs.value())

        ax.plot(freq, np.abs(fft))
        ax.set_title("Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")

        self.canvas.draw()

    def reset_signal(self):
        self.signal = None
        self.t = None
        self.figure.clear()
        self.canvas.draw()

    def plot_signal(self, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.t[:len(self.signal)], self.signal)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RFSimulator()
    window.show()
    sys.exit(app.exec_())
