## GNU Radio flowgraph for PlutoSDR Wi-Fi Band Sweep (2.4 GHz)
## Real-time FFT monitoring in 20 MHz chunks

from gnuradio import gr
from gnuradio import blocks, qtgui, analog
from gnuradio import eng_notation
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
import sys, sip, time
from PyQt5 import Qt

class pluto_wifi_sweep(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Pluto Wi-Fi Sweep")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Pluto Wi-Fi Sweep 2.4GHz")

        # GUI Layout
        self.top_layout = Qt.QVBoxLayout(self)

        # Variables
        self.center_freqs = [2.410e9, 2.430e9, 2.450e9, 2.470e9]  # sub-band centers
        self.samp_rate = 20e6

        # PlutoSDR Source
        self.pluto_source = blocks.null_source(gr.sizeof_gr_complex)  # placeholder
        try:
            from gnuradio import iio
            self.pluto_source = iio.pluto_source('', int(self.samp_rate), int(20e6), 1024, False, True, True, 'manual', 0.0)
        except ImportError:
            print("iio Pluto module not installed")

        # QT GUI FFT Sink
        self.fft_sink = qtgui.freq_sink_c(
            1024, firdes.WIN_BLACKMAN_hARRIS, 0, self.samp_rate, "FFT", 1)
        self.fft_sink.set_update_time(0.1)
        self.fft_sink.enable_grid(True)
        self._fft_win = sip.wrapinstance(self.fft_sink.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._fft_win)

        # Connect Source to FFT Sink
        self.connect((self.pluto_source, 0), (self.fft_sink, 0))

    def sweep(self):
        for f in self.center_freqs:
            print(f"Setting center frequency to {f/1e9:.3f} GHz")
            self.pluto_source.set_params(f, self.samp_rate, 0)
            self.start()
            Qt.QApplication.processEvents()
            time.sleep(2)  # hold each sub-band for 2 seconds
            self.stop()

if __name__ == '__main__':
    qapp = Qt.QApplication(sys.argv)
    tb = pluto_wifi_sweep()
    tb.show()
    tb.sweep()
    qapp.exec_()
