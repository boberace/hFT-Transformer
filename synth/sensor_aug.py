#! python
# Headstock contact-sensor domain augmentation (vibration sensor clipped to the headstock,
# e.g. Syntiant V2S200D): no air path, so no room sound is ever added.
#
#   per-string transfer function   string -> nut/fret -> neck -> headstock -> sensor; applied to
#                                  each string stem before mixing. Shared headstock response
#                                  (resonant peaks/notches + tilt) plus per-string deviations
#                                  (coupling gain, extra peaks, tilt).
#   band-limiting                  sensor/ADC high-pass and low-pass
#   sensor noise                   white + pink floor, low-frequency handling rumble, sparse
#                                  finger/hand thumps, optional soft clipping
#
# Everything is described by a magnitude response in dB on a frequency grid and turned into a
# minimum-phase FIR, so a measured response can replace the parametric one:
#   SensorModel(cfg, rng, measured=MeasuredResponse(freqs_hz, mag_db))  # mag_db [n_freq] or [6, n_freq]

from dataclasses import dataclass, field

import numpy as np
import scipy.signal

N_STRING = 6


@dataclass
class SensorConfig:
    fs: int = 16000
    n_fft: int = 4096                       # frequency grid of the responses
    n_taps: int = 1024                      # minimum-phase FIR length
    # shared headstock response
    n_peaks: tuple = (2, 7)
    peak_hz: tuple = (120.0, 6000.0)        # log-uniform
    peak_gain_db: tuple = (-10.0, 14.0)
    peak_q: tuple = (1.5, 12.0)
    tilt_db_oct: tuple = (-4.0, 3.0)        # around 1 kHz
    # per-string deviations
    string_gain_db: tuple = (-6.0, 6.0)
    string_n_peaks: tuple = (0, 3)
    string_peak_gain_db: tuple = (-6.0, 8.0)
    string_tilt_db_oct: tuple = (-1.5, 1.5)
    # band-limiting
    hp_hz: tuple = (40.0, 150.0)
    hp_order: tuple = (1, 3)
    lp_hz: tuple = (3500.0, 7600.0)
    lp_order: tuple = (2, 6)
    # noise (SNR against the mix RMS)
    snr_db: tuple = (20.0, 50.0)
    pink_frac: tuple = (0.0, 0.7)
    rumble_prob: float = 0.5
    rumble_db: tuple = (-35.0, -15.0)       # relative to the mix RMS
    thump_rate: tuple = (0.0, 0.5)          # per second
    thump_db: tuple = (-25.0, -5.0)
    clip_prob: float = 0.1
    clip_drive: tuple = (1.0, 3.0)


@dataclass
class MeasuredResponse:
    freqs_hz: np.ndarray
    mag_db: np.ndarray                      # [n_freq] (all strings) or [n_string, n_freq]
    jitter_db: float = 0.0                  # optional random smooth deviation added on top
    extra: dict = field(default_factory=dict)


def _peak_db(f, fc, gain_db, q):
    # magnitude (dB) of an RBJ peaking EQ, evaluated analytically on the grid
    w0 = 2.0 * np.pi * fc
    w = 2.0 * np.pi * np.maximum(f, 1e-3)
    A = 10.0 ** (gain_db / 40.0)
    # analog prototype H(s) = (s^2 + s w0 A/q + w0^2) / (s^2 + s w0/(A q) + w0^2)
    s = 1j * w
    h = (s ** 2 + s * w0 * A / q + w0 ** 2) / (s ** 2 + s * w0 / (A * q) + w0 ** 2)
    return 20.0 * np.log10(np.abs(h))


def _butter_db(f, fc, order, kind):
    # Butterworth magnitude, |H|^2 = 1 / (1 + (f/fc)^(2 order)) for low-pass
    r = np.maximum(f, 1e-3) / fc
    if kind == 'high':
        r = 1.0 / r
    return -10.0 * np.log10(1.0 + r ** (2 * order))


def min_phase_fir(mag_db, n_fft, n_taps):
    # minimum-phase FIR from a one-sided magnitude response (n_fft // 2 + 1 points), cepstral method
    mag = 10.0 ** (np.asarray(mag_db) / 20.0)
    full = np.concatenate([mag, mag[-2:0:-1]])
    cep = np.fft.ifft(np.log(np.maximum(full, 1e-6))).real
    fold = np.zeros(n_fft)
    fold[0] = cep[0]
    fold[1:n_fft // 2] = 2.0 * cep[1:n_fft // 2]
    fold[n_fft // 2] = cep[n_fft // 2]
    h = np.fft.ifft(np.exp(np.fft.fft(fold))).real[:n_taps]
    h *= np.hanning(2 * n_taps)[n_taps:]   # fade the tail
    return h


def _pink(n, rng):
    x = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(x))
    x[1:] /= np.sqrt(f[1:])
    x[0] = 0.0
    y = np.fft.irfft(x, n)
    return y / (np.std(y) + 1e-12)


class SensorModel:
    # one random sensor/mounting instance; reuse it for all clips that share a mounting
    def __init__(self, cfg=None, rng=None, measured=None):
        self.cfg = cfg or SensorConfig()
        self.rng = rng or np.random.default_rng()
        c = self.cfg
        self.freqs = np.fft.rfftfreq(c.n_fft, 1.0 / c.fs)
        if measured is None:
            self.string_db = self._parametric()
        else:
            self.string_db = self._from_measured(measured)
        # band-limit (applied after mixing)
        r = self.rng
        self.hp = (r.uniform(*c.hp_hz), int(r.integers(c.hp_order[0], c.hp_order[1] + 1)))
        self.lp = (r.uniform(*c.lp_hz), int(r.integers(c.lp_order[0], c.lp_order[1] + 1)))
        self.band_db = (_butter_db(self.freqs, self.hp[0], self.hp[1], 'high')
                        + _butter_db(self.freqs, self.lp[0], self.lp[1], 'low'))
        self.string_fir = [min_phase_fir(db, c.n_fft, c.n_taps) for db in self.string_db]
        self.band_fir = min_phase_fir(self.band_db, c.n_fft, c.n_taps)

    def _loguni(self, lo_hi):
        return float(np.exp(self.rng.uniform(np.log(lo_hi[0]), np.log(lo_hi[1]))))

    def _parametric(self):
        c, r, f = self.cfg, self.rng, self.freqs
        oct_1k = np.log2(np.maximum(f, 20.0) / 1000.0)
        shared = r.uniform(*c.tilt_db_oct) * oct_1k
        for _ in range(int(r.integers(c.n_peaks[0], c.n_peaks[1] + 1))):
            shared += _peak_db(f, self._loguni(c.peak_hz), r.uniform(*c.peak_gain_db), self._loguni(c.peak_q))
        out = []
        for _ in range(N_STRING):
            db = shared + r.uniform(*c.string_gain_db) + r.uniform(*c.string_tilt_db_oct) * oct_1k
            for _ in range(int(r.integers(c.string_n_peaks[0], c.string_n_peaks[1] + 1))):
                db = db + _peak_db(f, self._loguni(c.peak_hz), r.uniform(*c.string_peak_gain_db),
                                   self._loguni(c.peak_q))
            out.append(db)
        return np.array(out)

    def _from_measured(self, m):
        mag = np.atleast_2d(np.asarray(m.mag_db, dtype=np.float64))
        if mag.shape[0] == 1:
            mag = np.repeat(mag, N_STRING, axis=0)
        out = np.array([np.interp(self.freqs, m.freqs_hz, row) for row in mag])
        if m.jitter_db > 0.0:
            oct_1k = np.log2(np.maximum(self.freqs, 20.0) / 1000.0)
            out += np.array([self.rng.normal(0.0, m.jitter_db) * oct_1k / 3.0
                             + self.rng.normal(0.0, m.jitter_db) for _ in range(N_STRING)])
        return out

    def response_db(self):
        # (freqs, per-string dB including the band-limit) for plots
        return self.freqs, self.string_db + self.band_db

    def apply_stems(self, stems):
        # stems [n_string, n] -> filtered stems (same length)
        n = stems.shape[1]
        return np.stack([scipy.signal.oaconvolve(s, h)[:n] for s, h in zip(stems, self.string_fir)])

    def finalize(self, mix):
        # band-limit + sensor noise (+ optional soft clip); returns float64
        c, r = self.cfg, self.rng
        n = len(mix)
        y = scipy.signal.oaconvolve(mix, self.band_fir)[:n]
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        noise_rms = rms * 10.0 ** (-r.uniform(*c.snr_db) / 20.0)
        pf = r.uniform(*c.pink_frac)
        noise = (np.sqrt(1.0 - pf) * r.standard_normal(n) + np.sqrt(pf) * _pink(n, r)) * noise_rms
        if r.random() < c.rumble_prob:
            # handling rumble: low-passed noise below ~30 Hz, slowly modulated
            b, a = scipy.signal.butter(2, r.uniform(8.0, 30.0) / (c.fs / 2.0))
            rum = scipy.signal.lfilter(b, a, r.standard_normal(n))
            rum *= (1.0 + 0.8 * np.sin(2.0 * np.pi * r.uniform(0.05, 0.5) * np.arange(n) / c.fs))
            noise += rum / (np.std(rum) + 1e-12) * rms * 10.0 ** (r.uniform(*c.rumble_db) / 20.0)
        n_thump = r.poisson(r.uniform(*c.thump_rate) * n / c.fs)
        for _ in range(n_thump):
            # finger / hand contact on the neck: short damped low-mid burst
            L = int(c.fs * r.uniform(0.01, 0.06))
            t = np.arange(L) / c.fs
            burst = np.sin(2.0 * np.pi * r.uniform(60.0, 400.0) * t) * np.exp(-t / r.uniform(0.003, 0.015))
            i = int(r.integers(0, max(n - L, 1)))
            noise[i:i + L] += burst[:n - i] * rms * 10.0 ** (r.uniform(*c.thump_db) / 20.0) * 3.0
        y = y + noise
        if r.random() < c.clip_prob:
            d = r.uniform(*c.clip_drive)
            peak = np.max(np.abs(y)) + 1e-9
            y = np.tanh(d * y / peak) / np.tanh(d) * peak
        return y
