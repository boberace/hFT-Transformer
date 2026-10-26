#! python
# Per-string digital waveguide (extended Karplus-Strong) for guitar.
#
# loop:  y = x + z^-N * L(z) * D(z) * y
#   N     integer delay
#   L(z)  one-pole loss filter matched to T60 at the fundamental and at a high frequency
#   D(z)  cascade of 2nd-order allpasses fitted (least squares on phase) so that the loop
#         phase delay gives stiff-string partials f_n = n f0 sqrt(1 + B n^2); it also absorbs
#         the fractional part of the delay
# The whole loop is one rational transfer function and is run with scipy.signal.lfilter.
#
# Excitation: smoothed pick/finger pulse plus a little noise, brightness set by velocity,
# then a pluck-position comb (1 - z^-(beta P)). An optional pickup-position comb is applied to
# the output (electric). Both positions are absolute distances from the bridge, so the
# relative position beta = d / L_fret moves with the fret.
#
# Parameters come from GAUGES: string gauge, core diameter, tension and Young's modulus give
# B = pi^3 E d_core^4 / (64 T L^2) (Fletcher). Retuning scales T with f^2; fretting shortens L,
# so B_fret = B_open * 2^(2 fret / 12).

import functools
import math
from dataclasses import dataclass, replace

import numpy as np
import scipy.signal

FS_INTERNAL = 32000          # strings are rendered at 2x the feature rate, then decimated
INCH = 0.0254
LB = 4.4482216
E_STEEL = 2.0e11             # Pa
E_NYLON = 5.0e9              # Pa, plain nylon trebles
E_NYLON_CORE = 1.5e9         # Pa, effective for a multifilament bass core


def midi2hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


@dataclass(frozen=True)
class StringSpec:
    name: str           # note name at standard tuning
    open_midi: int      # pitch the tension refers to
    gauge_in: float     # outer diameter (inch)
    wound: bool
    core_in: float      # diameter carrying the bending stiffness (inch); = gauge for plain strings
    tension_lb: float   # tension at open_midi (approximate, manufacturer tension charts)
    youngs: float       # Pa, of the core
    b1: float           # loss sigma(f) = b1 + b3 f^2 [1/s], T60 = 6.91 / sigma
    b3: float


@dataclass(frozen=True)
class StringSet:
    name: str
    scale_m: float
    strings: tuple      # 6 StringSpec, index 0 = low E
    fret_b1: float      # extra b1 for a fretted (finger-stopped) note [1/s]
    fret_b1_slope: float  # relative b1 increase per fret
    pick_hardness: float  # 0 (soft flesh) .. 1 (hard pick): excitation brightness


# --- gauge table ------------------------------------------------------------------------
# steel acoustic: phosphor bronze 12-53 light, 25.5"
STEEL_ACOUSTIC = StringSet(
    name='steel_acoustic', scale_m=25.5 * INCH,
    strings=(
        StringSpec('E2', 40, 0.053, True,  0.0200, 26.0, E_STEEL, 0.70, 3.0e-6),
        StringSpec('A2', 45, 0.042, True,  0.0170, 29.0, E_STEEL, 0.75, 3.0e-6),
        StringSpec('D3', 50, 0.032, True,  0.0140, 30.4, E_STEEL, 0.80, 2.6e-6),
        StringSpec('G3', 55, 0.024, True,  0.0115, 29.6, E_STEEL, 0.85, 2.3e-6),
        StringSpec('B3', 59, 0.016, False, 0.0160, 24.6, E_STEEL, 0.95, 1.2e-6),
        StringSpec('E4', 64, 0.012, False, 0.0120, 23.3, E_STEEL, 1.00, 1.1e-6),
    ),
    fret_b1=0.35, fret_b1_slope=0.03, pick_hardness=0.8)

# electric: nickel-wound 10-46, 25.5"; solid body sustains longer
STEEL_ELECTRIC = StringSet(
    name='steel_electric', scale_m=25.5 * INCH,
    strings=(
        StringSpec('E2', 40, 0.046, True,  0.0185, 17.5, E_STEEL, 0.40, 1.8e-6),
        StringSpec('A2', 45, 0.036, True,  0.0160, 19.5, E_STEEL, 0.42, 1.8e-6),
        StringSpec('D3', 50, 0.026, True,  0.0135, 18.4, E_STEEL, 0.45, 1.6e-6),
        StringSpec('G3', 55, 0.017, False, 0.0170, 16.6, E_STEEL, 0.50, 0.8e-6),
        StringSpec('B3', 59, 0.013, False, 0.0130, 15.4, E_STEEL, 0.55, 0.8e-6),
        StringSpec('E4', 64, 0.010, False, 0.0100, 16.2, E_STEEL, 0.60, 0.7e-6),
    ),
    fret_b1=0.25, fret_b1_slope=0.03, pick_hardness=0.9)

# classical: nylon trebles, silver-plated copper on multifilament nylon basses, normal tension, 650 mm
NYLON = StringSet(
    name='nylon', scale_m=0.650,
    strings=(
        StringSpec('E2', 40, 0.043, True,  0.0250, 15.8, E_NYLON_CORE, 1.30, 7.0e-6),
        StringSpec('A2', 45, 0.035, True,  0.0200, 15.6, E_NYLON_CORE, 1.35, 7.0e-6),
        StringSpec('D3', 50, 0.029, True,  0.0170, 15.9, E_NYLON_CORE, 1.40, 6.5e-6),
        StringSpec('G3', 55, 0.0403, False, 0.0403, 12.6, E_NYLON, 1.60, 4.5e-6),
        StringSpec('B3', 59, 0.0322, False, 0.0322, 12.2, E_NYLON, 1.65, 4.0e-6),
        StringSpec('E4', 64, 0.0285, False, 0.0285, 15.7, E_NYLON, 1.70, 3.5e-6),
    ),
    fret_b1=0.45, fret_b1_slope=0.03, pick_hardness=0.35)

GAUGES = {'acoustic': STEEL_ACOUSTIC, 'electric': STEEL_ELECTRIC, 'nylon': NYLON}


def inharmonicity_open(spec, scale_m, open_midi=None):
    # B of the open string, optionally retuned to open_midi (same gauge, tension ~ f^2)
    d = spec.core_in * INCH
    tension = spec.tension_lb * LB
    if open_midi is not None:
        tension *= 2.0 ** ((open_midi - spec.open_midi) / 6.0)
    return math.pi ** 3 * spec.youngs * d ** 4 / (64.0 * tension * scale_m ** 2)


def inharmonicity(spec, scale_m, open_midi, fret):
    return inharmonicity_open(spec, scale_m, open_midi) * 2.0 ** (2.0 * fret / 12.0)


def jitter_set(string_set, rng, b_range=(0.75, 1.35), loss_sigma=0.2):
    # per-instrument variation: string age/brand (loss) and core/tension uncertainty (B)
    strings = []
    for sp in string_set.strings:
        kb = rng.uniform(*b_range)
        strings.append(replace(sp,
                               core_in=sp.core_in * kb ** 0.25,
                               b1=sp.b1 * math.exp(rng.normal(0.0, loss_sigma)),
                               b3=sp.b3 * math.exp(rng.normal(0.0, loss_sigma))))
    return replace(string_set, strings=tuple(strings))


# --- loop filter design -----------------------------------------------------------------
def stiff_partial_index(f, f0, B):
    # continuous partial index n at frequency f for f_n = n f0 sqrt(1 + B n^2)
    r = np.asarray(f, dtype=np.float64) / f0
    if B < 1e-12:
        return r
    return np.sqrt((np.sqrt(1.0 + 4.0 * B * r ** 2) - 1.0) / (2.0 * B))


def loss_filter(f0, b1, b3, fs, f_hf=None, p_step=0.02):
    # one-pole H(z) = g (1 - p) / (1 - p z^-1), |H| per period matched to T60 at f0 and f_hf
    if f_hf is None:
        f_hf = min(4000.0, 0.4 * fs)
    f_hf = max(f_hf, 2.0 * f0)
    period = fs / f0
    def per_period_gain(f):
        t60 = 6.91 / (b1 + b3 * f ** 2)
        return 10.0 ** (-3.0 * period / (fs * t60))
    g = per_period_gain(f0)
    ghf = per_period_gain(f_hf)
    # |H(w)|^2 = g^2 (1-p)^2 / (1 - 2p cos w + p^2) = r^2, r = ghf / g  ->  quadratic in p
    r2 = min((ghf / g) ** 2, 0.999999)
    c = math.cos(2.0 * math.pi * f_hf / fs)
    # (1 - r2) p^2 - 2 (1 - r2 c) p + (1 - r2) = 0
    qa, qb = 1.0 - r2, -2.0 * (1.0 - r2 * c)
    p = (-qb - math.sqrt(max(qb * qb - 4.0 * qa * qa, 0.0))) / (2.0 * qa)
    p = min(max(p, 0.0), 0.95)
    if p_step:
        p = round(p / p_step) * p_step   # so that dispersion designs can be cached per p
    # the gain at f0 is slightly below g because of the low-pass; correct it
    w0 = 2.0 * math.pi * f0 / fs
    mag0 = (1.0 - p) / math.sqrt(1.0 - 2.0 * p * math.cos(w0) + p * p)
    g = min(g / mag0, 0.99999)
    return np.array([g * (1.0 - p)]), np.array([1.0, -p])


def _phase_delay(b, a, w):
    _, h = scipy.signal.freqz(b, a, worN=w)
    return -np.unwrap(np.angle(h)) / w


def _allpass2_ls(d_target, w, weight, n_iter=5):
    # 2nd-order allpass H = (a2 + a1 z^-1 + z^-2) / (1 + a1 z^-1 + a2 z^-2) with phase delay ~ d_target
    # (equation-error least squares, Lang & Laakso 1994, reweighted by 1/|A| so that it converges
    # to the true phase-error solution)
    M = 2
    theta = -w * d_target
    beta = -(theta + M * w) / 2.0
    k = np.arange(1, M + 1)
    basis = np.sin(np.outer(w, k) + beta[:, None])
    rhs = -np.sin(beta)
    wt = weight.copy()
    for _ in range(n_iter):
        a = np.linalg.lstsq(basis * wt[:, None], rhs * wt, rcond=None)[0]
        den = np.concatenate([[1.0], a])
        if not np.all(np.abs(np.roots(den)) < 0.995):
            break
        _, h = scipy.signal.freqz(den, [1.0], worN=w)
        wt = weight / np.maximum(np.abs(h), 1e-3)
    return den[::-1].copy(), den


def _fit_sections(p_dis, w, weight, n_sec, greedy, n_min, n_shift=0):
    # n_sec 2nd-order allpass sections whose summed phase delay ~ p_dis - N; None if unstable.
    # greedy: each section fits the residual left by the previous ones.
    # An allpass section has exactly 2 samples of phase delay at Nyquist; n_shift moves the
    # integer delay so that the fractional mismatch there can land where the weight is low.
    N = int(math.floor(p_dis[-1] - 2.0 * n_sec)) - n_shift
    if N < n_min:
        return None
    rem = p_dis - N
    secs = []
    for k in range(n_sec):
        if greedy or k == 0:
            b, a = _allpass2_ls(rem / (n_sec - k), w, weight)
            if not np.all(np.abs(np.roots(a)) < 0.995):
                return None
            rem = rem - _phase_delay(b, a, w)
        secs.append((b, a))
    return N, secs


F0_STEP_CENTS = 25.0     # dispersion designs are cached on an f0/B grid; the Thiran stage tunes exactly
B_STEP = 0.05            # relative


def quantize_B(B):
    return float(math.exp(round(math.log(max(B, 1e-9)) / B_STEP) * B_STEP))


@functools.lru_cache(maxsize=20000)
def _dispersion_cached(f0q, Bq, p_loss, fs, n_fit):
    # cascade of 2nd-order allpasses with phase delay ~ (fs/f0)/sqrt(1 + B n(w)^2) - N - 1 - loss
    # (the 1 is the nominal delay of the Thiran tuning stage, loss the phase delay of the one-pole
    # loss filter with pole p_loss). Fitted with the stiff-string formula; partials up to n_fit
    # (or 8 kHz) dominate, the rest of the band up to 8 kHz weighs 0.05, above 8 kHz almost nothing.
    f0, B = f0q, Bq
    f_band = min(8000.0, 0.45 * fs)          # everything above is removed by the decimation
    f_hi = min(f_band, n_fit * f0 * math.sqrt(1.0 + B * n_fit ** 2))
    w_hi = 2.0 * np.pi * f_hi / fs
    w_band = 2.0 * np.pi * f_band / fs
    w = np.concatenate([np.linspace(2.0 * np.pi * f0 * 0.5 / fs, w_hi, 256),
                        np.linspace(w_hi, w_band, 33)[1:],
                        np.linspace(w_band, np.pi * 0.98, 33)[1:]])
    n_idx = stiff_partial_index(w * fs / (2.0 * np.pi), f0, B)
    p_dis = (fs / f0) / np.sqrt(1.0 + B * n_idx ** 2) - 1.0 - _phase_delay([1.0 - p_loss], [1.0, -p_loss], w)
    spread = p_dis[0] - p_dis[-1]
    in_band = w <= w_hi
    weight = np.where(in_band, 1.0 / (1.0 + n_idx / 10.0), np.where(w <= w_band, 0.05, 0.002))
    # headroom in the integer delay for the f0 grid offset and the loss filter's phase delay
    n_min = 6 + int(math.ceil(p_dis[0] * (2.0 ** (F0_STEP_CENTS / 2400.0) - 1.0)))
    # section counts: ~per samples of the spread each, plus the most the integer delay allows
    k_max = int((p_dis[-1] + 1.0 - n_min) // 2)
    counts = [max(1, int(math.ceil(spread / per))) for per in (2.0, 1.5, 1.0, 0.7, 0.5)]
    counts = sorted({k for k in counts if k <= k_max} | {k for k in (k_max, k_max - 1) if k >= 1})
    best, best_err = None, np.inf
    for greedy in (True, False):
        for n_sec in counts:
            for n_shift in (-1, 0, 1):
                cand = _fit_sections(p_dis, w, weight, n_sec, greedy, n_min, n_shift)
                if cand is None:
                    continue
                N, secs = cand
                d = N + sum(_phase_delay(b, a, w) for b, a in secs)
                err = np.abs(1200.0 * np.log2(d / p_dis))[in_band].max()
                if err < best_err:
                    best, best_err = cand, err
            if best_err < 2.0:
                break
        if best_err < 2.0:
            break
    if best is None:
        raise ValueError('no stable dispersion filter: f0=%.1f B=%.2e' % (f0, B))
    N, secs = best
    return N, tuple((tuple(b), tuple(a)) for b, a in secs)


def _sos_row(b, a):
    b = list(b) + [0.0] * (3 - len(b))
    a = list(a) + [0.0] * (3 - len(a))
    return b + a


def _sos_phase(sos, w):
    # unwrapped phase of a cascade, summed per section (a product polynomial would lose precision)
    ph = np.zeros(len(w))
    for row in sos:
        _, h = scipy.signal.freqz(row[:3], row[3:], worN=w)
        ph += np.unwrap(np.angle(h))
    return ph


@dataclass
class LoopDesign:
    N: int              # integer delay
    sos: np.ndarray     # loop filter G(z) as 2nd-order sections: loss, Thiran tuning, dispersion
    B: float            # inharmonicity actually modelled (quantized)


def design_loop(f0, B, b1, b3, fs=FS_INTERNAL, n_fit=40):
    # string loop y = x + z^-N G(z) y.  f0 is the stiff-string reference frequency
    # (partial 1 at f0 sqrt(1 + B)).
    # f0 grid: 25 cents, but at most 0.1 sample of period for short loops, where the Thiran
    # stage would otherwise have to move far from its nominal 1-sample delay
    period = fs / f0
    if period < 80.0:
        f0q = fs / (round(period / 0.1) * 0.1)
    else:
        f0q = 440.0 * 2.0 ** (round(1200.0 * math.log2(f0 / 440.0) / F0_STEP_CENTS) * F0_STEP_CENTS / 1200.0)
    lb, la = loss_filter(f0, b1, b3, fs)
    while True:
        Bq = quantize_B(B)
        try:
            _, secs = _dispersion_cached(round(f0q, 6), Bq, round(-la[1], 6), fs, n_fit)
            break
        except ValueError:
            if B < 1e-7:
                raise
            B *= 0.8    # last resort: model a slightly less stiff string
    rows = [_sos_row(lb, la)] + [_sos_row(b, a) for b, a in secs]
    # tuning: integer delay + first-order Thiran so that the loop phase delay at partial 1 is
    # exactly fs / f1 (this absorbs the f0 grid offset, the loss filter's phase delay and fit error)
    f1 = f0 * math.sqrt(1.0 + Bq)
    w_lo = np.linspace(1e-4, 2.0 * np.pi * f1 / fs, 32)
    pd1 = -_sos_phase(rows, w_lo)[-1] / w_lo[-1]
    rest = fs / f1 - pd1
    N = int(math.floor(rest - 0.5))
    if N < 1:
        raise ValueError('pitch too high for fs: f0=%.1f' % f0)
    delta = rest - N
    t = (1.0 - delta) / (1.0 + delta)
    rows.insert(1, _sos_row([t, 1.0], [1.0, t]))
    return LoopDesign(N=N, sos=np.array(rows), B=Bq)


def run_loop(x, design):
    # y[n] = x[n] + v[n - N], v = G(y); processed in blocks of N samples so that the feedback
    # term of a block only needs v from earlier blocks
    N, sos = design.N, design.sos
    L = len(x)
    y = np.zeros(L)
    v = np.zeros(L)
    zi = np.zeros((len(sos), 2))
    for start in range(0, L, N):
        end = min(start + N, L)
        blk = x[start:end].copy()
        if start >= N:
            blk += v[start - N:end - N]
        y[start:end] = blk
        v[start:end], zi = scipy.signal.sosfilt(sos, blk, zi=zi)
    return y


def loop_partials(design, fs=FS_INTERNAL, f_max=8000.0):
    # partial frequencies actually produced by a designed loop (where loop phase = 2 pi n)
    f = np.linspace(1.0, f_max, 80000)
    w = 2.0 * np.pi * f / fs
    k = -(_sos_phase(design.sos, w) - w * design.N) / (2.0 * np.pi)
    k -= round(k[0])
    n = np.arange(1, int(k[-1]) + 1)
    return np.interp(n, k, f)


# --- note rendering ---------------------------------------------------------------------
def _frac_delay_comb(x, delay):
    # x - x delayed by a fractional number of samples (linear interpolation)
    if delay <= 0.0:
        return x
    n = np.arange(len(x), dtype=np.float64)
    return x - np.interp(n - delay, n, x, left=0.0, right=0.0)


def _rel_position(dist_m, l_fret):
    # relative position on the vibrating segment; the comb only depends on min(beta, 1 - beta)
    # (a pick near the fret on a high note acts like one near the bridge)
    beta = min(max(dist_m / l_fret, 0.02), 0.98)
    return min(beta, 1.0 - beta)


def _excitation(n_len, velocity, hardness, rng, fs):
    # smoothed pulse (pick/finger) + some noise; harder and louder = brighter
    v = velocity / 127.0
    width = min(max(2, int(round(fs * (0.0006 - 0.00045 * hardness * v ** 0.5)))), n_len)
    pulse = np.zeros(n_len)
    win = np.hanning(width + 2)[1:-1]
    pulse[:width] = win / win.sum()
    noise = rng.standard_normal(n_len) * np.hanning(n_len) * (0.15 / n_len)
    x = pulse + noise * (0.3 + 0.7 * hardness)
    fc = 1500.0 + 7000.0 * hardness * v + 1500.0 * v
    p = math.exp(-2.0 * math.pi * min(fc, 0.45 * fs) / fs)
    x = scipy.signal.lfilter([1.0 - p], [1.0, -p], x)
    return x * (0.2 + 0.8 * v ** 1.2)


@dataclass
class NoteParams:
    midi: float           # sounding pitch (may include detune)
    fret: int
    velocity: int
    dur: float            # seconds until the string is damped
    pluck_m: float        # pluck distance from bridge (m)
    pickup_m: tuple = ()  # pickup distances from bridge (m), electric only
    release: float = 0.04  # damping time constant after dur (s)


def render_note(spec, string_set, open_midi, note, rng, fs=FS_INTERNAL, two_pol=True):
    # returns float64 signal starting at the onset
    f_target = float(midi2hz(note.midi))
    B = inharmonicity(spec, string_set.scale_m, open_midi, note.fret)
    f0 = f_target / math.sqrt(1.0 + B)   # tuned so that partial 1 lands on the target pitch
    b1 = spec.b1
    if note.fret > 0:
        b1 = (b1 + string_set.fret_b1) * (1.0 + string_set.fret_b1_slope * note.fret)
    l_fret = string_set.scale_m * 2.0 ** (-note.fret / 12.0)
    period = fs / f0

    n_on = int(round(note.dur * fs))
    n_tail = int(round(note.release * fs * 7.0))   # ~60 dB below after the damping starts
    n_total = n_on + n_tail
    n_exc = max(int(round(period)), 8)
    x = np.zeros(n_total)
    x[:min(n_exc, n_total)] = _excitation(n_exc, note.velocity, string_set.pick_hardness, rng, fs)[:n_total]
    x = _frac_delay_comb(x, _rel_position(note.pluck_m, l_fret) * period)

    y = run_loop(x, design_loop(f0, B, b1, spec.b3, fs))
    if two_pol:
        # second polarization: slightly detuned, different decay, weaker
        f0b = f0 * 2.0 ** (rng.uniform(0.3, 1.5) / 1200.0)
        design = design_loop(f0b, B, b1 * rng.uniform(0.5, 0.9), spec.b3 * rng.uniform(0.8, 1.2), fs)
        y = y + rng.uniform(0.2, 0.5) * run_loop(x, design)

    if note.pickup_m:
        y = sum(_frac_delay_comb(y, _rel_position(d, l_fret) * period) for d in note.pickup_m)

    env = np.ones(n_total)
    t = np.arange(n_tail) / fs
    env[n_on:] = np.exp(-t / note.release)
    return y * env
