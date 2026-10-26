#! python
# Render labeled synthetic guitar audio.
#
# per clip: random guitar instance (gauge table + jitter), tuning, tab (tabs_procedural)
#   -> per-string stems from the waveguide model at 32 kHz (electric: pickup-position comb per note)
#   -> decimate to 16 kHz
#   -> sensor per-string transfer function (sensor_aug, optional)
#   -> mix; acoustic/nylon: body impulse response (synthetic or from --body_ir_dir);
#      electric: pickup resonance low-pass
#   -> sensor band-limit + noise (optional), peak normalize, 16-bit wav
# label json: {"guitar", "tuning", "sr", "notes": [{onset, offset, pitch, velocity, string, fret}]}
#
# python -m synth.render --out corpus/Synth --hours 10 --workers 8

import argparse
import glob
import json
import math
import multiprocessing
import os
import time

import numpy as np
import scipy.io.wavfile
import scipy.signal

from synth import sensor_aug, strings, tabs_procedural

FS = 16000
GUITAR_TYPES = ('acoustic', 'electric', 'nylon')

# electric pickup distances from the bridge (m), 25.5" scale
PICKUPS = {'bridge': 0.040, 'middle': 0.100, 'neck': 0.160}
PICKUP_SELECTIONS = (('bridge',), ('middle',), ('neck',), ('neck', 'middle'), ('middle', 'bridge'),
                     ('neck', 'bridge'))
PLUCK_M = {'acoustic': (0.08, 0.17), 'electric': (0.05, 0.15), 'nylon': (0.08, 0.18)}


# --- body / pickup ----------------------------------------------------------------------
def synth_body_ir(rng, guitar_type, fs=FS, length=0.5):
    # modal body IR: Helmholtz air mode, main top-plate modes, then a dense set of higher modes
    n = int(length * fs)
    t = np.arange(n) / fs
    if guitar_type == 'nylon':
        modes = [(rng.uniform(85, 105), rng.uniform(10, 25), 1.0),
                 (rng.uniform(170, 210), rng.uniform(15, 35), 0.9),
                 (rng.uniform(260, 330), rng.uniform(15, 35), 0.5)]
        n_dense, f_top, slope = 40, 5000.0, 0.9
    else:
        modes = [(rng.uniform(90, 115), rng.uniform(10, 25), 1.0),
                 (rng.uniform(180, 230), rng.uniform(15, 40), 0.9),
                 (rng.uniform(380, 480), rng.uniform(20, 40), 0.6)]
        n_dense, f_top, slope = 60, 7000.0, 0.6
    for _ in range(n_dense):
        f = math.exp(rng.uniform(math.log(250.0), math.log(f_top)))
        modes.append((f, rng.uniform(10, 60), (f / 300.0) ** -slope * math.exp(rng.normal(0.0, 0.5))))
    ir = np.zeros(n)
    for f, q, a in modes:
        if f >= 0.45 * fs:
            continue
        ir += a * np.exp(-np.pi * f * t / q) * np.sin(2.0 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    ir[0] += rng.uniform(0.1, 0.4)    # direct (bridge) path
    return ir


def load_body_irs(d):
    irs = []
    for f in sorted(glob.glob(os.path.join(d, '*.wav'))):
        sr, x = scipy.io.wavfile.read(f)
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != FS:
            g = math.gcd(sr, FS)
            x = scipy.signal.resample_poly(x, FS // g, sr // g)
        irs.append(x / (np.max(np.abs(x)) + 1e-12))
    return irs


def pickup_lowpass(rng, fs=FS):
    # pickup inductance + cable capacitance: resonant 2nd-order low-pass
    fc = rng.uniform(2500.0, 5500.0)
    q = rng.uniform(0.7, 3.0)
    w0 = 2.0 * np.pi * fc
    b, a = scipy.signal.bilinear([w0 ** 2], [1.0, w0 / q, w0 ** 2], fs)
    return b, a


# --- one clip -----------------------------------------------------------------------------
def render_stems(events, tuning, string_set, guitar_type, rng, n_out, pickups=()):
    # per-string stems at FS (decimated from the internal rate)
    fs_i = strings.FS_INTERNAL
    n_i = n_out * (fs_i // FS)
    stems = np.zeros((tabs_procedural.N_STRING, n_i))
    detune_global = rng.uniform(-15.0, 15.0)                   # cents, reference pitch
    detune_string = rng.uniform(-3.0, 3.0, size=tabs_procedural.N_STRING)
    pluck_base = rng.uniform(*PLUCK_M[guitar_type])
    by_string = {}
    for e in events:
        by_string.setdefault(e.string, []).append(e)
    for s, es in by_string.items():
        es.sort(key=lambda e: e.onset)
        for i, e in enumerate(es):
            cut = i + 1 < len(es) and es[i + 1].onset - e.offset < 0.05
            note = strings.NoteParams(
                midi=tuning[s] + e.fret + (detune_global + detune_string[s]) / 100.0,
                fret=e.fret, velocity=e.velocity, dur=e.duration,
                pluck_m=max(0.02, pluck_base + rng.normal(0.0, 0.01)),
                pickup_m=tuple(PICKUPS[p] for p in pickups),
                release=rng.uniform(0.005, 0.02) if cut else rng.uniform(0.03, 0.08))
            y = strings.render_note(string_set.strings[s], string_set, tuning[s], note, rng)
            i0 = int(round(e.onset * fs_i))
            if i0 >= n_i:
                continue
            m = min(len(y), n_i - i0)
            stems[s, i0:i0 + m] += y[:m]
    return scipy.signal.resample_poly(stems, 1, fs_i // FS, axis=1)


def render_clip(seed, guitar_type, duration, sensor=True, body_irs=None, tuning=None, events=None):
    # returns (audio float32 [n] at FS, label dict); seed: int or SeedSequence entropy
    rng = np.random.default_rng(seed)
    n_out = int(round(duration * FS))
    if events is None:
        events, tuning = tabs_procedural.generate_sequence(rng, duration, tuning)
    string_set = strings.jitter_set(strings.GAUGES[guitar_type], rng)
    pickups = PICKUP_SELECTIONS[rng.integers(len(PICKUP_SELECTIONS))] if guitar_type == 'electric' else ()
    stems = render_stems(events, tuning, string_set, guitar_type, rng, n_out, pickups)

    model = sensor_aug.SensorModel(rng=rng) if sensor else None
    if model is not None:
        stems = model.apply_stems(stems)
    mix = stems.sum(axis=0)
    if guitar_type == 'electric':
        b, a = pickup_lowpass(rng)
        mix = scipy.signal.lfilter(b, a, mix)
    else:
        if body_irs:
            ir = body_irs[rng.integers(len(body_irs))]
        else:
            ir = synth_body_ir(rng, guitar_type)
        mix = scipy.signal.oaconvolve(mix, ir)[:n_out]
    if model is not None:
        mix = model.finalize(mix)
    mix = mix * (10.0 ** (-rng.uniform(1.0, 6.0) / 20.0) / (np.max(np.abs(mix)) + 1e-9))

    notes = [{'onset': round(e.onset, 4), 'offset': round(e.offset, 4),
              'pitch': int(tuning[e.string] + e.fret), 'velocity': int(e.velocity),
              'string': int(e.string), 'fret': int(e.fret)} for e in events]
    notes.sort(key=lambda d: (d['onset'], d['pitch']))
    label = {'guitar': guitar_type, 'tuning': [int(t) for t in tuning], 'sr': FS,
             'sensor_aug': bool(sensor),
             'pickups': list(pickups), 'notes': notes}
    return mix.astype(np.float32), label


def write_clip(d_out, name, audio, label):
    os.makedirs(os.path.join(d_out, 'wav'), exist_ok=True)
    os.makedirs(os.path.join(d_out, 'note'), exist_ok=True)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(os.path.join(d_out, 'wav', name + '.wav'), FS, pcm)
    with open(os.path.join(d_out, 'note', name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(label, f, ensure_ascii=False, indent=1)


_BODY_IRS = None


def _worker_init(body_ir_dir):
    global _BODY_IRS
    _BODY_IRS = load_body_irs(body_ir_dir) if body_ir_dir else None


def _job(args):
    idx, seed, guitar_type, clip_sec, sensor, d_out, prefix = args
    name = '%s_%s_%06d' % (prefix, guitar_type, idx)
    audio, label = render_clip([seed, idx], guitar_type, clip_sec, sensor, _BODY_IRS)
    write_clip(d_out, name, audio, label)
    return name, len(label['notes'])


def main():
    ap = argparse.ArgumentParser(description='render synthetic labeled guitar audio')
    ap.add_argument('--out', default='corpus/Synth', help='output dir (gets wav/ and note/)')
    ap.add_argument('--hours', type=float, default=1.0, help='total audio to render')
    ap.add_argument('--clip_sec', type=float, default=30.0)
    ap.add_argument('--types', default=','.join(GUITAR_TYPES), help='comma separated: ' + ','.join(GUITAR_TYPES))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start', type=int, default=0, help='first clip index (to extend a corpus)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no_sensor', action='store_true', help='skip the contact-sensor augmentation')
    ap.add_argument('--body_ir_dir', default=None, help='dir of body IR wavs (default: synthetic IRs)')
    ap.add_argument('--prefix', default='synth')
    args = ap.parse_args()

    types = [t for t in args.types.split(',') if t]
    for t in types:
        if t not in GUITAR_TYPES:
            raise SystemExit('unknown guitar type: ' + t)
    clip_sec = min(args.clip_sec, args.hours * 3600.0)
    n_clip = max(1, int(math.ceil(args.hours * 3600.0 / clip_sec - 1e-3)))
    jobs = [(args.start + i, args.seed, types[(args.start + i) % len(types)], clip_sec,
             not args.no_sensor, args.out, args.prefix) for i in range(n_clip)]
    print('rendering %d clips x %.1f s (%.2f h) -> %s' % (n_clip, clip_sec, n_clip * clip_sec / 3600.0, args.out))
    t0 = time.time()
    if args.workers <= 1:
        _worker_init(args.body_ir_dir)
        it = map(_job, jobs)
    else:
        pool = multiprocessing.Pool(args.workers, initializer=_worker_init, initargs=(args.body_ir_dir,))
        it = pool.imap_unordered(_job, jobs)
    for k, (name, n_note) in enumerate(it, 1):
        el = time.time() - t0
        print('[%d/%d] %s (%d notes) %.0fs elapsed, %.0fs left' % (k, n_clip, name, n_note, el, el / k * (n_clip - k)),
              flush=True)


if __name__ == '__main__':
    main()
