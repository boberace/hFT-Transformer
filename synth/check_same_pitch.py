#! python
# Same pitch on different strings: render one MIDI pitch on every string that reaches it
# (raw string model, nominal gauges, one polarization, no body/sensor), measure partials,
# fit B, compare spectral envelopes, and plot.
#
# python -m synth.check_same_pitch --pitch 64 --out corpus/Synth/check

import argparse
import itertools
import json
import os

import numpy as np
import scipy.signal

from synth import render, strings, tabs_procedural

FS = render.FS
STRING_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300')


def render_raw(guitar_type, s, fret, pitch, dur=2.0, velocity=90, pluck_m=0.12, seed=0):
    gs = strings.GAUGES[guitar_type]
    tuning = tabs_procedural.TUNING_STANDARD
    note = strings.NoteParams(midi=pitch, fret=fret, velocity=velocity, dur=dur, pluck_m=pluck_m, release=0.05)
    y = strings.render_note(gs.strings[s], gs, tuning[s], note, np.random.default_rng(seed), two_pol=False)
    y = scipy.signal.resample_poly(y, 1, strings.FS_INTERNAL // FS)
    return y / (np.max(np.abs(y)) + 1e-12)


def spectrum(y, t0=0.05, t1=1.05, n_fft=1 << 17):
    seg = y[int(t0 * FS):int(t1 * FS)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    return f, 20.0 * np.log10(X / X.max() + 1e-10)


def find_partials(f, db, f1, f_max=7500.0, floor_db=-80.0):
    # walk up the partials: each search is centred on the previous partial + the local spacing.
    # Partials below floor_db (e.g. in a pluck-position notch) are skipped, not tracked.
    out = []
    prev, spacing, n = 0.0, f1, 1
    while True:
        guess = prev + spacing
        if guess > f_max:
            break
        m = (f > guess - 0.25 * f1) & (f < guess + 0.25 * f1)
        i = np.flatnonzero(m)[np.argmax(db[m])]
        if db[i] < floor_db:
            prev, n = guess, n + 1
            continue
        # parabolic interpolation of the peak
        a, b, c = db[i - 1], db[i], db[i + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c)
        fp = f[i] + d * (f[1] - f[0])
        out.append((n, fp, b - 0.25 * (a - c) * d))
        spacing = (fp - out[-2][1]) / (n - out[-2][0]) if len(out) > 1 else fp
        prev, n = fp, n + 1
    return np.array(out)


def fit_B(partials):
    # (f_n / n)^2 = f0^2 (1 + B n^2): linear in n^2
    n, fn = partials[:, 0], partials[:, 1]
    slope, icpt = np.polyfit(n ** 2, (fn / n) ** 2, 1)
    return slope / icpt, np.sqrt(icpt)


def mel_db(y, n_mels=256, n_fft=2048, hop=256, t1=0.5):
    # mean log-mel over the first t1 s (features downstream: 16 kHz, 256 mel, hop 256)
    seg = y[:int(t1 * FS)]
    _, _, Z = scipy.signal.stft(seg, FS, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    P = np.abs(Z) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    mel = lambda x: 2595.0 * np.log10(1.0 + x / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(mel(0.0), mel(FS / 2), n_mels + 2) / 2595.0) - 1.0)
    fb = np.zeros((n_mels, len(f)))
    for k in range(n_mels):
        lo, c, hi = edges[k:k + 3]
        fb[k] = np.clip(np.minimum((f - lo) / (c - lo + 1e-9), (hi - f) / (hi - c + 1e-9)), 0.0, None)
    M = 10.0 * np.log10(fb @ P + 1e-12).mean(axis=1)
    return M - M.max()


def analyse(guitar_type, pitch):
    gs = strings.GAUGES[guitar_type]
    tuning = tabs_procedural.TUNING_STANDARD
    rows = []
    for s in range(6):
        fret = pitch - tuning[s]
        if not 0 <= fret <= tabs_procedural.MAX_FRET:
            continue
        y = render_raw(guitar_type, s, fret, pitch)
        f, db = spectrum(y)
        f1 = float(strings.midi2hz(pitch))
        P = find_partials(f, db, f1)
        B_fit, _ = fit_B(P)
        B_model = strings.inharmonicity(gs.strings[s], gs.scale_m, tuning[s], fret)
        amp = P[:, 2] - P[0, 2]
        lin = 10.0 ** (P[:, 2] / 20.0)
        rows.append({'string': s, 'fret': fret, 'B_model': B_model, 'B_fit': B_fit,
                     'partials': P, 'amp_db': amp, 'centroid_hz': float(np.sum(P[:, 1] * lin ** 2) / np.sum(lin ** 2)),
                     'f': f, 'db': db, 'mel': mel_db(y)})
    return rows


def summarize(guitar_type, rows, n_env=12):
    out = {'guitar': guitar_type, 'strings': [], 'pairs': []}
    for r in rows:
        P = r['partials']
        dev = 1200.0 * np.log2(P[:, 1] / (P[:, 0] * P[0, 1]))
        k10 = min(10, len(P)) - 1
        out['strings'].append({
            'string': r['string'], 'fret': r['fret'], 'B_model': r['B_model'], 'B_fit': float(r['B_fit']),
            'n_partials': int(len(P)), 'stretch_cents_n%d' % (k10 + 1): float(dev[k10]),
            'centroid_hz': r['centroid_hz']})
    for a, b in itertools.combinations(rows, 2):
        n = min(n_env, len(a['amp_db']), len(b['amp_db']))
        env = float(np.sqrt(np.mean((a['amp_db'][:n] - b['amp_db'][:n]) ** 2)))
        mel = float(np.sqrt(np.mean((a['mel'] - b['mel']) ** 2)))
        out['pairs'].append({'strings': [a['string'], b['string']], 'env_rms_db': env, 'mel_rms_db': mel})
    return out


def plot(results, pitch, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#999999', 'axes.labelcolor': '#333333',
                         'xtick.color': '#555555', 'ytick.color': '#555555'})
    types = list(results)
    fig, axes = plt.subplots(3, len(types), figsize=(5.2 * len(types), 11), squeeze=False)
    for j, g in enumerate(types):
        rows = results[g]
        ax = axes[0, j]
        for k, r in enumerate(rows):
            m = r['f'] <= 8000
            ax.plot(r['f'][m], r['db'][m] - 60.0 * k, color=STRING_COLORS[r['string']], lw=0.8,
                    label='string %d, fret %d' % (r['string'], r['fret']))
        ax.set_title('%s: MIDI %d spectrum (offset 60 dB per string)' % (g, pitch), fontsize=10)
        ax.set_xlabel('frequency (Hz)')
        ax.set_yticks([])
        ax.set_xlim(0, 8000)
        ax.legend(fontsize=7, frameon=False, loc='upper right')

        ax = axes[1, j]
        for r in rows:
            P = r['partials']
            n = P[:, 0]
            dev = 1200.0 * np.log2(P[:, 1] / (n * P[0, 1]))
            c = STRING_COLORS[r['string']]
            ax.plot(n, dev, 'o', ms=4, color=c, mec='white', mew=0.6,
                    label='s%d f%d  B fit %.1e / model %.1e' % (r['string'], r['fret'], r['B_fit'], r['B_model']))
            nn = np.linspace(1, n[-1], 100)
            Bm = r['B_model']
            ax.plot(nn, 1200.0 * np.log2(np.sqrt((1 + Bm * nn ** 2) / (1 + Bm))), color=c, lw=1.0, alpha=0.6)
        ax.set_title('partial deviation from n f1 (dots measured, lines model)', fontsize=10)
        ax.set_xlabel('partial n')
        ax.set_ylabel('cents')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)

        ax = axes[2, j]
        for r in rows:
            P = r['partials']
            ax.plot(P[:, 0], r['amp_db'], '-o', ms=3, lw=1.2, color=STRING_COLORS[r['string']],
                    mec='white', mew=0.5, label='s%d f%d' % (r['string'], r['fret']))
        ax.set_title('partial amplitudes (dB re partial 1)', fontsize=10)
        ax.set_xlabel('partial n')
        ax.set_ylabel('dB')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pitch', type=int, default=64)
    ap.add_argument('--types', default=','.join(render.GUITAR_TYPES))
    ap.add_argument('--out', default='corpus/Synth/check')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results, summary = {}, []
    for g in args.types.split(','):
        results[g] = analyse(g, args.pitch)
        summary.append(summarize(g, results[g]))
    png = os.path.join(args.out, 'same_pitch_%d.png' % args.pitch)
    plot(results, args.pitch, png)
    with open(os.path.join(args.out, 'same_pitch_%d.json' % args.pitch), 'w') as f:
        json.dump(summary, f, indent=1)
    for sm in summary:
        print('== %s, MIDI %d' % (sm['guitar'], args.pitch))
        for r in sm['strings']:
            k = [x for x in r if x.startswith('stretch')][0]
            print('  string %d fret %2d  B model %.2e fit %.2e  partials %2d  %s %6.1f c  centroid %5.0f Hz'
                  % (r['string'], r['fret'], r['B_model'], r['B_fit'], r['n_partials'], k, r[k], r['centroid_hz']))
        env = np.array([p['env_rms_db'] for p in sm['pairs']])
        mel = np.array([p['mel_rms_db'] for p in sm['pairs']])
        print('  pairwise partial-envelope RMS diff (first 12): mean %.1f dB, range %.1f-%.1f dB' % (env.mean(), env.min(), env.max()))
        print('  pairwise log-mel RMS diff: mean %.1f dB, range %.1f-%.1f dB' % (mel.mean(), mel.min(), mel.max()))
    print('plot:', png)


if __name__ == '__main__':
    main()
