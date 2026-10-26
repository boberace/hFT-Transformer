#! python
# Procedural guitar tablature: note events (onset, duration, string, fret, velocity) from
# scale runs, arpeggios, single-note lines and strummed/plucked chord voicings.
#
# Constraints (enforce_constraints, checked by check_constraints):
#   - one note per string at a time (a re-plucked string cuts the previous note)
#   - realistic fret-hand span: fretted notes sounding together span at most
#     max_stretch(lowest fret) frets and need at most 4 fingers (a barre counts as one)
#   - pitch within [pitch_min, pitch_max], fret within [0, max_fret]
# string 0 = low E. All randomness comes from the numpy Generator passed in.

from dataclasses import dataclass, asdict

import numpy as np

N_STRING = 6
TUNING_STANDARD = (40, 45, 50, 55, 59, 64)
PITCH_MIN, PITCH_MAX = 36, 88     # corpus/config_guitar.json note_min / note_max
MAX_FRET = 24

NAMED_TUNINGS = {
    'standard':    (40, 45, 50, 55, 59, 64),
    'half_down':   (39, 44, 49, 54, 58, 63),
    'full_down':   (38, 43, 48, 53, 57, 62),
    'drop_d':      (38, 45, 50, 55, 59, 64),
    'drop_c':      (36, 43, 48, 53, 57, 62),
    'dadgad':      (38, 45, 50, 55, 57, 62),
    'open_g':      (38, 43, 50, 55, 59, 62),
    'open_d':      (38, 45, 50, 54, 57, 62),
    'open_e':      (40, 47, 52, 56, 59, 64),
    'open_c':      (36, 43, 48, 55, 60, 64),
}

SCALES = {
    'major':      (0, 2, 4, 5, 7, 9, 11),
    'minor':      (0, 2, 3, 5, 7, 8, 10),
    'harm_minor': (0, 2, 3, 5, 7, 8, 11),
    'dorian':     (0, 2, 3, 5, 7, 9, 10),
    'mixolydian': (0, 2, 4, 5, 7, 9, 10),
    'phrygian':   (0, 1, 3, 5, 7, 8, 10),
    'pent_major': (0, 2, 4, 7, 9),
    'pent_minor': (0, 3, 5, 7, 10),
    'blues':      (0, 3, 5, 6, 7, 10),
    'chromatic':  tuple(range(12)),
}

CHORDS = {
    'maj': (0, 4, 7), 'min': (0, 3, 7), '7': (0, 4, 7, 10), 'maj7': (0, 4, 7, 11),
    'm7': (0, 3, 7, 10), 'sus2': (0, 2, 7), 'sus4': (0, 5, 7), 'dim': (0, 3, 6),
    'aug': (0, 4, 8), '5': (0, 7), 'add9': (0, 2, 4, 7), 'm7b5': (0, 3, 6, 10),
    '6': (0, 4, 7, 9), '9': (0, 2, 4, 7, 10),
}


@dataclass
class NoteEvent:
    onset: float
    duration: float
    string: int
    fret: int
    velocity: int

    @property
    def offset(self):
        return self.onset + self.duration


def max_stretch(lowest_fret):
    # widest fret distance a hand covers (frets get narrower up the neck)
    if lowest_fret <= 4:
        return 3
    if lowest_fret <= 11:
        return 4
    return 5


def sample_tuning(rng, p_standard=0.4, p_named=0.35):
    # standard, a named alternate tuning, or random per-string offsets (-4..+2 semitones)
    u = rng.random()
    if u < p_standard:
        return list(TUNING_STANDARD)
    if u < p_standard + p_named:
        names = sorted(NAMED_TUNINGS)
        return list(NAMED_TUNINGS[names[rng.integers(len(names))]])
    while True:
        t = [max(PITCH_MIN, b + int(rng.integers(-4, 3))) for b in TUNING_STANDARD]
        if all(t[i + 1] - t[i] >= 2 for i in range(N_STRING - 1)):
            return t


class _Ctx:
    def __init__(self, rng, tuning, max_fret, pitch_min, pitch_max):
        self.rng = rng
        self.tuning = list(tuning)
        self.max_fret = max_fret
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

    def playable(self, s, f):
        p = self.tuning[s] + f
        return 0 <= f <= self.max_fret and self.pitch_min <= p <= self.pitch_max

    def position(self):
        # lowest fret of the hand window; 0 = open position (frets 0..3)
        return int(self.rng.integers(0, self.max_fret - 2))

    def window(self, pos, allow_open):
        if pos == 0:
            return list(range(0, 4))
        frets = list(range(pos, min(pos + max_stretch(pos), self.max_fret) + 1))
        return ([0] + frets) if allow_open else frets

    def tempo(self):
        return float(self.rng.uniform(60.0, 180.0))

    def velocity(self, base, accent=False):
        v = base + self.rng.normal(0.0, 8.0) + (12.0 if accent else 0.0)
        return int(np.clip(round(v), 15, 127))

    def dynamic(self):
        return float(np.clip(self.rng.normal(78.0, 18.0), 30.0, 120.0))


def _scale_positions(ctx, pos, scale, root, allow_open):
    # all (string, fret) in the hand window whose pitch class is in the scale, sorted by pitch;
    # one position per pitch (random choice when two strings give the same pitch)
    pcs = {(root + i) % 12 for i in scale}
    by_pitch = {}
    for s in range(N_STRING):
        for f in ctx.window(pos, allow_open):
            if ctx.playable(s, f) and (ctx.tuning[s] + f) % 12 in pcs:
                by_pitch.setdefault(ctx.tuning[s] + f, []).append((s, f))
    out = []
    for p in sorted(by_pitch):
        c = by_pitch[p]
        out.append(c[ctx.rng.integers(len(c))])
    return out


def gen_scale_run(ctx, t0):
    rng = ctx.rng
    scale = SCALES[list(SCALES)[rng.integers(len(SCALES))]]
    pos = ctx.position()
    seq = _scale_positions(ctx, pos, scale, int(rng.integers(12)), allow_open=rng.random() < 0.3)
    if len(seq) < 3:
        return [], t0
    shape = rng.integers(4)
    if shape == 1:
        seq = seq[::-1]
    elif shape == 2:
        seq = seq + seq[-2::-1]
    elif shape == 3:
        # sequence in thirds: 1 3 2 4 3 5 ...
        seq = [x for i in range(len(seq) - 2) for x in (seq[i], seq[i + 2])]
    beat = 60.0 / ctx.tempo()
    step = beat / [2, 3, 4][rng.integers(3)]
    legato = rng.uniform(0.6, 1.0)
    base = ctx.dynamic()
    ev = []
    t = t0
    for i, (s, f) in enumerate(seq):
        ev.append(NoteEvent(t, step * legato, s, f, ctx.velocity(base, i % 4 == 0)))
        t += step
    return ev, t


def chord_voicing(ctx, pos=None, quality=None, root=None, n_try=200):
    # random playable voicing: >= 3 strings (or 2 for power chords) on mostly adjacent strings,
    # fret span within max_stretch, <= 4 fingers (barre = one finger), contains the root
    rng = ctx.rng
    if quality is None:
        quality = list(CHORDS)[rng.integers(len(CHORDS))]
    if root is None:
        root = int(rng.integers(12))
    pcs = {(root + i) % 12 for i in CHORDS[quality]}
    n_min = 2 if quality == '5' else 3
    best = None
    for _ in range(n_try):
        p = ctx.position() if pos is None else pos
        frets_win = ctx.window(p, allow_open=(p == 0 or rng.random() < 0.15))
        lo_s = int(rng.integers(0, N_STRING - n_min + 1))
        hi_s = int(rng.integers(lo_s + n_min - 1, N_STRING))
        v = {}
        for s in range(lo_s, hi_s + 1):
            cands = [f for f in frets_win if ctx.playable(s, f) and (ctx.tuning[s] + f) % 12 in pcs]
            if cands and rng.random() > 0.08:
                v[s] = cands[rng.integers(len(cands))]
        if len(v) < n_min:
            continue
        fretted = [f for f in v.values() if f > 0]
        if fretted and max(fretted) - min(fretted) > max_stretch(min(fretted)):
            continue
        fingers = len(set(fretted)) if fretted else 0
        if fingers > 4:
            continue
        if root not in {(ctx.tuning[s] + f) % 12 for s, f in v.items()}:
            continue
        bass_is_root = (ctx.tuning[min(v)] + v[min(v)]) % 12 == root
        if bass_is_root or best is None:
            best = v
        if bass_is_root or rng.random() < 0.2:   # inversions now and then
            break
    return best


def gen_chords(ctx, t0):
    # strummed or plucked block chords over a rhythm pattern
    rng = ctx.rng
    beat = 60.0 / ctx.tempo()
    n_chord = int(rng.integers(2, 6))
    base = ctx.dynamic()
    ev = []
    t = t0
    for _ in range(n_chord):
        v = chord_voicing(ctx)
        if v is None:
            continue
        n_hit = int(rng.integers(1, 5))
        hit_len = beat * [0.5, 1.0, 2.0][rng.integers(3)]
        muted = rng.random() < 0.15     # palm mute / staccato
        block = rng.random() < 0.25     # fingers pluck simultaneously instead of a strum
        for h in range(n_hit):
            down = (h % 2 == 0) or rng.random() < 0.3
            strings = sorted(v) if down else sorted(v, reverse=True)
            spread = 0.003 if block else rng.uniform(0.008, 0.035)
            vel = base + (0.0 if down else -10.0)
            for k, s in enumerate(strings):
                dur = rng.uniform(0.08, 0.25) if muted else hit_len * rng.uniform(0.85, 1.6)
                ev.append(NoteEvent(t + k * spread + abs(rng.normal(0.0, 0.002)), dur, s, v[s],
                                    ctx.velocity(vel, accent=(h == 0))))
            t += hit_len
    return ev, t


def gen_arpeggio(ctx, t0):
    # chord shapes picked string by string; notes ring until the shape changes
    rng = ctx.rng
    beat = 60.0 / ctx.tempo()
    step = beat / [2, 3, 4][rng.integers(3)]
    base = ctx.dynamic()
    patterns = ('up', 'down', 'updown', 'pima', 'random')
    pattern = patterns[rng.integers(len(patterns))]
    ev = []
    t = t0
    for _ in range(int(rng.integers(1, 5))):
        v = chord_voicing(ctx)
        if v is None:
            continue
        ss = sorted(v)
        if pattern == 'up':
            order = ss
        elif pattern == 'down':
            order = ss[::-1]
        elif pattern == 'updown':
            order = ss + ss[-2:0:-1]
        elif pattern == 'pima':
            # thumb on the bass, fingers on the top strings
            top = ss[-3:] if len(ss) >= 4 else ss[1:]
            order = [ss[0]] + top + top[-2::-1]
        else:
            order = [ss[i] for i in rng.integers(len(ss), size=len(ss) + 2)]
        n_rep = int(rng.integers(1, 4))
        shape_end = t + step * len(order) * n_rep
        for r in range(n_rep):
            for k, s in enumerate(order):
                ring = shape_end - t + rng.uniform(0.0, 0.3)
                ev.append(NoteEvent(t, ring, s, v[s], ctx.velocity(base, accent=(k == 0))))
                t += step
    return ev, t


def gen_line(ctx, t0):
    # single-note melody: random walk on a scale inside a hand position, with position shifts
    rng = ctx.rng
    scale = SCALES[list(SCALES)[rng.integers(len(SCALES) - 1)]]   # no chromatic
    root = int(rng.integers(12))
    beat = 60.0 / ctx.tempo()
    base = ctx.dynamic()
    ev = []
    t = t0
    pos = ctx.position()
    for _ in range(int(rng.integers(1, 4))):
        seq = _scale_positions(ctx, pos, scale, root, allow_open=rng.random() < 0.3)
        if len(seq) >= 2:
            i = int(rng.integers(len(seq)))
            for _ in range(int(rng.integers(4, 16))):
                dur = beat * [0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 2.0][rng.integers(7)]
                if rng.random() < 0.1:
                    t += dur      # rest
                    continue
                s, f = seq[i]
                ev.append(NoteEvent(t, dur * rng.uniform(0.7, 1.0), s, f, ctx.velocity(base)))
                t += dur
                i = int(np.clip(i + rng.choice([-2, -1, -1, 1, 1, 2, 3, -3]), 0, len(seq) - 1))
        pos = int(np.clip(pos + rng.integers(-5, 6), 0, ctx.max_fret - 3))
    return ev, t


GENERATORS = {'scale': gen_scale_run, 'chord': gen_chords, 'arpeggio': gen_arpeggio, 'line': gen_line}
DEFAULT_WEIGHTS = {'scale': 0.2, 'chord': 0.3, 'arpeggio': 0.25, 'line': 0.25}


def enforce_constraints(events, tuning, max_fret=MAX_FRET, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX,
                        gap=0.005, min_dur=0.03):
    ev = [NoteEvent(**asdict(e)) for e in events
          if 0 <= e.fret <= max_fret and pitch_min <= tuning[e.string] + e.fret <= pitch_max]
    ev.sort(key=lambda e: (e.onset, e.string))
    # one note per string: a new pluck cuts the previous note; drop notes that end up too short
    keep = []
    last = {}
    for e in ev:
        p = last.get(e.string)
        if p is not None and p.offset > e.onset - gap:
            p.duration = e.onset - gap - p.onset
        keep.append(e)
        last[e.string] = e
    ev = [e for e in keep if e.duration >= min_dur]
    # fret hand: at every onset, fretted notes that are still sounding must fit the hand
    # together with the new note; otherwise the finger left them, cut them there
    for i, e in enumerate(ev):
        if e.fret == 0:
            continue
        while True:
            sounding = [q for q in ev[:i] if q.fret > 0 and q.offset > e.onset]
            frets = [q.fret for q in sounding] + [e.fret]
            lo, hi = min(frets), max(frets)
            if hi - lo <= max_stretch(lo) and len(set(frets)) <= 4:
                break
            # cut the sounding note farthest from the new one, then re-check
            far = max(sounding, key=lambda q: (abs(q.fret - e.fret), -q.onset))
            far.duration = max(e.onset - far.onset, 0.0)
    return [e for e in ev if e.duration >= min_dur]


def check_constraints(events, tuning, max_fret=MAX_FRET, pitch_min=PITCH_MIN, pitch_max=PITCH_MAX):
    # list of violation strings (empty = ok)
    bad = []
    for e in events:
        if not (0 <= e.fret <= max_fret):
            bad.append('fret %d' % e.fret)
        if not (pitch_min <= tuning[e.string] + e.fret <= pitch_max):
            bad.append('pitch %d' % (tuning[e.string] + e.fret))
    for s in range(N_STRING):
        es = sorted((e for e in events if e.string == s), key=lambda e: e.onset)
        for a, b in zip(es, es[1:]):
            if a.offset > b.onset + 1e-9:
                bad.append('overlap on string %d at %.3f' % (s, b.onset))
    for e in events:
        sounding = [q.fret for q in events
                    if q.fret > 0 and q.onset <= e.onset + 1e-9 and q.offset > e.onset + 1e-9]
        if sounding:
            lo, hi = min(sounding), max(sounding)
            if hi - lo > max_stretch(lo) or len(set(sounding)) > 4:
                bad.append('span %s at %.3f' % (sorted(sounding), e.onset))
    return bad


def generate_sequence(rng, duration, tuning=None, max_fret=MAX_FRET, weights=None,
                      pitch_min=PITCH_MIN, pitch_max=PITCH_MAX):
    # events covering [0, duration) s; returns (events, tuning)
    if tuning is None:
        tuning = sample_tuning(rng)
    ctx = _Ctx(rng, tuning, max_fret, pitch_min, pitch_max)
    weights = weights or DEFAULT_WEIGHTS
    names = sorted(weights)
    prob = np.array([weights[k] for k in names], dtype=np.float64)
    prob /= prob.sum()
    t = float(rng.uniform(0.0, 0.5))
    events = []
    while t < duration:
        ev, t_end = GENERATORS[names[rng.choice(len(names), p=prob)]](ctx, t)
        events += ev
        t = max(t_end, t + 0.1) + float(rng.exponential(0.3))
    events = [e for e in events if e.onset < duration - 0.05]
    for e in events:
        e.duration = min(e.duration, duration - e.onset)
    return enforce_constraints(events, tuning, max_fret, pitch_min, pitch_max), list(tuning)
