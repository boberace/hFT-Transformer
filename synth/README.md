# synth: labeled synthetic guitar audio

Pretraining data for string/fret transcription: the same pitch must sound different on
different strings.

| file | what |
|---|---|
| `tabs_procedural.py` | note events (onset, duration, string, fret, velocity) from scale runs, arpeggios, single-note lines, strummed/plucked chords; random tunings; one note per string, fret-hand span and finger limits (`check_constraints`) |
| `strings.py` | per-string digital waveguide: integer delay + one-pole loss (T60 at f0 and 4 kHz, from `sigma = b1 + b3 f^2`, extra finger damping on fretted notes) + Thiran tuning + cascade of 2nd-order dispersion allpasses fitted to `f_n = n f0 sqrt(1 + B n^2)`; pluck- and pickup-position combs at absolute distances from the bridge. Gauge table `GAUGES` (steel acoustic 12-53, electric 10-46, nylon normal tension) gives B from core diameter, tension, Young's modulus and scale length; retuning scales tension, fretting scales B by `2^(2 fret / 12)` |
| `sensor_aug.py` | headstock contact-sensor domain: per-string transfer function (shared headstock resonances + per-string deviations) on the stems, band-limit, white/pink noise, handling rumble, finger thumps, optional soft clip; no room. Responses are dB curves turned into minimum-phase FIRs, so `MeasuredResponse(freqs_hz, mag_db)` can replace the parametric model |
| `render.py` | clip renderer + CLI. Stems at 32 kHz -> 16 kHz -> sensor per-string TF -> mix -> body IR (acoustic/nylon; synthetic modal IR or `--body_ir_dir`) or pickup resonance low-pass (electric) -> sensor band-limit/noise -> wav |
| `check_same_pitch.py` | renders one pitch on every string that reaches it, fits B, compares partial envelopes and log-mel spectra, saves a plot |

```bash
python -m synth.render --out corpus/Synth --hours 10 --clip_sec 30 --workers 8
python -m synth.render --out corpus/Synth_clean --hours 1 --no_sensor
python -m synth.check_same_pitch --pitch 64 --out corpus/Synth/check
```

Output: `corpus/Synth/wav/<name>.wav` (16 kHz mono int16) and `corpus/Synth/note/<name>.json`:

```json
{"guitar": "acoustic", "tuning": [40, 45, 50, 55, 59, 64], "sr": 16000, "sensor_aug": true,
 "pickups": [], "notes": [{"onset": 0.31, "offset": 0.52, "pitch": 64, "velocity": 87, "string": 5, "fret": 0}]}
```

`pitch = tuning[string] + fret` (integer MIDI); the audio carries a random global detune of
±15 cents and ±3 cents per string that is not in the label. `offset` is where the string is
damped (a short release follows). Clip names are `<prefix>_<guitar>_<index>`; the seed of a clip is
`(--seed, index)`, so `--start` extends a corpus reproducibly.

Notes / limits
- Dispersion filters are cached on a 25-cent f0 grid (0.1-sample period grid for short loops) and
  5 % B steps; the Thiran stage retunes partial 1 exactly. Partial error is < 0.1 cent at n = 1 and
  at most ~17 cents (n <= 40) where the target stretch is 400+ cents.
- Not modelled: bends, slides, hammer-on/pull-off articulation, fret buzz, sympathetic strings,
  string-to-string coupling through the bridge/nut.
- For the headstock sensor, the string's energy reaches the neck at the fret (fretted) or nut
  (open); a measured per-string/per-fret response would be the next step.
