#! python
# Round-trip check: decode tab/tab_onset labels back to (string, fret, onset) notes and compare with the raw JAMS.
# Also checks that the dataset .npy slices equal the per-file label pickles.
# usage: python corpus/guitar/check_roundtrip_guitar.py -d corpus/GuitarSet -split test -n 3

import argparse
import json
import os
import pickle

import jams
import numpy as np


def decode_tab(tab, tab_onset, nframe_in_sec):
    # onset = local maximum (>= 0.5) of tab_onset; fret = tab at that frame; note lasts while tab keeps that value
    a_note = []
    nframe, n_string = tab.shape
    for s in range(n_string):
        v = np.concatenate([[0.0], tab_onset[:, s].astype(np.float32), [0.0]])
        a_peak = [f for f in range(nframe) if v[f+1] >= 0.5 and v[f+1] >= v[f] and v[f+1] > v[f+2]]
        for k, f in enumerate(a_peak):
            if tab[f, s] == 0:
                continue
            end = a_peak[k+1] if k+1 < len(a_peak) else nframe
            g = f
            while g+1 < end and tab[g+1, s] == tab[f, s]:
                g += 1
            a_note.append({'string': s, 'fret': int(tab[f, s]) - 1, 'onset': f / nframe_in_sec, 'offset': g / nframe_in_sec})
    return a_note


def jams_notes(f_jams, tuning):
    a_note = []
    for ann in jams.load(f_jams).search(namespace='note_midi'):
        s = int(ann.annotation_metadata.data_source)
        for obs in ann.data:
            a_note.append({'string': s, 'fret': int(round(obs.value)) - tuning[s], 'onset': obs.time, 'offset': obs.time + obs.duration})
    return a_note


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='GuitarSet corpus directory', default='corpus/GuitarSet')
    parser.add_argument('-split', default='test')
    parser.add_argument('-n', type=int, default=3)
    args = parser.parse_args()

    with open(os.path.join(args.d, 'dataset', 'config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    nframe_in_sec = config['feature']['sr'] / config['feature']['hop_sample']
    tol = 0.5 / nframe_in_sec + 1e-6

    d_split = os.path.join(args.d, 'dataset', args.split)
    with open(os.path.join(d_split, 'fname.list'), 'r', encoding='utf-8') as f:
        a_fname = [line.rstrip('\n') for line in f]
    idx = np.load(os.path.join(d_split, 'idx.npy'))
    ds = {name: np.load(os.path.join(d_split, name+'.npy'), mmap_mode='r') for name in ['onset', 'offset', 'mpe', 'tab', 'tab_onset', 'tuning']}

    ok_all = True
    loc_i = 0
    for i, fname in enumerate(a_fname):
        with open(os.path.join(args.d, 'label', fname+'.pkl'), 'rb') as f:
            label = pickle.load(f)
        nl = len(label['mpe'])
        with open(os.path.join(args.d, 'feature', fname+'.pkl'), 'rb') as f:
            num_frame = max(pickle.load(f).shape[0], nl)
        start = int(idx[loc_i])
        loc_i += num_frame
        if i >= args.n:
            continue

        # dataset slice == label pickle
        same = all(np.array_equal(np.asarray(ds[k][start:start+nl]), label[k].astype(ds[k].dtype)) for k in ['onset', 'offset', 'mpe', 'tab', 'tab_onset'])
        same = same and np.all(ds['tuning'][start:start+num_frame] == label['tuning'])

        tuning = [int(t) for t in label['tuning']]
        a_ref = jams_notes(os.path.join(args.d, 'jams', fname+'.jams'), tuning)
        a_est = decode_tab(np.asarray(ds['tab'][start:start+nl]), np.asarray(ds['tab_onset'][start:start+nl]), nframe_in_sec)

        used = [False] * len(a_est)
        a_miss = []
        for r in a_ref:
            hit = None
            for j, e in enumerate(a_est):
                if not used[j] and e['string'] == r['string'] and e['fret'] == r['fret'] and abs(e['onset'] - r['onset']) <= tol:
                    hit = j
                    break
            if hit is None:
                a_miss.append(r)
            else:
                used[hit] = True
        n_extra = used.count(False)
        ok = same and not a_miss and n_extra == 0
        ok_all = ok_all and ok
        print(f'{fname}: dataset==label {same}, jams notes {len(a_ref)}, decoded {len(a_est)}, '
              f'matched {len(a_ref)-len(a_miss)}, missed {len(a_miss)}, extra {n_extra} -> {"OK" if ok else "MISMATCH"}')
        for r in a_miss[:10]:
            print('   missed', r)

    print('round-trip', 'OK' if ok_all else 'MISMATCH')
