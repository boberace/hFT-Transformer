#! python
# Guitar dataset: same windowing as dataset.MyDataset, read from memory-mapped .npy files.
#
# d_dataset/<split>/ (all arrays share the padded, concatenated frame axis used by corpus/make_dataset.py)
#   feature.npy    float16 [n_total, n_bins]
#   onset.npy      float16 [n_total, n_note]
#   offset.npy     float16 [n_total, n_note]
#   mpe.npy        uint8   [n_total, n_note]
#   velocity.npy   int8    [n_total, n_note]   (optional; velocity loss is skipped without it)
#   tab.npy        int8    [n_total, n_string] (0: silent, 1..n_fret+1: fret 0..n_fret)
#   tab_onset.npy  float16 [n_total, n_string]
#   tuning.npy     int8    [n_total, n_string] open-string MIDI numbers per frame
#   idx.npy        int32   [n_window]          window start frames

import os

import numpy as np
import torch


class GuitarDataset(torch.utils.data.Dataset):
    def __init__(self, d_split, config, n_slice):
        super().__init__()
        def load(name):
            return np.load(os.path.join(d_split, name+'.npy'), mmap_mode='r')

        self.feature = load('feature')
        self.label_onset = load('onset')
        self.label_offset = load('offset')
        self.label_mpe = load('mpe')
        self.flag_velocity = os.path.exists(os.path.join(d_split, 'velocity.npy'))
        if self.flag_velocity:
            self.label_velocity = load('velocity')
        self.label_tab = load('tab')
        self.label_tab_onset = load('tab_onset')
        self.tuning = load('tuning')

        idx = np.load(os.path.join(d_split, 'idx.npy'))
        if n_slice > 1:
            idx = idx[:int(len(idx) / n_slice) * n_slice][::n_slice]
        self.idx = idx
        self.config = config

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = int(self.idx[i])
        fs = s - self.config['input']['margin_b']
        fe = s + self.config['input']['num_frame'] + self.config['input']['margin_f']
        e = s + self.config['input']['num_frame']

        item = {
            # spec: [n_bins, margin+num_frame+margin]
            'spec': torch.from_numpy(np.ascontiguousarray(self.feature[fs:fe].T)).float(),
            # labels: [num_frame, n_note] / [num_frame, n_string]
            'onset': torch.from_numpy(np.array(self.label_onset[s:e])).float(),
            'offset': torch.from_numpy(np.array(self.label_offset[s:e])).float(),
            'mpe': torch.from_numpy(np.array(self.label_mpe[s:e])).float(),
            'tab': torch.from_numpy(np.array(self.label_tab[s:e])).long(),
            'tab_onset': torch.from_numpy(np.array(self.label_tab_onset[s:e])).float(),
            # tuning is constant within a file; take it at the window start
            'tuning': torch.from_numpy(np.array(self.tuning[s])).long(),
        }
        if self.flag_velocity:
            item['velocity'] = torch.from_numpy(np.array(self.label_velocity[s:e])).long()
        return item
