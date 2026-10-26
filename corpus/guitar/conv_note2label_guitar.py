#! python
# Convert guitar note files (corpus/guitar/conv_jams2note.py) to frame labels.
#
# <fname>.pkl: {'mpe'      : bool    [nframe, num_note]  (corpus/conv_note2label.py note2label)
#               'onset'    : float32 [nframe, num_note]
#               'offset'   : float32 [nframe, num_note]
#               'tab'      : int8    [nframe, n_string]  0: silent, fret+1 while sounding
#               'tab_onset': float32 [nframe, n_string]  triangular onset (same shape as 'onset')
#               'tuning'   : int8    [n_string]}

import argparse
import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from conv_note2label import note2label


def note2tab(config, a_note, nframe):
    hop_ms = 1000 * config['feature']['hop_sample'] / config['feature']['sr']
    onset_sharpness = int(50.0 / hop_ms + 0.5)
    nframe_in_sec = config['feature']['sr'] / config['feature']['hop_sample']
    n_string = config['guitar']['n_string']

    a_tab = np.zeros((nframe, n_string), dtype=np.int8)
    a_tab_onset = np.zeros((nframe, n_string), dtype=np.float32)

    for string in range(n_string):
        a_note_string = sorted([note for note in a_note if note['string'] == string], key=lambda x: x['onset'])
        for i, note in enumerate(a_note_string):
            onset_frame = int(note['onset'] * nframe_in_sec + 0.5)
            offset_frame = int(note['offset'] * nframe_in_sec + 0.5)
            # a string sounds one note at a time: a new onset cuts the previous note
            if i+1 < len(a_note_string):
                offset_frame = min(offset_frame, int(a_note_string[i+1]['onset'] * nframe_in_sec + 0.5) - 1)
            a_tab[onset_frame:max(onset_frame, offset_frame)+1, string] = note['fret'] + 1

            onset_ms = note['onset'] * 1000.0
            for j in range(-onset_sharpness, onset_sharpness+1):
                f = onset_frame + j
                if (f < 0) or (f >= nframe):
                    continue
                onset_val = max(0.0, 1.0 - (abs(f * hop_ms - onset_ms) / (onset_sharpness * hop_ms)))
                a_tab_onset[f, string] = max(a_tab_onset[f, string], onset_val)

    return a_tab, a_tab_onset


def note2label_guitar(config, f_note):
    with open(f_note, 'r', encoding='utf-8') as f:
        a_note_file = json.load(f)
    a_note = a_note_file['note']

    # note2label reads a plain note list from a file
    with tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8') as f_tmp:
        json.dump(a_note, f_tmp)
        f_tmp.flush()
        a_label = note2label(config, f_tmp.name, False)

    label = {
        'mpe': np.array(a_label['mpe'], dtype=bool),
        'onset': np.array(a_label['onset'], dtype=np.float32),
        'offset': np.array(a_label['offset'], dtype=np.float32),
    }
    label['tab'], label['tab_onset'] = note2tab(config, a_note, len(label['mpe']))
    label['tuning'] = np.array(a_note_file['tuning'], dtype=np.int8)

    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_note', help='note file directory (input)')
    parser.add_argument('-d_label', help='label file directory (output)')
    parser.add_argument('-config', help='config file')
    args = parser.parse_args()

    print('** conv_note2label_guitar: convert note to label **')
    print(' directory')
    print('  note (input)  : '+str(args.d_note))
    print('  label (output): '+str(args.d_label))
    print('  corpus list   : '+str(args.d_list))
    print(' config file    : '+str(args.config))

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    a_attribute = ['train', 'test', 'valid']
    for attribute in a_attribute:
        print('-'+attribute+'-')
        with open(args.d_list.rstrip('/')+'/'+str(attribute)+'.list', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        for i in range(len(a_input)):
            fname = a_input[i].rstrip('\n')
            print(fname)

            a_label = note2label_guitar(config, args.d_note.rstrip('/')+'/'+fname+'.json')
            with open(args.d_label.rstrip('/')+'/'+fname+'.pkl', 'wb') as f:
                pickle.dump(a_label, f, protocol=4)

    print('** done **')
