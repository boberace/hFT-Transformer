#! python
# Make the guitar dataset read by training/dataset_guitar.py GuitarDataset.
#
# d_dataset/<split>/{feature,onset,offset,mpe,tab,tab_onset,tuning,idx}.npy
# Files are concatenated as in corpus/make_dataset.py: margin_b pad frames first, then each file
# followed by margin_f+num_frame-1 pad frames. Pad frames hold feature log(log_offset), zero labels,
# and the tuning of the file they follow (the leading margin_b takes the first file's tuning).
# No velocity.npy: GuitarSet has no velocity.

import argparse
import json
import os
import pickle

import numpy as np


def make_dataset(filelist, attribute, d_feature, d_label, d_dataset, config):
    print('-'+str(attribute)+'-')
    margin_b = config['input']['margin_b']
    pad = config['input']['margin_f'] + config['input']['num_frame'] - 1
    num_note = config['midi']['num_note']
    n_string = config['guitar']['n_string']
    n_bins = config['feature']['mel_bins']

    with open(filelist, 'r', encoding='utf-8') as f:
        a_fname = [line.rstrip('\n') for line in f if not line.startswith('#') and line.strip()]

    # num frame
    a_num_frame = []
    for fname in a_fname:
        with open(d_feature+'/'+fname+'.pkl', 'rb') as f:
            num_frame_feature = pickle.load(f).shape[0]
        with open(d_label+'/'+fname+'.pkl', 'rb') as f:
            num_frame_label = len(pickle.load(f)['mpe'])
        if num_frame_feature < num_frame_label:
            print('(warning) '+str(fname)+': num_frame_feature('+str(num_frame_feature)+') < num_frame_label('+str(num_frame_label)+')')
        a_num_frame.append(max(num_frame_feature, num_frame_label))
    total_num_frame = margin_b + sum(n + pad for n in a_num_frame)
    print('files: '+str(len(a_fname))+' total_num_frame: '+str(total_num_frame))

    d_split = os.path.join(d_dataset, attribute)
    os.makedirs(d_split, exist_ok=True)
    def open_array(name, dtype, ncol, fill):
        a = np.lib.format.open_memmap(os.path.join(d_split, name+'.npy'), mode='w+', dtype=dtype, shape=(total_num_frame, ncol))
        a[:] = fill
        return a

    zero_value = np.log(config['feature']['log_offset']) if config['feature']['log_offset'] > 0.0 else config['feature']['log_offset']
    a_feature = open_array('feature', np.float16, n_bins, zero_value)
    a_onset = open_array('onset', np.float16, num_note, 0)
    a_offset = open_array('offset', np.float16, num_note, 0)
    a_mpe = open_array('mpe', np.uint8, num_note, 0)
    a_tab = open_array('tab', np.int8, n_string, 0)
    a_tab_onset = open_array('tab_onset', np.float16, n_string, 0)
    a_tuning = open_array('tuning', np.int8, n_string, 0)
    a_idx = np.zeros(sum(a_num_frame), dtype=np.int32)

    loc_i = 0
    loc_d = margin_b
    for i, fname in enumerate(a_fname):
        print('('+str(i)+'/'+str(len(a_fname))+') '+fname)
        num_frame = a_num_frame[i]
        with open(d_feature+'/'+fname+'.pkl', 'rb') as f:
            feature = np.asarray(pickle.load(f), dtype=np.float32)
        with open(d_label+'/'+fname+'.pkl', 'rb') as f:
            label = pickle.load(f)
        nl = len(label['mpe'])

        a_feature[loc_d:loc_d+len(feature)] = feature
        a_onset[loc_d:loc_d+nl] = label['onset']
        a_offset[loc_d:loc_d+nl] = label['offset']
        a_mpe[loc_d:loc_d+nl] = label['mpe']
        a_tab[loc_d:loc_d+nl] = label['tab']
        a_tab_onset[loc_d:loc_d+nl] = label['tab_onset']
        if i == 0:
            a_tuning[:loc_d] = label['tuning']
        a_tuning[loc_d:loc_d+num_frame+pad] = label['tuning']

        a_idx[loc_i:loc_i+num_frame] = np.arange(loc_d, loc_d+num_frame)
        loc_i += num_frame
        loc_d += num_frame + pad

    for a in [a_feature, a_onset, a_offset, a_mpe, a_tab, a_tab_onset, a_tuning]:
        a.flush()
    np.save(os.path.join(d_split, 'idx.npy'), a_idx)
    with open(os.path.join(d_split, 'fname.list'), 'w', encoding='utf-8') as f:
        f.write(''.join(fname+'\n' for fname in a_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_dataset', help='dataset directory (output)')
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_feature', help='feature file directory')
    parser.add_argument('-d_label', help='label file directory')
    parser.add_argument('-f_config_in', help='config json file (input)')
    parser.add_argument('-f_config_out', help='config json file (output)')
    args = parser.parse_args()

    print('** make_dataset_guitar **')
    print(' corpus list : '+str(args.d_list))
    print(' feature     : '+str(args.d_feature))
    print(' label       : '+str(args.d_label))
    print(' dataset     : '+str(args.d_dataset))

    with open(args.f_config_in, 'r', encoding='utf-8') as f:
        config = json.load(f)

    d_list = args.d_list.rstrip('/')
    for attribute in ['train', 'valid', 'test']:
        make_dataset(d_list+'/'+attribute+'.list', attribute, args.d_feature.rstrip('/'), args.d_label.rstrip('/'),
                     args.d_dataset.rstrip('/'), config)

    # same extra entries as corpus/make_dataset.py (features are not normalized)
    config['input']['max_value'] = 0.0
    config['input']['min_value'] = float(np.log(config['feature']['log_offset']))
    config['feature']['n_bins'] = config['feature']['mel_bins']
    with open(args.f_config_out, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=False)

    print('** done **')
