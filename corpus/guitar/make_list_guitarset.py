#! python
# Make train/valid/test lists for GuitarSet (leave-one-player-out) and link audio/annotation under plain names.
# file name: <player>_<style><n>-<tempo>-<key>_<comp|solo>  (e.g. 00_BN1-129-Eb_comp)

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_annotation', help='GuitarSet annotation directory (input, *.jams)')
    parser.add_argument('-d_audio', help='GuitarSet audio_mono-mic directory (input, *_mic.wav)')
    parser.add_argument('-d_wav', help='wav directory (output, symlinks <fname>.wav)')
    parser.add_argument('-d_jams', help='jams directory (output, symlinks <fname>.jams)')
    parser.add_argument('-d_list', help='corpus list directory (output)')
    parser.add_argument('-player_valid', default='04')
    parser.add_argument('-player_test', default='05')
    args = parser.parse_args()

    print('** make_list_guitarset **')
    a_list = {'train': [], 'valid': [], 'test': []}
    for f_jams in sorted(os.listdir(args.d_annotation)):
        if not f_jams.endswith('.jams'):
            continue
        fname = f_jams[:-len('.jams')]
        f_wav = os.path.join(args.d_audio, fname+'_mic.wav')
        if not os.path.exists(f_wav):
            print('(warning) no audio for '+fname)
            continue

        player = fname.split('_')[0]
        if player == args.player_test:
            attribute = 'test'
        elif player == args.player_valid:
            attribute = 'valid'
        else:
            attribute = 'train'
        a_list[attribute].append(fname)

        for src, dst in [(f_wav, os.path.join(args.d_wav, fname+'.wav')),
                         (os.path.join(args.d_annotation, f_jams), os.path.join(args.d_jams, fname+'.jams'))]:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)

    for attribute, a_fname in a_list.items():
        print(' '+attribute+': '+str(len(a_fname)))
        with open(os.path.join(args.d_list, attribute+'.list'), 'w', encoding='utf-8') as f:
            for fname in a_fname:
                f.write(fname+'\n')

    print('** done **')
