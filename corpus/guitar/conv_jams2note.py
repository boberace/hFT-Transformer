#! python
# Convert GuitarSet JAMS to note files.
#
# <fname>.json: {"tuning": [6 open-string MIDI numbers, string 0 = low E],
#                "note": [{"onset", "offset", "pitch", "velocity", "string", "fret"}, ...]}
#   pitch = rounded MIDI of the note_midi annotation, fret = pitch - tuning[string], velocity = 0 (not annotated)
# <fname>.txt : same 4-column text as corpus/conv_midi2note.py (read by corpus/conv_note2ref.py)

import argparse
import json

import jams


def jams2note(config, f_jams, tuning):
    jam = jams.load(f_jams)
    n_fret = config['guitar']['n_fret']

    a_note = []
    for ann in jam.search(namespace='note_midi'):
        string = int(ann.annotation_metadata.data_source)
        for obs in ann.data:
            pitch = int(round(obs.value))
            fret = pitch - tuning[string]
            if (fret < 0) or (fret > n_fret):
                # pitch-bend/annotation drift below the open string or past the last fret
                print('(warning) '+f_jams+': string '+str(string)+' pitch '+str(obs.value)+' -> fret '+str(fret)+', clipped')
                fret = min(max(fret, 0), n_fret)
                pitch = tuning[string] + fret
            a_note.append({'onset': float(obs.time),
                           'offset': float(obs.time + obs.duration),
                           'pitch': pitch,
                           'velocity': 0,
                           'string': string,
                           'fret': fret})
    a_note_sort = sorted(sorted(a_note, key=lambda x: x['pitch']), key=lambda x: x['onset'])

    return a_note_sort


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d_list', help='corpus list directory')
    parser.add_argument('-d_jams', help='jams file directory (input)')
    parser.add_argument('-d_note', help='note file directory (output)')
    parser.add_argument('-config', help='config file')
    args = parser.parse_args()

    print('** conv_jams2note: convert jams to note **')
    print(' directory')
    print('  jams (input)  : '+str(args.d_jams))
    print('  note (output) : '+str(args.d_note))
    print('  corpus list   : '+str(args.d_list))
    print(' config file    : '+str(args.config))

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # GuitarSet is recorded in standard tuning
    tuning = config['guitar']['tuning_standard']

    a_attribute = ['train', 'test', 'valid']
    for attribute in a_attribute:
        print('-'+attribute+'-')
        with open(args.d_list.rstrip('/')+'/'+str(attribute)+'.list', 'r', encoding='utf-8') as f:
            a_input = f.readlines()

        for i in range(len(a_input)):
            fname = a_input[i].rstrip('\n')
            print(fname)

            a_note = jams2note(config, args.d_jams.rstrip('/')+'/'+fname+'.jams', tuning)
            with open(args.d_note.rstrip('/')+'/'+fname+'.json', 'w', encoding='utf-8') as f:
                json.dump({'tuning': tuning, 'note': a_note}, f, ensure_ascii=False, indent=4, sort_keys=False)
            with open(args.d_note.rstrip('/')+'/'+fname+'.txt', 'w', encoding='utf-8') as f:
                f.write('OnsetTime\tOffsetTime\tVelocity\tMidiPitch\n')
                for note in a_note:
                    f.write(str(note['onset'])+'\t')
                    f.write(str(note['offset'])+'\t')
                    f.write(str(note['velocity'])+'\t')
                    f.write(str(note['pitch'])+'\n')

    print('** done **')
