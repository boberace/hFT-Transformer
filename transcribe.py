#! python
# Transcribe a piano recording (wav/flac/ogg) to MIDI with the pretrained hFT-Transformer.
# usage: python transcribe.py input.flac output.mid [-max_sec 60]

import argparse
import io
import json
import os
import pickle
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'model'))
import amt  # noqa: E402
import model_spec2midi  # noqa: E402,F401  (needed so pickle can find the model classes)


def load_feature(f_wav, config, max_sec=None):
    wave, sr = sf.read(f_wav, dtype='float32', always_2d=True)
    if max_sec is not None:
        wave = wave[:int(max_sec * sr)]
    wave_mono = torch.from_numpy(wave.mean(axis=1))
    wave_16k = torchaudio.functional.resample(wave_mono, sr, config['feature']['sr'])
    tr_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['feature']['sr'], n_fft=config['feature']['fft_bins'],
        win_length=config['feature']['window_length'], hop_length=config['feature']['hop_sample'],
        pad_mode=config['feature']['pad_mode'], n_mels=config['feature']['mel_bins'], norm='slaney')
    mel_spec = tr_mel(wave_16k)
    return (torch.log(mel_spec + config['feature']['log_offset'])).T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('-model', default=os.path.join(ROOT, 'checkpoint/MAESTRO-V3/model_016_003.pkl'))
    parser.add_argument('-config', default=os.path.join(ROOT, 'corpus/config.json'))
    parser.add_argument('-max_sec', type=float, default=None, help='only transcribe the first N seconds')
    parser.add_argument('-thred', type=float, default=0.5, help='onset/offset/mpe threshold')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    config['input']['min_value'] = float(np.log(config['feature']['log_offset']))

    t0 = time.time()
    # the checkpoint is a whole pickled model saved on CUDA; remap its tensors to CPU while unpickling
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
    a = amt.AMT(config, None)
    if torch.backends.mps.is_available():
        a.device = 'mps'
    with open(args.model, 'rb') as f:
        a.model = pickle.load(f).to(a.device).eval()
    for m in a.model.modules():  # submodules also remember the device they were trained on
        if hasattr(m, 'device'):
            m.device = a.device
        for k, v in vars(m).items():  # and keep plain (unregistered) tensors like scale_freq
            if isinstance(v, torch.Tensor):
                setattr(m, k, v.to(a.device))
    print(f'model loaded on {a.device}')

    feature = load_feature(args.input, config, args.max_sec)
    print(f'feature: {tuple(feature.shape)} ({feature.shape[0] * config["feature"]["hop_sample"] / config["feature"]["sr"]:.1f}s)')

    # 'combination' mode returns 1st-stage (A) and 2nd-stage (B) outputs; the paper reports the 2nd
    _, _, _, _, onset, offset, mpe, velocity = a.transcript(feature, mode='combination')
    notes = a.mpe2note(a_onset=onset, a_offset=offset, a_mpe=mpe, a_velocity=velocity,
                       thred_onset=args.thred, thred_offset=args.thred, thred_mpe=args.thred,
                       mode_velocity='ignore_zero', mode_offset='shorter')
    a.note2midi(notes, args.output)
    with open(os.path.splitext(args.output)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(notes, f, indent=1)
    print(f'{len(notes)} notes -> {args.output} ({time.time() - t0:.1f}s)')
