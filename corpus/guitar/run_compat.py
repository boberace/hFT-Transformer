#! python
# Run an unmodified corpus/*.py script under the current numpy/torchaudio.
# usage: python corpus/guitar/run_compat.py corpus/conv_wav2fe.py [script args...]
#
# - numpy >= 1.24 removed np.int (used by corpus/conv_note2ref.py)
# - torchaudio >= 2.9 routes torchaudio.load through torchcodec (needs ffmpeg);
#   without torchcodec, fall back to soundfile, which returns the same [channel, sample] float32 tensor

import runpy
import sys

import numpy as np

if not hasattr(np, 'int'):
    np.int = int

try:
    import torchcodec  # noqa: F401
except ImportError:
    import soundfile as sf
    import torch
    import torchaudio

    def load_soundfile(f_wav, *args, **kwargs):
        wave, sr = sf.read(f_wav, dtype='float32', always_2d=True)
        return torch.from_numpy(wave.T.copy()), sr

    torchaudio.load = load_soundfile

if __name__ == '__main__':
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name='__main__')
