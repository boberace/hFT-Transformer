#! python
# Convert the pickled piano checkpoint (whole model, saved on CUDA) to a plain CPU state_dict.
# usage: python model/convert_piano_ckpt.py [-i checkpoint/MAESTRO-V3/model_016_003.pkl] [-o checkpoint/MAESTRO-V3/model_016_003_state.pt]

import argparse
import io
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'model'))
sys.path.insert(0, ROOT)
import model_spec2midi  # noqa: E402,F401  (needed so pickle can find the model classes)


def load_piano_pickle(f_pkl):
    # remap CUDA tensors to CPU while unpickling (same workaround as transcribe.py)
    load_orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
    try:
        with open(f_pkl, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = load_orig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default=os.path.join(ROOT, 'checkpoint/MAESTRO-V3/model_016_003.pkl'))
    parser.add_argument('-o', default=os.path.join(ROOT, 'checkpoint/MAESTRO-V3/model_016_003_state.pt'))
    args = parser.parse_args()

    model = load_piano_pickle(args.i)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, args.o)
    print(f'{len(state)} tensors, {sum(v.numel() for v in state.values())} params -> {args.o}')
