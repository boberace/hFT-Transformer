#! python
# Train the guitar hFT-Transformer (pitch + string/fret) on CUDA, Apple MPS or CPU.
# usage: python training/m_training_guitar.py -config corpus/config_guitar.json -d_dataset corpus/GuitarSet/dataset \
#            -d_out checkpoint/GuitarSet -init_piano checkpoint/MAESTRO-V3/model_016_003_state.pt

import argparse
import datetime
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
import train_guitar
import dataset_guitar
from model.model_guitar import build_model, load_piano_init, use_sdpa


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='config json file', default='corpus/config_guitar.json')
    parser.add_argument('-d_out', help='output directory', default='checkpoint/guitar')
    parser.add_argument('-d_dataset', help='dataset directory (train/ valid/ subdirs)', required=True)
    parser.add_argument('-n_slice', help='use every n-th window start(16)', type=int, default=16)
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch', type=int, default=4)
    parser.add_argument('-accum', help='gradient accumulation steps(2)', type=int, default=2)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-clip', type=float, default=1.0)
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-cnn_channel', type=int, default=4)
    parser.add_argument('-cnn_kernel', type=int, default=5)
    parser.add_argument('-hid_dim', type=int, default=256)
    parser.add_argument('-pf_dim', type=int, default=512)
    parser.add_argument('-enc_layer', type=int, default=3)
    parser.add_argument('-dec_layer', type=int, default=3)
    parser.add_argument('-enc_head', type=int, default=4)
    parser.add_argument('-dec_head', type=int, default=4)
    parser.add_argument('-str_layer', help='string decoder layers(2)', type=int, default=2)
    parser.add_argument('-str_head', type=int, default=4)
    parser.add_argument('-weight_A', type=float, default=1.0)
    parser.add_argument('-weight_B', type=float, default=1.0)
    parser.add_argument('-weight_tab', type=float, default=1.0)
    parser.add_argument('-weight_tab_silent', help='CE weight of the silent class(0.2)', type=float, default=0.2)
    parser.add_argument('-weight_string_onset', type=float, default=1.0)
    parser.add_argument('-init_piano', help='piano state_dict (model/convert_piano_ckpt.py) to initialize from', default=None)
    parser.add_argument('-resume', help='checkpoint .pt to resume from', default=None)
    parser.add_argument('-amp', help='fp16 autocast', action='store_true')
    parser.add_argument('-no_sdpa', help='use the original attention (keeps attention maps, much more memory)', action='store_true')
    parser.add_argument('-num_workers', type=int, default=2)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f'** guitar hFT training ** torch {torch.__version__} device {device}')

    model = build_model(config, hid_dim=args.hid_dim, pf_dim=args.pf_dim,
                        cnn_channel=args.cnn_channel, cnn_kernel=args.cnn_kernel,
                        enc_layer=args.enc_layer, enc_head=args.enc_head,
                        dec_layer=args.dec_layer, dec_head=args.dec_head,
                        str_layer=args.str_layer, str_head=args.str_head,
                        dropout=args.dropout, device=device)
    model.apply(initialize_weights)
    if args.init_piano is not None:
        loaded, skipped = load_piano_init(model, torch.load(args.init_piano, map_location=device), config['midi']['note_min'])
        print(f' piano init: {len(loaded)} tensors loaded, {len(skipped)} skipped {skipped}')
    if not args.no_sdpa:
        use_sdpa(model)
    print(f' {count_parameters(model)} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    weight = {'A': args.weight_A, 'B': args.weight_B, 'tab': args.weight_tab,
              'tab_silent': args.weight_tab_silent, 'string_onset': args.weight_string_onset}

    d_dataset = args.d_dataset.rstrip('/')
    dataset_train = dataset_guitar.GuitarDataset(d_dataset+'/train', config, args.n_slice)
    dataset_valid = dataset_guitar.GuitarDataset(d_dataset+'/valid', config, args.n_slice)
    loader_kw = {'batch_size': args.batch, 'num_workers': args.num_workers, 'persistent_workers': args.num_workers > 0}
    dataloader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, **loader_kw)
    dataloader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False, **loader_kw)
    print(f' windows: train {len(dataset_train)} valid {len(dataset_valid)}')

    os.makedirs(args.d_out, exist_ok=True)
    with open(os.path.join(args.d_out, 'parameter.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'config': config, 'parameters': count_parameters(model)}, f, indent=4, sort_keys=True)

    epoch_start = 0
    best_loss_valid = float('inf')
    history = []
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch_start = ckpt['epoch'] + 1
        best_loss_valid = ckpt['best_loss_valid']
        history = ckpt['history']
        print(f' resumed from {args.resume} (epoch {ckpt["epoch"]})')

    for epoch in range(epoch_start, args.epoch):
        print(f'[epoch {epoch:03d}]')
        loss_train = train_guitar.run_epoch(model, dataloader_train, device, weight, optimizer=optimizer,
                                            clip=args.clip, accum=args.accum, amp=args.amp)
        loss_valid = train_guitar.run_epoch(model, dataloader_valid, device, weight, amp=args.amp)
        scheduler.step(loss_valid['total'])
        print(' train ' + ' '.join(f'{k} {v:.4f}' for k, v in loss_train.items()))
        print(' valid ' + ' '.join(f'{k} {v:.4f}' for k, v in loss_valid.items()))

        history.append({'epoch': epoch, 'train': loss_train, 'valid': loss_valid,
                        'lr': optimizer.param_groups[0]['lr'], 'datetime': datetime.datetime.now().isoformat()})
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'best_loss_valid': min(best_loss_valid, loss_valid['total']), 'history': history,
                'config': config, 'args': vars(args)}
        torch.save(ckpt, os.path.join(args.d_out, f'model_{epoch:03d}.pt'))
        if loss_valid['total'] < best_loss_valid:
            best_loss_valid = loss_valid['total']
            torch.save(ckpt, os.path.join(args.d_out, 'model_best.pt'))
            print(f' best valid {best_loss_valid:.4f}')
        with open(os.path.join(args.d_out, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=1)
