#! python

import time

import torch
import torch.nn.functional as F


def compute_loss(out, batch, weight):
    # pitch heads (A: after CAfreq, B: after SAtime), same terms as train.train; computed in fp32
    loss = {}
    for stage in ['A', 'B']:
        w = weight[stage]
        l = F.binary_cross_entropy(out['onset_'+stage].float(), batch['onset']) \
            + F.binary_cross_entropy(out['offset_'+stage].float(), batch['offset']) \
            + F.binary_cross_entropy(out['mpe_'+stage].float(), batch['mpe'])
        if 'velocity' in batch:
            v = out['velocity_'+stage].float()
            l = l + F.cross_entropy(v.reshape(-1, v.shape[-1]), batch['velocity'].reshape(-1))
        loss['pitch_'+stage] = w * l

    # string/fret: per-string 26-way softmax, silent class down-weighted (most frames are silent)
    tab_logits = out['tab_logits'].float()
    class_weight = torch.ones(tab_logits.shape[-1], device=tab_logits.device)
    class_weight[0] = weight['tab_silent']
    loss['tab'] = weight['tab'] * F.cross_entropy(tab_logits.reshape(-1, tab_logits.shape[-1]), batch['tab'].reshape(-1), weight=class_weight)
    loss['string_onset'] = weight['string_onset'] * F.binary_cross_entropy(out['string_onset'].float(), batch['tab_onset'])

    loss['total'] = sum(loss.values())
    return loss


def run_epoch(model, iterator, device, weight, optimizer=None, clip=1.0, accum=1, amp=False, log_every=50):
    # trains when an optimizer is given, otherwise evaluates; returns mean loss per term
    training = optimizer is not None
    model.train(training)
    sums = {}
    n = 0
    t0 = time.time()
    if training:
        optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        for i, batch in enumerate(iterator):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                out = model(batch['spec'], batch['tuning'])
            loss = compute_loss(out, batch, weight)

            if training:
                (loss['total'] / accum).backward()
                if (i + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                    optimizer.zero_grad()

            bs = batch['spec'].shape[0]
            n += bs
            for k, v in loss.items():
                sums[k] = sums.get(k, 0.0) + v.item() * bs
            if training and log_every > 0 and (i + 1) % log_every == 0:
                print(f'  step {i+1}/{len(iterator)} loss {sums["total"]/n:.4f} ({(time.time()-t0)/(i+1):.2f}s/step)', flush=True)
    return {k: v / max(n, 1) for k, v in sums.items()}
