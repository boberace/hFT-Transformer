#! python
# hFT-Transformer for guitar: the piano model's pitch outputs (onset/offset/mpe/velocity)
# plus a string decoder that assigns each sounding note to a string and fret.

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.model_spec2midi import Encoder_SPEC2MIDI, Decoder_SPEC2MIDI, EncoderLayer, DecoderLayer, MultiHeadAttentionLayer
except ImportError:
    from model_spec2midi import Encoder_SPEC2MIDI, Decoder_SPEC2MIDI, EncoderLayer, DecoderLayer, MultiHeadAttentionLayer

##
## Model
##
class Model_GUITAR(nn.Module):
    # attribute names match Model_SPEC2MIDI so piano weights load by key
    def __init__(self, encoder, decoder, string_decoder):
        super().__init__()
        self.encoder_spec2midi = encoder
        self.decoder_spec2midi = decoder
        self.string_decoder = string_decoder

    def forward(self, input_spec, tuning):
        #input_spec = [batch_size, n_bin, margin+n_frame+margin] (8, 256, 192)
        #tuning = [batch_size, n_string] open-string MIDI numbers, e.g. [40, 45, 50, 55, 59, 64]
        enc_vector = self.encoder_spec2midi(input_spec)
        #enc_vector = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)

        out = self.decoder_spec2midi(enc_vector)
        #out['note_state'] = [batch_size, n_frame, n_note, hid_dim] (8, 128, 53, 256)

        tab_logits, string_onset = self.string_decoder(enc_vector, out['note_state'], tuning)
        #tab_logits = [batch_size, n_frame, n_string, 1+n_fret+1] (8, 128, 6, 26) (0: silent, 1..25: fret 0..24)
        #string_onset = [batch_size, n_frame, n_string] (8, 128, 6)
        out['tab_logits'] = tab_logits
        out['string_onset'] = string_onset
        return out


##
## Note decoder (piano Decoder_SPEC2MIDI that also exposes the note states)
##
class Decoder_GUITAR(Decoder_SPEC2MIDI):
    def forward(self, enc_spec):
        batch_size = enc_spec.shape[0]
        enc_spec = enc_spec.reshape([batch_size*self.n_frame, self.n_bin, self.hid_dim])

        ## CAfreq freq(256)/note(53)
        pos_freq = torch.arange(0, self.n_note, device=enc_spec.device).unsqueeze(0).repeat(batch_size*self.n_frame, 1)
        midi_freq = self.pos_embedding_freq(pos_freq)
        midi_freq, attention_freq = self.layer_zero_freq(enc_spec, midi_freq)
        for layer_freq in self.layers_freq:
            midi_freq, attention_freq = layer_freq(enc_spec, midi_freq)
        if attention_freq is not None:  # None when attention runs through SDPA (see use_sdpa)
            dim = attention_freq.shape
            attention_freq = attention_freq.reshape([batch_size, self.n_frame, dim[1], dim[2], dim[3]])
        #midi_freq = [batch_size*n_frame, n_note, hid_dim] (8*128, 53, 256)

        ## output(freq)
        onset_A = self.sigmoid(self.fc_onset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        offset_A = self.sigmoid(self.fc_offset_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        mpe_A = self.sigmoid(self.fc_mpe_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note]))
        velocity_A = self.fc_velocity_freq(midi_freq).reshape([batch_size, self.n_frame, self.n_note, self.n_velocity])

        ## SAtime time(128)
        midi_time = midi_freq.reshape([batch_size, self.n_frame, self.n_note, self.hid_dim]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_note, self.n_frame, self.hid_dim])
        pos_time = torch.arange(0, self.n_frame, device=enc_spec.device).unsqueeze(0).repeat(batch_size*self.n_note, 1)
        midi_time = self.dropout((midi_time * self.scale_time) + self.pos_embedding_time(pos_time))
        for layer_time in self.layers_time:
            midi_time = layer_time(midi_time)
        #midi_time = [batch_size*n_note, n_frame, hid_dim] (8*53, 128, 256)

        ## output(time)
        onset_B = self.sigmoid(self.fc_onset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        offset_B = self.sigmoid(self.fc_offset_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        mpe_B = self.sigmoid(self.fc_mpe_time(midi_time).reshape([batch_size, self.n_note, self.n_frame]).permute(0, 2, 1).contiguous())
        velocity_B = self.fc_velocity_time(midi_time).reshape([batch_size, self.n_note, self.n_frame, self.n_velocity]).permute(0, 2, 1, 3).contiguous()

        note_state = midi_time.reshape([batch_size, self.n_note, self.n_frame, self.hid_dim]).permute(0, 2, 1, 3).contiguous()
        #note_state = [batch_size, n_frame, n_note, hid_dim] (8, 128, 53, 256)

        return {
            'onset_A': onset_A, 'offset_A': offset_A, 'mpe_A': mpe_A, 'velocity_A': velocity_A,
            'onset_B': onset_B, 'offset_B': offset_B, 'mpe_B': mpe_B, 'velocity_B': velocity_B,
            'attention': attention_freq, 'note_state': note_state,
        }


##
## String decoder
##
class StringDecoder(nn.Module):
    def __init__(self, n_frame, n_bin, n_note, note_min, n_string, n_fret, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.n_frame = n_frame
        self.n_bin = n_bin
        self.n_note = n_note
        self.note_min = note_min
        self.n_string = n_string
        self.n_pos = n_fret + 1  # fret 0 (open) .. n_fret
        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)
        self.scale = hid_dim ** 0.5

        # string queries: which string + what it is tuned to
        self.string_embedding = nn.Embedding(n_string, hid_dim)
        self.pitch_embedding = nn.Embedding(128, hid_dim)
        # memory = note states (53) + encoder bins (256); tell the two apart
        self.memory_type_embedding = nn.Embedding(2, hid_dim)

        # SA over strings + CA over memory, per frame
        self.layers_string = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        # SA over time, per string
        self.pos_embedding_time = nn.Embedding(n_frame, hid_dim)
        self.layers_time = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        # fret logit[s, f] = <W q_s, note_state[tuning_s + f]> + fret_bias[f]
        self.fc_fret_query = nn.Linear(hid_dim, hid_dim)
        self.fret_bias = nn.Parameter(torch.zeros(self.n_pos))
        self.fc_silent = nn.Linear(hid_dim, 1)
        self.fc_onset = nn.Linear(hid_dim, 1)

    def fret_note_map(self, tuning):
        # one-hot [batch_size, n_string, n_pos, n_note]: string s at fret f -> note index; all-zero if out of range
        # (a matmul instead of a data-dependent gather keeps shapes static for NPU export)
        frets = torch.arange(self.n_pos, device=tuning.device)
        notes = torch.arange(self.n_note, device=tuning.device)
        target = tuning.long().unsqueeze(-1) + frets - self.note_min
        #target = [batch_size, n_string, n_pos]
        return (target.unsqueeze(-1) == notes).float()

    def forward(self, enc_spec, note_state, tuning):
        #enc_spec = [batch_size, n_frame, n_bin, hid_dim] (8, 128, 256, 256)
        #note_state = [batch_size, n_frame, n_note, hid_dim] (8, 128, 53, 256)
        #tuning = [batch_size, n_string] (8, 6)
        batch_size = enc_spec.shape[0]
        device = enc_spec.device

        memory = torch.cat([note_state + self.memory_type_embedding.weight[0], enc_spec + self.memory_type_embedding.weight[1]], dim=2)
        memory = memory.reshape([batch_size*self.n_frame, self.n_note+self.n_bin, self.hid_dim])
        #memory = [batch_size*n_frame, n_note+n_bin, hid_dim] (8*128, 309, 256)

        string_idx = torch.arange(self.n_string, device=device)
        query = self.string_embedding(string_idx) + self.pitch_embedding(tuning.long())
        #query = [batch_size, n_string, hid_dim]
        query = query.unsqueeze(1).repeat(1, self.n_frame, 1, 1).reshape([batch_size*self.n_frame, self.n_string, self.hid_dim])

        for layer in self.layers_string:
            query, _ = layer(memory, query)
        #query = [batch_size*n_frame, n_string, hid_dim] (8*128, 6, 256)

        query = query.reshape([batch_size, self.n_frame, self.n_string, self.hid_dim]).permute(0, 2, 1, 3).contiguous().reshape([batch_size*self.n_string, self.n_frame, self.hid_dim])
        pos_time = torch.arange(0, self.n_frame, device=device).unsqueeze(0).repeat(batch_size*self.n_string, 1)
        query = self.dropout(query * self.scale + self.pos_embedding_time(pos_time))
        for layer in self.layers_time:
            query = layer(query)
        query = query.reshape([batch_size, self.n_string, self.n_frame, self.hid_dim]).permute(0, 2, 1, 3).contiguous()
        #query = [batch_size, n_frame, n_string, hid_dim] (8, 128, 6, 256)

        # tied fret logits
        fret_map = self.fret_note_map(tuning)
        #fret_map = [batch_size, n_string, n_pos, n_note] (8, 6, 25, 53)
        note_logits = torch.einsum('btsh,btnh->btsn', self.fc_fret_query(query), note_state) / self.scale
        #note_logits = [batch_size, n_frame, n_string, n_note] (8, 128, 6, 53)
        fret_logits = torch.einsum('btsn,bspn->btsp', note_logits, fret_map) + self.fret_bias
        fret_valid = fret_map.sum(-1).unsqueeze(1) > 0
        fret_logits = fret_logits.masked_fill(~fret_valid, -1e4)

        silent_logits = self.fc_silent(query)
        tab_logits = torch.cat([silent_logits, fret_logits], dim=-1)
        #tab_logits = [batch_size, n_frame, n_string, 1+n_pos] (8, 128, 6, 26)

        string_onset = torch.sigmoid(self.fc_onset(query).squeeze(-1))
        #string_onset = [batch_size, n_frame, n_string] (8, 128, 6)

        return tab_logits, string_onset


##
## builder / piano init
##
def build_model(config, hid_dim=256, pf_dim=512, cnn_channel=4, cnn_kernel=5,
                enc_layer=3, enc_head=4, dec_layer=3, dec_head=4, str_layer=2, str_head=4,
                dropout=0.1, device='cpu'):
    encoder = Encoder_SPEC2MIDI(config['input']['margin_b'],
                                config['input']['num_frame'],
                                config['feature']['n_bins'],
                                cnn_channel, cnn_kernel,
                                hid_dim, enc_layer, enc_head, pf_dim, dropout, device)
    decoder = Decoder_GUITAR(config['input']['num_frame'],
                             config['feature']['n_bins'],
                             config['midi']['num_note'],
                             config['midi']['num_velocity'],
                             hid_dim, dec_layer, dec_head, pf_dim, dropout, device)
    string_decoder = StringDecoder(config['input']['num_frame'],
                                   config['feature']['n_bins'],
                                   config['midi']['num_note'],
                                   config['midi']['note_min'],
                                   config['guitar']['n_string'],
                                   config['guitar']['n_fret'],
                                   hid_dim, str_layer, str_head, pf_dim, dropout, device)
    return Model_GUITAR(encoder, decoder, string_decoder).to(device)


def _mha_forward_sdpa(self, query, key, value):
    # same math as MultiHeadAttentionLayer.forward, but fused (no attention matrix kept); returns attention=None
    batch_size = query.shape[0]
    Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    x = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
    x = x.transpose(1, 2).reshape(batch_size, -1, self.hid_dim)
    return self.fc_o(x), None


def use_sdpa(model, flag=True):
    # switch every attention layer between the fused path and the original (which returns attention maps)
    for m in model.modules():
        if isinstance(m, MultiHeadAttentionLayer):
            if flag:
                m.forward = types.MethodType(_mha_forward_sdpa, m)
            elif 'forward' in vars(m):
                del m.forward
    return model


def load_piano_init(model, piano_state, note_min, piano_note_min=21):
    # copy encoder + note decoder weights from a Model_SPEC2MIDI state_dict;
    # note-indexed rows are sliced from the piano range to the guitar range
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in piano_state.items():
        if k not in own:
            skipped.append(k)
            continue
        if k == 'decoder_spec2midi.pos_embedding_freq.weight':
            s = note_min - piano_note_min
            v = v[s:s+own[k].shape[0]]
        if v.shape != own[k].shape:
            skipped.append(k)
            continue
        own[k] = v
        loaded.append(k)
    model.load_state_dict(own)
    return loaded, skipped
