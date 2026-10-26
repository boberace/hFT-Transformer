# hFT-Transformer

This repository contains the official PyTorch implementation of **"Automatic Piano Transcription with Hierarchical Frequency-Time Transformer"** presented in ISMIR2023 ([arXiv 2307.04305](https://arxiv.org/abs/2307.04305)).

## Guitar (`guitar` branch)

This branch adapts hFT-Transformer to 6-string guitar with up to 24 frets. The model keeps hFT's per-pitch outputs (onset, offset, frame, velocity) and adds the **string and fret** of every note. Development and training target Apple Silicon (MPS). The planned deployment is a vibration-sensing headstock clip (Syntiant V2S200D sensor plus RV1103 NPU) for live tab and chord identification. The piano code below is unchanged.

### Model
`model/model_guitar.py`
- **Pitch stages:** the hFT encoder and note decoder over MIDI 36–88 (53 notes). They can be initialized from the MAESTRO piano checkpoint.
- **String decoder:** 6 string queries per frame. Each is built from the string's identity and its open-string pitch.
  - Cross-attention over the note states and the encoder's frequency bins.
  - Self-attention across strings (chord shapes), then across time per string.
- **Outputs:**
  - `tab_logits` `[batch, frame, 6, 26]`: class 0 is silent, classes 1–25 are frets 0–24.
  - `string_onset` `[batch, frame, 6]`.
- **Tuning is an input.** The score for fret *f* on string *s* is computed against the note state at pitch `tuning[s] + f`. String/fret output therefore agrees with the detected pitch, and drop or open tunings need no retraining. Frets that fall outside the note range are masked.
- **Fused attention:** `use_sdpa(model)` switches every attention layer to `scaled_dot_product_attention`. This fits training in Apple GPU memory but returns no attention maps. Training enables it by default.

Configuration: `corpus/config_guitar.json` (same 16 kHz, 256-bin log-mel features as the piano model).

### Setup (macOS, Apple Silicon)
```
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-mac.txt
```
`requirements.txt` is the original Linux/CUDA environment for the piano model.

### Data
1) GuitarSet (acoustic; per-string annotations; CC BY 4.0). The script downloads it, splits by player (05 test, 04 valid, rest train) and builds the dataset under `corpus/GuitarSet/` (about 0.6 GB):
```
PYTHON=.venv/bin/python bash corpus/guitar/EXE-CORPUS-GUITARSET.sh
.venv/bin/python corpus/guitar/check_roundtrip_guitar.py -d corpus/GuitarSet -split test -n 3
```
The check decodes the labels back to notes and compares them with the GuitarSet annotations.

2) Synthetic data (`synth/`). Renders labeled acoustic, electric and nylon clips in which the same pitch sounds different on each string:
   - procedural tabs with random tunings and hand-span limits
   - a per-string waveguide with gauge-based inharmonicity
   - a body IR or pickup filter, depending on guitar type
   - a parametric headstock-sensor model (`--no_sensor` turns it off)

```
.venv/bin/python -m synth.render --out corpus/Synth --hours 10 --clip_sec 30 --workers 8
.venv/bin/python -m synth.check_same_pitch --pitch 64 --out corpus/Synth/check
```
See `synth/README.md` for the output format and the model's limits. Synthetic clips are not yet converted into a training dataset.

Dataset layout (`training/dataset_guitar.py`): `<dataset>/<split>/{feature,onset,offset,mpe,tab,tab_onset,tuning,idx}.npy` (plus optional `velocity.npy`). These are memory-mapped, so large sets need not fit in RAM.

### Training
Convert the pickled piano checkpoint to a plain state_dict once, then train:
```
.venv/bin/python model/convert_piano_ckpt.py
.venv/bin/python training/m_training_guitar.py -d_dataset corpus/GuitarSet/dataset -d_out checkpoint/GuitarSet \
    -init_piano checkpoint/MAESTRO-V3/model_016_003_state.pt
```
- **Device:** picks CUDA, then MPS, then CPU.
- **Defaults:** batch 4 with 2-step gradient accumulation. On a 32 GB Mac, batch 8 runs out of memory; batch 4 takes about 2.5 s/step.
- **Options:**
  - `-amp`: fp16 autocast
  - `-resume <model_XXX.pt>`: continue a run
  - `-hid_dim 128 -pf_dim 256`: a smaller model for quick experiments
  - `-n_slice`: training window stride (larger is faster)
- **Checkpoints:** `model_XXX.pt` and `model_best.pt`, each holding the state_dict, optimizer, config and loss history.

### Status
- Done:
  - model and Mac training loop
  - GuitarSet pipeline
  - synthesizer (one short GuitarSet epoch verified end to end)
- Next:
  - guitar inference and tab evaluation (tab precision/recall/F1 and disambiguation rate)
  - a synthetic-data dataset builder
  - pitch-shift/tuning augmentation
  - EGDB and IDMT-SMT-Guitar
  - a smaller streaming model for the RV1103

## Piano (original)

### Development Environment
- OS
  + Ubuntu 18.04
- memory
  + 32GB
- GPU
  + corpus generation, evaluation
    - NVIDIA GeForce RTX 2080 Ti
  + training
    - NVIDIA A100
- Python
  + 3.6.9
- Required Python libraries
  + [requirements.txt](requirements.txt)

### Usage
1) corpus generation (MAESTRO-V3)
```
$ ./corpus/EXE-CORPUS-MAESTRO.sh
```
2) training
```
$ ./training/EXE-TRAINING-MAESTRO.sh
```
3) evaluation

If you want to avoid training models from scratch, you can download and put the model under the `checkpoint/MAESTRO-V3` directory.

`model_016_003.pkl` is the model for MAESTRO.

```
$ wget https://github.com/sony/hFT-Transformer/releases/download/ismir2023/checkpoint.zip
$ unzip checkpoint.zip
$ ./evaluation/EXE-EVALUATION-MAESTRO.sh model_016_003.pkl test
```

If you want to evaluate the trained model using the validation set, you can change the second argument as below.
```
$ ./evaluation/EXE-EVALUATION-MAESTRO.sh model_016_003.pkl valid
```

## Citation
Keisuke Toyama, Taketo Akama, Yukara Ikemiya, Yuhta Takida, Wei-Hsiang Liao, and Yuki Mitsufuji, "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer," in Proceedings of the 24th International Society for Music Information Retrieval Conference, 2023.
```
@inproceedings{toyama2023,
    author={Keisuke Toyama and Taketo Akama and Yukara Ikemiya and Yuhta Takida and Wei-Hsiang Liao and Yuki Mitsufuji},
    title={Automatic Piano Transcription with Hierarchical Frequency-Time Transformer},
    booktitle={Proceedings of the 24th International Society for Music Information Retrieval Conference},
    year={2023}
}
```

## Contact
- Keisuke Toyama (keisuke.toyama@sony.com)

## Reference
- PyTorch Seq2Seq (https://github.com/bentrevett/pytorch-seq2seq)
