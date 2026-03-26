# TACIT: TCN And Conformer Integrated Two-stream model for Background Music Removal

This repository contains code for the paper titled "TACIT: TCN And Conformer Integrated
Two-stream model for Background Music Removal" (Accepted for publication, ICCAI 2026). 
```
@inproceedings{mohammedTACIT2026,
  title = {TACIT: TCN And Conformer Integrated Two-stream Model for Background Music Removal},
  author = {Mohammed, Abdus Saboor Gaffari and Ghebretatios, Solomon Iyassu and Alhosani, Latifa and Zualkernan, Imran},
  year = {2026},
  address = {Okinawa, Japan},
  booktitle = {Proceedings of the 2026 12th International Conference on Computing and Artificial Intelligence},
}
```
The main model, **TACIT** (**T**CN **A**nd **C**onformer **I**ntegrated **T**wo-stream), separates an input mixture into:

- a `speech` target containing speech plus environmental/background noise
- a `music` target containing the background music component

The repository also includes:

- synthetic dataset generation scripts
- training and evaluation code for the proposed TACIT model
- Conv-TasNet and HTDemucs baselines
- utilities for exporting separated audio and evaluation artifacts

## 1. Repository Structure

```text
.
├── datagen/
│   ├── generate_dataset.py
│   └── generate_dataset_segmented.py
├── our/
│   ├── model.py
│   ├── train.py
│   ├── train_recon.py
│   ├── train_with_recon.py
│   ├── evaluate_model.py
│   ├── separate_batch.py
│   ├── audio_datasets.py
│   └── windowed_audio_datasets.py
├── baselines/
│   ├── conv-tasnet/
│   │   ├── Conv_TasNet.py
│   │   ├── train.py
│   │   ├── separate_batch.py
│   │   └── audio_datasets.py
│   └── htdemucs/
│       ├── train.py
│       ├── evaluate_model.py
│       ├── separate_batch.py
│       ├── audio_datasets.py
│       ├── windowed_audio_datasets.py
│       ├── requirements.txt
│       └── environment-*.yml
└── environment.yml
```

## 2. Task Definition

Each synthesized sample is built from three sources:

- `speech`: speech content from VCTK
- `noise`: environmental sounds from ESC-50
- `music`: music tracks from MUSDB18

The data generation scripts first combine speech and environmental noise, then mix the result with music. In the generated dataset:

- `mix.wav` or `mixture.wav`: input mixture
- `speech.wav`: speech plus environmental noise target
- `music.wav`: background music target
- `speech_isolated.wav`: clean speech only
- `noise.wav`: isolated environmental noise

Only `mix`/`mixture`, `speech`, and `music` are used by the training and evaluation scripts.

## 3. Environment Setup

### 3.1 Main TACIT Environment

The root `environment.yml` is intended for the main TACIT and Conv-TasNet pipelines.

```bash
conda env create -f environment.yml
conda activate music_rem
python -m pip install pandas librosa
```

Notes:

- The dataset generation scripts require `pandas` and `librosa`.
- The training and evaluation code requires a working PyTorch, TorchAudio, and TorchVision installation.
- If the pinned `torch==2.7.0+cu126` wheels in `environment.yml` do not resolve on your machine, install a PyTorch build that matches your local CUDA or CPU setup first, then install the remaining Python packages.

### 3.2 HTDemucs Baseline Environment

The HTDemucs baseline imports `demucs.htdemucs.HTDemucs` and has separate environment files under `baselines/htdemucs/`.

Use one of:

- `baselines/htdemucs/environment-cuda.yml`
- `baselines/htdemucs/environment-cpu.yml`
- `baselines/htdemucs/environment_demucs.yml`

or install from:

```bash
python -m pip install -r baselines/htdemucs/requirements.txt
```

Additional note:

- The HTDemucs scripts require the `demucs` Python package itself to be importable in the active environment, because the baseline code calls `from demucs.htdemucs import HTDemucs`.

## 4. Datasets

The experiments are built from the following public datasets.

### Speech Dataset

- CSTR VCTK Corpus: <https://datashare.ed.ac.uk/handle/10283/3443>

### Music Dataset

- MUSDB18: <https://sigsep.github.io/datasets/musdb.html>

### Environmental Sounds Dataset

- ESC-50: <https://github.com/karolpiczak/ESC-50>

Practical path examples after extraction:

- `VCTK`: point `--speech_path` to a directory containing VCTK waveform files, e.g. `wav48_silence_trimmed/`
- `MUSDB18`: point `--music_path` to the extracted MUSDB18 audio root
- `ESC-50`: point `--noise_path` to the `audio/` directory

The data generators recursively scan for `.wav` and `.mp3` files.

## 5. Dataset Preparation

Two dataset generation modes are provided.

### 5.1 Full-Track Dataset Generation

Recommended for the main TACIT pipeline in `our/train.py` and `our/train_with_recon.py`, which use windowed loading from longer waveforms.

```bash
python datagen/generate_dataset.py \
  --music_path /path/to/musdb18 \
  --speech_path /path/to/vctk \
  --noise_path /path/to/esc50/audio \
  --output_base_path /path/to/generated_dataset \
  --sample_rate 16000 \
  --speech_noise_snr 5.0 \
  --random_seed 42
```

Implementation details:

- default split ratio is `0.70 / 0.15 / 0.15` for `train / validation / test`
- one full-length sample directory is created per selected music file
- speech and noise streams are stitched to match the music duration
- speech-noise SNR defaults to `5 dB`
- music is mixed at a random SNR sampled internally from `[-5 dB, 5 dB]`

### 5.2 Pre-Segmented Dataset Generation

Useful when you want fixed-length clips written to disk directly.

```bash
python datagen/generate_dataset_segmented.py \
  --music_path /path/to/musdb18 \
  --speech_path /path/to/vctk \
  --noise_path /path/to/esc50/audio \
  --output_base_path /path/to/generated_dataset_segmented \
  --sample_rate 16000 \
  --speech_noise_snr 5.0 \
  --random_seed 42
```

Default split-specific segment settings:

- train: `1.0 s` segments with `0.5 s` stride
- validation: `1.5 s` segments with `0.75 s` stride
- test: `2.0 s` segments with `1.0 s` stride

### 5.3 Generated Dataset Layout

Both generators create a split-wise directory structure:

```text
generated_dataset/
├── train/
│   ├── 00000/
│   │   ├── mix.wav
│   │   ├── speech.wav
│   │   ├── music.wav
│   │   ├── speech_isolated.wav
│   │   └── noise.wav
│   ├── 00001/
│   └── metadata.csv
├── validation/
│   └── ...
└── test/
    └── ...
```

Notes:

- Training code accepts either `mix.wav` or `mixture.wav`.
- Validation directories may be named `validation`, `val`, or `valid`; the training scripts already handle all three.
- Audio is resampled on load to the sample rate specified by `--sr`.

## 6. Recommended Training Pipeline

The most complete TACIT training script is:

- `our/train_with_recon.py`

This script supports a weighted objective combining:

- speech SI-SNR loss
- optional music SI-SNR loss
- mixture reconstruction loss
- mel-spectrogram loss
- optional speech reconstruction L1 loss
- optional music reconstruction L1 loss

### 6.1 Recommended TACIT Training Command

Use a single visible GPU unless you have already adapted the device-selection logic in the code.

```bash
CUDA_VISIBLE_DEVICES=0 python our/train_with_recon.py \
  --root_dir /path/to/generated_dataset \
  --save_dir runs/tacit_main \
  --sr 8000 \
  --segment 1.0 \
  --hop_length_sec 0.2 \
  --batch_size 2 \
  --epochs 100 \
  --lr 1e-4 \
  --speech_sisnr_loss_weight 1.0 \
  --music_sisnr_loss_weight 0.0 \
  --mix_recon_weight 0.1 \
  --mel_weight 0.05 \
  --mel_weight_decay \
  --mel_weight_initial 0.1 \
  --mel_weight_decay_epochs 10
```

Important notes:

- The code requires `--sr` to be either `8000` or `16000` because PESQ is computed during training/evaluation.
- The main training scripts default to `8000 Hz`, even if the generated dataset was written at `16000 Hz`; resampling is handled in the dataset loader.
- `our/train_with_recon.py` and `our/train.py` use `windowed_audio_datasets.py`, so they are best paired with the full-track generator.
- `our/train_recon.py` uses `audio_datasets.py` and is more natural for pre-segmented data or fixed-length clip training.

### 6.2 Other TACIT Training Variants

- `our/train.py`: SI-SNR-based training with mixture reconstruction and optional mel loss
- `our/train_recon.py`: L1 reconstruction-oriented variant

## 7. Evaluation and Audio Export

### 7.1 Quantitative Evaluation

`our/evaluate_model.py` evaluates a checkpoint and writes per-file and average metrics.

```bash
python our/evaluate_model.py /path/to/generated_dataset \
  --checkpoint runs/tacit_main/best_model.pth \
  --split test \
  --output_dir runs/tacit_eval \
  --sr 8000 \
  --device cuda:0 \
  --eval_clip_duration 10.0 \
  --processing_segment_duration 1.0 \
  --save_audio_files \
  --save_spectrograms
```

Reported metrics include:

- SI-SNR
- speech SI-SNR improvement
- PESQ
- STOI

Saved summary files:

- `evaluation_metrics_per_file.csv`
- `evaluation_metrics_average.csv`

### 7.2 Batch Audio Separation

`our/separate_batch.py` is a convenience script for exporting estimated stems.

```bash
python our/separate_batch.py /path/to/generated_dataset \
  --checkpoint runs/tacit_main/best_model.pth \
  --split test \
  --output_dir runs/tacit_separated \
  --sr 8000 \
  --segment 1.0
```

Each processed sample is written as:

- `gt_mixture.wav`
- `gt_speech.wav`
- `gt_music.wav`
- `est_speech.wav`
- `est_music.wav`

## 8. Baselines

### 8.1 Conv-TasNet

Training:

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/conv-tasnet/train.py \
  --root_dir /path/to/generated_dataset_segmented \
  --save_dir runs/conv_tasnet \
  --sr 8000 \
  --segment 1.0 \
  --batch_size 2 \
  --epochs 100
```

Batch separation:

```bash
python baselines/conv-tasnet/separate_batch.py /path/to/generated_dataset_segmented \
  --checkpoint runs/conv_tasnet/best_model.pth \
  --split test \
  --output_dir runs/conv_tasnet_separated \
  --sr 8000
```

### 8.2 HTDemucs

HTDemucs requires the additional Demucs dependency stack described in Section 3.2.

If you see `ModuleNotFoundError: demucs`, the active environment does not yet expose the Demucs package.

Training:

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/htdemucs/train.py \
  --root_dir /path/to/generated_dataset \
  --save_dir runs/htdemucs \
  --sr 8000 \
  --segment 3.0 \
  --hop_length_sec 1.5 \
  --batch_size 4 \
  --epochs 100
```

Evaluation:

```bash
python baselines/htdemucs/evaluate_model.py /path/to/generated_dataset \
  --checkpoint runs/htdemucs/best_model.pth \
  --split test \
  --output_dir runs/htdemucs_eval \
  --sr 8000 \
  --device cuda:0
```

Batch separation:

```bash
python baselines/htdemucs/separate_batch.py /path/to/generated_dataset \
  --checkpoint runs/htdemucs/best_model.pth \
  --split test \
  --output_dir runs/htdemucs_separated \
  --sr 8000
```

## 9. Training Outputs

Training scripts create a run directory such as `runs/tacit_main/` containing:

- `checkpoint_epoch_*.pth`: checkpoint saved every epoch
- `best_model.pth`: best checkpoint according to validation loss
- `tensorboard_logs/`: TensorBoard scalars and audio summaries
- `audio_samples/`: ground-truth and epoch-wise sample reconstructions
- `spectrograms/`: ground-truth and epoch-wise spectrogram images
- `loss_plot*.png`: loss curves
- `si_snr_plot*.png`: SI-SNR curves

The TACIT and baseline training scripts automatically resume from the latest `checkpoint_epoch_*.pth` found inside `--save_dir`.

## 10. Reproducibility Notes

- Set `--random_seed 42` during dataset generation to keep the train/validation/test split deterministic.
- The data synthesis scripts split the three source corpora independently using the same ratio and random seed.
- Stereo audio is converted to mono at load time.
- Quantitative evaluation assumes `--sr` is `8000` or `16000`.
- The generated `validation/` split name is already compatible with the training code.
- On multi-GPU systems, several scripts default to selecting `cuda:2` when more than one GPU is visible. Using `CUDA_VISIBLE_DEVICES=0` is the safest way to run the code without editing the source.

## 11. Suggested End-to-End Reproduction Order

1. Download and extract VCTK, MUSDB18, and ESC-50.
2. Generate the full-track synthetic dataset with `datagen/generate_dataset.py`.
3. Train TACIT with `our/train_with_recon.py`.
4. Evaluate `best_model.pth` using `our/evaluate_model.py`.
5. Train and evaluate the baselines for comparison.


## 12. References

Conv-TasNet - [Link 1](https://github.com/JusperLee/Conv-TasNet) | [Link 2](https://github.com/naplab/Conv-TasNet)

HT Demucs - [Link](https://github.com/facebookresearch/demucs)
