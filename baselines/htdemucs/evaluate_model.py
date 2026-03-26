#!/usr/bin/env python3
# evaluate_model.py - Evaluates a separation model on a dataset, calculating various metrics.

import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd

from windowed_audio_datasets import FolderTripletDataset
from torch.utils.data import DataLoader
from separate_batch import load_model, save_audio, process_audio # Replaced copied functions with imports

import torchmetrics

#######
# Based on the code from https://github.com/facebookresearch/demucs
#######

# --- Copied/Adapted Helper Functions ---

def save_spectrogram_eval(tensor, filepath, sample_rate=8000, n_fft=512, hop_length=128, title=None):
    """Generate and save a spectrogram visualization. Adapted from train_with_recon.py"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if tensor.device != torch.device('cpu'): 
        tensor = tensor.cpu()
    
    # Remove batch dim if present, assume (C,T) or (T)
    if tensor.ndim == 3 and tensor.shape[0] == 1: # (B,C,T) -> (C,T)
        tensor = tensor.squeeze(0)
    
    if tensor.ndim == 2 and tensor.shape[0] > 1: # Multi-channel (C,T)
        tensor = tensor.mean(dim=0) # Average channels to get (T)
    elif tensor.ndim == 2 and tensor.shape[0] == 1: # Mono (1,T)
        tensor = tensor.squeeze(0) # (T)
    # If tensor is 1D (T), it's fine

    if tensor.ndim != 1:
        print(f"Warning: Spectrogram input tensor has unexpected shape {tensor.shape} for {filepath}. Taking mean over channel if applicable.")
        if tensor.ndim > 1: tensor = tensor.mean(dim=0) # Try to make it 1D
        if tensor.ndim != 1: 
            print(f"Skipping spectrogram for {filepath} due to unhandled shape.")
            return

    spec_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        center=True,
        pad_mode="reflect"
    )
    
    spec = spec_transform(tensor) 
    spec_db = T.AmplitudeToDB(top_db=80)(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Frames')
    plt.ylabel('Frequency Bins')
    if title:
        plt.title(title)
    plt.tight_layout()
    try:
        plt.savefig(filepath)
        print(f"Saved spectrogram: {filepath}")
    except Exception as e:
        print(f"Error saving spectrogram {filepath}: {e}")
    plt.close()

def run_evaluation(args):
    """Main function to run the evaluation."""
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and args.gpu_index is not None:
            device = torch.device(f"cuda:{args.gpu_index}")
            print(f"Multiple GPUs detected. Using GPU {args.gpu_index}.")
        else:
            device = torch.device("cuda")
            print("CUDA available. Using GPU 0 (default) or first available.")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    print(f"Using device: {device}")

    # --- Load Model ---
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, device) 
    
    # Determine sample rate
    sr_from_checkpoint = model_args.get('sr', None) 
    if sr_from_checkpoint is not None and sr_from_checkpoint != args.sr:
        print(f"Warning: SR from model args ({sr_from_checkpoint}) differs from specified SR ({args.sr}). Using specified SR: {args.sr}")
    sample_rate = args.sr
    
    # --- Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    # --- Dataset ---
    dataset = FolderTripletDataset(
        root=Path(args.dataset_dir),
        split=args.split,
        segment_length_sec=args.eval_clip_duration,
        hop_length_sec=args.eval_clip_duration, # Process distinct clips
        sr=sample_rate 
    )
    if not dataset:
        print(f"No audio files found in dataset: {args.dataset_dir} for split {args.split} with current settings. Exiting.")
        return
    
    actual_dataset_size = len(dataset)
    if actual_dataset_size == 0:
        print(f"No audio files found in dataset: {args.dataset_dir} for split {args.split} after filtering. Exiting.")
        return

    indices = list(range(actual_dataset_size))
    if args.max_files is not None and args.max_files > 0 and args.max_files < actual_dataset_size:
        indices = indices[:args.max_files]
        print(f"Processing a subset of {args.max_files} files out of {actual_dataset_size} available.")
    elif args.max_files is not None and args.max_files >= actual_dataset_size:
        print(f"max_files ({args.max_files}) is >= total files ({actual_dataset_size}). Processing all {actual_dataset_size} files.")
    else: # max_files is None or 0
        print(f"Processing all {actual_dataset_size} files.")

    subset_sampler = torch.utils.data.SubsetRandomSampler(indices)

    dataloader = DataLoader(
        dataset,
        batch_size=1, # Process one file at a time for detailed eval
        shuffle=False, # SubsetRandomSampler handles selection, shuffle=False for dataloader itself
        sampler=subset_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    if len(dataloader) == 0:
        print("Dataloader is empty. This might be due to max_files being 0 or very small, or no files in dataset. Exiting.")
        return

    # --- Metrics Initialization ---
    sisnr_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    
    pesq_mode = 'wb' if sample_rate == 16000 else 'nb'
    if sample_rate not in [8000, 16000]:
        print(f"Warning: PESQ supports 8kHz (nb) or 16kHz (wb). Current SR is {sample_rate}. PESQ might error or be inaccurate.")
        pesq_metric = None
    else:
        pesq_metric = torchmetrics.audio.PerceptualEvaluationSpeechQuality(sample_rate, pesq_mode).to(device)
        
    stoi_metric = torchmetrics.audio.ShortTimeObjectiveIntelligibility(sample_rate, extended=False).to(device)

    all_metrics_data = []
    processing_times = []

    num_files_to_process = len(indices)
    print(f"Starting evaluation on {num_files_to_process} files...")

    for i, (mixture_clip, target_speech_clip, target_music_clip) in enumerate(dataloader):
        original_file_dataset_index = indices[i] 
        file_id = f"sample_{original_file_dataset_index:04d}"
        
        loop_start_time = time.time()
        print(f"\n--- Processing File {i+1}/{num_files_to_process} (Dataset Index: {original_file_dataset_index}, ID: {file_id}) ---")

        mixture_clip = mixture_clip.squeeze(0).cpu() 
        target_speech_clip = target_speech_clip.squeeze(0).cpu()
        target_music_clip = target_music_clip.squeeze(0).cpu()

        sample_output_dir = output_dir / file_id
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        if args.save_audio_files:
            save_audio(mixture_clip, sample_output_dir / "gt_mixture.wav", sample_rate)
            save_audio(target_speech_clip, sample_output_dir / "gt_speech.wav", sample_rate)
            save_audio(target_music_clip, sample_output_dir / "gt_music.wav", sample_rate)

        if args.save_spectrograms:
            save_spectrogram_eval(mixture_clip, sample_output_dir / "gt_mixture_spec.png", sample_rate, args.n_fft, args.hop_length, title=f"{file_id} - GT Mixture")
            save_spectrogram_eval(target_speech_clip, sample_output_dir / "gt_speech_spec.png", sample_rate, args.n_fft, args.hop_length, title=f"{file_id} - GT Speech")
            save_spectrogram_eval(target_music_clip, sample_output_dir / "gt_music_spec.png", sample_rate, args.n_fft, args.hop_length, title=f"{file_id} - GT Music")

        est_speech, est_music = process_audio(
            mixture_clip, # (C, T) on CPU
            model,
            segment=args.processing_segment_duration, # segment is segment length in seconds
            sample_rate=sample_rate,
            device=device
        )
        # est_speech, est_music are returned on CPU, shape (C,T) or (1,C,T)

        if args.save_audio_files:
            save_audio(est_speech, sample_output_dir / "est_speech.wav", sample_rate)
            save_audio(est_music, sample_output_dir / "est_music.wav", sample_rate)
        
        if args.save_spectrograms:
            save_spectrogram_eval(est_speech, sample_output_dir / "est_speech_spec.png", sample_rate, args.n_fft, args.hop_length, title=f"{file_id} - Estimated Speech")
            save_spectrogram_eval(est_music, sample_output_dir / "est_music_spec.png", sample_rate, args.n_fft, args.hop_length, title=f"{file_id} - Estimated Music")

        # Ensure est_speech and est_music are (C,T) by removing leading batch dim if present
        _est_speech_no_batch = est_speech.squeeze(0) if est_speech.ndim == 3 and est_speech.shape[0] == 1 else est_speech
        _est_music_no_batch = est_music.squeeze(0) if est_music.ndim == 3 and est_music.shape[0] == 1 else est_music

        # Prepare tensors for metrics (send to device)
        # Squeeze channel dim if it's 1, or mean if C > 1, to get (Time)
        mixture_for_metric = (mixture_clip.mean(dim=0) if mixture_clip.shape[0] > 1 else mixture_clip.squeeze(0)).to(device)
        target_speech_for_metric = (target_speech_clip.mean(dim=0) if target_speech_clip.shape[0] > 1 else target_speech_clip.squeeze(0)).to(device)
        target_music_for_metric = (target_music_clip.mean(dim=0) if target_music_clip.shape[0] > 1 else target_music_clip.squeeze(0)).to(device)
        
        est_speech_for_metric = (_est_speech_no_batch.mean(dim=0) if _est_speech_no_batch.shape[0] > 1 else _est_speech_no_batch.squeeze(0)).to(device)
        est_music_for_metric = (_est_music_no_batch.mean(dim=0) if _est_music_no_batch.shape[0] > 1 else _est_music_no_batch.squeeze(0)).to(device)


        current_metrics = {'file_id': file_id}
        try:
            current_metrics['s_sisnr_speech'] = sisnr_metric(est_speech_for_metric, target_speech_for_metric).item()
            current_metrics['s_sisnr_music'] = sisnr_metric(est_music_for_metric, target_music_for_metric).item()
            est_stacked = torch.stack([est_speech_for_metric, est_music_for_metric], dim=0).unsqueeze(0) 
            target_stacked = torch.stack([target_speech_for_metric, target_music_for_metric], dim=0).unsqueeze(0)
            current_metrics['s_sisnr_overall'] = sisnr_metric(est_stacked, target_stacked).item()
            sisnr_input_speech = sisnr_metric(mixture_for_metric, target_speech_for_metric).item()
            current_metrics['s_sisnri_speech'] = current_metrics['s_sisnr_speech'] - sisnr_input_speech
            
            if pesq_metric:
                current_metrics['pesq'] = pesq_metric(est_speech_for_metric, target_speech_for_metric).item()
            else:
                current_metrics['pesq'] = np.nan
                
            current_metrics['stoi'] = stoi_metric(est_speech_for_metric, target_speech_for_metric).item()

        except Exception as e:
            print(f"Error calculating metrics for {file_id}: {e}")
            for key in ['s_sisnr_speech', 's_sisnr_music', 's_sisnr_overall', 's_sisnri_speech', 'pesq', 'stoi']:
                current_metrics.setdefault(key, np.nan)
        
        all_metrics_data.append(current_metrics)
        
        print(f"Metrics for {file_id}:")
        for k, v in current_metrics.items():
            if k != 'file_id': print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
        
        loop_end_time = time.time()
        processing_times.append(loop_end_time - loop_start_time)
        print(f"Time for {file_id}: {processing_times[-1]:.2f}s")

    if not all_metrics_data:
        print("No files were processed or no metrics collected.")
        return

    metrics_df = pd.DataFrame(all_metrics_data)
    mean_metrics = metrics_df.drop(columns=['file_id'], errors='ignore').mean().add_prefix("avg_") # errors='ignore' if file_id not in all
    
    print("\n" + "="*30 + " AVERAGE METRICS " + "="*30)
    if not mean_metrics.empty:
        print(mean_metrics.to_string())
    else:
        print("No metrics available to average.")
    print("="*77)
    
    metrics_csv_path = output_dir / "evaluation_metrics_per_file.csv"
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.4f')
    print(f"Per-file metrics saved to: {metrics_csv_path}")
    
    if not mean_metrics.empty:
        avg_metrics_path = output_dir / "evaluation_metrics_average.csv"
        mean_metrics.to_frame(name="average_value").to_csv(avg_metrics_path, float_format='%.4f')
        print(f"Average metrics saved to: {avg_metrics_path}")

    total_time = sum(processing_times)
    avg_time_per_file = total_time / len(processing_times) if processing_times else 0
    print(f"Total evaluation time: {total_time:.2f}s ({len(processing_times)} files)")
    print(f"Average time per file: {avg_time_per_file:.2f}s")
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a speech/music separation model.")
    
    parser.add_argument('dataset_dir', type=str, help="Root directory of the dataset (e.g., where train/val/test folders are).")
    parser.add_argument('--output_dir', type=str, default="evaluation_results", help="Directory to save evaluation outputs (audio, spectrograms, metrics).")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint (.pth file).")
    parser.add_argument('--split', type=str, default="test", choices=['train', 'val', 'test', 'valid', 'validation'], help="Dataset split to evaluate (default: test).")
    
    parser.add_argument('--eval_clip_duration', type=float, default=10.0, help="Duration of audio clips (in seconds) to load from the dataset for evaluation (default: 10.0s).")
    parser.add_argument('--processing_segment_duration', type=float, default=1.0, help="Duration of segments (in seconds) for processing long clips. If 0 or None, process whole clip at once (default: 1.0s).")
    
    parser.add_argument('--sr', type=int, default=8000, help="Sample rate to use for loading and processing (default: 8000). Ensure this matches model training if not specified in checkpoint.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader (default: 0).")
    parser.add_argument('--max_files', type=int, default=None, help="Maximum number of files to process from the dataset (default: all). Set to 0 for all files.")
    
    parser.add_argument('--device', type=str, default=None, help="Device to use (e.g., 'cuda:0', 'cpu'). Autodetects if None.")
    parser.add_argument('--gpu_index', type=int, default=0, help="GPU index to use if multiple GPUs are available and device is not explicitly set to a specific cuda device (default: 0).")

    parser.add_argument('--n_fft', type=int, default=512, help="FFT size for spectrograms (default: 512).")
    parser.add_argument('--hop_length', type=int, default=128, help="Hop length for spectrograms (default: 128).")
    parser.add_argument('--save_spectrograms', action='store_true', help="Save spectrograms of GT and estimated audio (default: False).")
    parser.add_argument('--save_audio_files', action='store_true', help="Save GT and estimated audio files (default: False).")

    script_args = parser.parse_args()

    if script_args.processing_segment_duration is not None and script_args.processing_segment_duration <= 0:
        script_args.processing_segment_duration = None # process_audio expects None to process whole
        print("Processing segment duration set to <= 0, will process clips whole.")
    
    if script_args.max_files is not None and script_args.max_files <= 0:
        script_args.max_files = None # None means all files
        print("max_files set to <=0, interpreting as process all files.")


    run_evaluation(script_args) 