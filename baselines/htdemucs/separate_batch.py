#!/usr/bin/env python3
# separate_batch.py - Separates speech and music from all mixture files in a TripletDataset

import argparse
import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
import time

# Import model and other necessary components
from demucs.htdemucs import HTDemucs
from audio_datasets import FolderTripletDataset
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device):
    """Load HTDemucs model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model_args from checkpoint, default to an empty dict if not found
    model_args = checkpoint.get('args', checkpoint.get('model_args', {}))
    if not isinstance(model_args, dict): # Ensure model_args is a dictionary
        print(f"Warning: Loaded model_args is not a dict: {type(model_args)}. Using empty dict instead.")
        model_args = {}

    # Initialize HTDemucs model
    # Use samplerate from model_args if available, otherwise default or make it a parameter
    # For now, let's keep the hardcoded samplerate but acknowledge it might need changing
    # based on how model_args are structured and used later.
    # sr = model_args.get('sr', model_args.get('samplerate', 8000)) # Example if sr is in args
    model = HTDemucs(
        sources=["speech", "music"],
        audio_channels=1, # model_args.get('audio_channels', 1)
        samplerate=8000   # sr
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model, model_args


def save_audio(tensor, filepath, sample_rate=8000):
    """Save audio tensor to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if tensor.device != torch.device('cpu'): 
        tensor = tensor.cpu()
    if tensor.ndim == 4: 
        tensor = tensor[0, 0]  # (B, N, C, T) -> (C, T)
    elif tensor.ndim == 3: 
        tensor = tensor[0]  # (B, C, T) or (N, C, T) -> (C, T)
    elif tensor.ndim == 1: 
        tensor = tensor.unsqueeze(0)  # (T) -> (C=1, T)
    
    if tensor.ndim != 2 or tensor.shape[0] > 16:  # Basic check for sensible shape
        print(f"Warning: Unexpected tensor shape for audio saving: {tensor.shape}. Forcing to mono.")
        tensor = tensor.mean(dim=0, keepdim=True)  # Average channels if multi-channel
        tensor = tensor.reshape(1, -1)  # Force to (1, Time)
    
    max_val = torch.max(torch.abs(tensor))
    if max_val > 0.999: 
        tensor = tensor / (max_val * 1.05)
    
    torchaudio.save(filepath, tensor, sample_rate)
    print(f"Saved audio to {filepath}")


def process_audio(mixture, model, segment=None, sample_rate=8000, device=torch.device('cpu')):
    """Process audio mixture through the HTDemucs model"""
    duration = mixture.shape[1] / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")
    
    with torch.no_grad():
        if segment is not None and duration > segment:
            print(f"Processing in segments of {segment} seconds")
            # Process in segments
            segment_samples = int(segment * sample_rate)
            num_segments = int(np.ceil(mixture.shape[1] / segment_samples))
            
            # Initialize output tensors
            speech_output = torch.zeros_like(mixture)
            music_output = torch.zeros_like(mixture)
            
            for i in range(num_segments):
                start_idx = i * segment_samples
                end_idx = min((i + 1) * segment_samples, mixture.shape[1])
                
                segment_waveform = mixture[:, start_idx:end_idx]
                
                # Pad if needed to reach segment size
                if segment_waveform.shape[1] < segment_samples:
                    padding = segment_samples - segment_waveform.shape[1]
                    segment_waveform = torch.nn.functional.pad(segment_waveform, (0, padding))
                
                # Ensure batch dimension and move to device
                segment_input = segment_waveform.unsqueeze(0).to(device)
                
                # Process through model
                outputs = model(segment_input)
                
                # Extract speech and music estimates
                speech_est = outputs[0, 0].cpu()  # First source is speech
                music_est = outputs[0, 1].cpu()   # Second source is music
                
                # Remove padding if added
                actual_length = min(segment_samples, end_idx - start_idx)
                speech_est = speech_est[:, :actual_length]
                music_est = music_est[:, :actual_length]
                
                # Copy to output
                speech_output[:, start_idx:end_idx] = speech_est
                music_output[:, start_idx:end_idx] = music_est
                
                print(f"Processed segment {i+1}/{num_segments}")
        
        else:
            # Process entire file at once
            # Ensure batch dimension
            input_tensor = mixture.unsqueeze(0).to(device)
            
            # Process through model
            outputs = model(input_tensor)
            
            # Extract speech and music estimates
            speech_output = outputs[0, 0].cpu()  # First source is speech
            music_output = outputs[0, 1].cpu()   # Second source is music
    
    return speech_output, music_output


def batch_separate(dataset_dir, output_dir=None, checkpoint_path=None, split="test",
                  segment=None, sr=8000, batch_size=1, num_workers=0, max_files=None):
    """
    Batch process all triplet files in the dataset using HTDemucs model
    
    Args:
        dataset_dir: Root directory of the dataset
        output_dir: Directory to save output files
        checkpoint_path: Path to model checkpoint
        split: Dataset split to use (test, val, train)
        segment: Max segment length in seconds
        sr: Sample rate
        batch_size: Batch size for dataloader (typically 1 for separation)
        num_workers: Number of workers for dataloader
        max_files: Maximum number of files to process (None for all)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set default paths
    if checkpoint_path is None:
        possible_paths = [
            'checkpoints_htdemucs/best_model.pth',
            'checkpoints/best_model.pth',
            'models/htdemucs.pth'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        if checkpoint_path is None:
            raise FileNotFoundError("Could not find a model checkpoint. Please specify with --checkpoint")
    
    # Set default output directory
    if output_dir is None:
        model_name = Path(checkpoint_path).stem.replace('best_model', 'results')
        output_dir = f"batch_results_{model_name}_{split}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Load model
    model, model_args = load_model(checkpoint_path, device)
    
    # Create dataset and dataloader
    dataset = FolderTripletDataset(Path(dataset_dir), split=split, segment=3.0, sr=sr)
    
    # Create a subset of the dataset if max_files is specified
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if max_files is not None and max_files < dataset_size:
        indices = indices[:max_files]
        print(f"Limiting to {max_files} files (out of {dataset_size} found)")
    
    # We need to use the sample_dirs from the dataset to get the actual directory paths
    sample_dirs = [dataset.sample_dirs[i] for i in indices]
    
    start_time = time.time()
    
    # Process each sample directory
    for i, sample_dir in enumerate(sample_dirs):
        # Get the sample ID from the directory name
        sample_id = sample_dir.name
        print(f"\nProcessing file {i+1}/{len(indices)}: {sample_id}")
        
        # Create output directory for this file
        output_sample_dir = os.path.join(output_dir, sample_id)
        os.makedirs(output_sample_dir, exist_ok=True)
        
        # Load the mixture file
        mix_path = sample_dir / "mixture.wav"
        if not mix_path.exists():
            mix_path = sample_dir / "mix.wav"
            if not mix_path.exists():
                print(f"Skipping sample {sample_id}: No mixture file found")
                continue
        
        # Load ground truth files if they exist
        speech_path = sample_dir / "speech.wav"
        music_path = sample_dir / "music.wav"
        
        # Load the audio files
        mixture, sr_mix = torchaudio.load(str(mix_path))
        if sr_mix != sr:
            mixture = torchaudio.functional.resample(mixture, sr_mix, sr)
        
        # Convert to mono if needed
        if mixture.size(0) > 1:
            mixture = mixture.mean(dim=0, keepdim=True)
        
        # Process audio through model
        speech_output, music_output = process_audio(
            mixture, 
            model, 
            segment=segment,
            sample_rate=sr,
            device=device
        )
        
        # Save estimated outputs
        save_audio(mixture, os.path.join(output_sample_dir, 'gt_mixture.wav'), sr)
        save_audio(speech_output, os.path.join(output_sample_dir, 'est_speech.wav'), sr)
        save_audio(music_output, os.path.join(output_sample_dir, 'est_music.wav'), sr)
        
        # Save ground truth files if they exist
        if speech_path.exists():
            speech_gt, sr_speech = torchaudio.load(str(speech_path))
            if sr_speech != sr:
                speech_gt = torchaudio.functional.resample(speech_gt, sr_speech, sr)
            if speech_gt.size(0) > 1:
                speech_gt = speech_gt.mean(dim=0, keepdim=True)
            save_audio(speech_gt, os.path.join(output_sample_dir, 'gt_speech.wav'), sr)
        
        if music_path.exists():
            music_gt, sr_music = torchaudio.load(str(music_path))
            if sr_music != sr:
                music_gt = torchaudio.functional.resample(music_gt, sr_music, sr)
            if music_gt.size(0) > 1:
                music_gt = music_gt.mean(dim=0, keepdim=True)
            save_audio(music_gt, os.path.join(output_sample_dir, 'gt_music.wav'), sr)
        
        print(f"Processed {sample_id}: Separation complete")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Batch Processing Complete - {len(indices)} files")
    print(f"Results saved to: {output_dir}")
    print(f"Total processing time: {time.time() - start_time:.1f} seconds")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch separate speech and music from a dataset using HTDemucs")
    parser.add_argument('dataset_dir', type=str, help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save output files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (defaults to automatic detection)')
    parser.add_argument('--split', type=str, default="test",
                        help='Dataset split to use (test, val, train)')
    parser.add_argument('--segment', type=float, default=3.0,
                        help='Max segment length in seconds for processing long files (default: 3.0)')
    parser.add_argument('--sr', type=int, default=8000, 
                        help='Target sample rate (default: 8000)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for dataloader (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for dataloader (default: 0)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (default: all)')
    
    args = parser.parse_args()
    
    batch_separate(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        segment=args.segment,
        sr=args.sr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_files=args.max_files
    ) 