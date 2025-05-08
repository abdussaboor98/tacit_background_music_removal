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
from Conv_TasNet import ConvTasNet
from audio_datasets import FolderTripletDataset
from torch.utils.data import DataLoader

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get('args', {})
    
    # Create model with the same architecture as used in training
    model = ConvTasNet(
        N=model_args.get('N', 512),               # Number of filters in autoencoder
        L=model_args.get('L', 16),                # Length of the filters (in samples)
        B=model_args.get('B', 128),               # Number of channels in bottleneck and residual paths
        H=model_args.get('H', 512),               # Number of channels in convolutional blocks
        P=model_args.get('P', 3),                 # Kernel size in convolutional blocks
        X=model_args.get('X', 8),                 # Number of convolutional blocks in each repeat
        R=model_args.get('R', 3),                 # Number of repeats
        norm=model_args.get('norm', "gln"),       # Normalization type
        num_spks=2,                               # Number of speakers (fixed at 2: speech and music)
        activate=model_args.get('activate', "relu"), # Activation function
        causal=model_args.get('causal', False)    # Causal or non-causal model
    ).to(device)
    
    # Load the weights
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
    """Process audio mixture through the model"""
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
                
                # ConvTasNet expects (B, T) or (T)
                if segment_waveform.shape[0] == 1:  # Mono
                    segment_input = segment_waveform.squeeze(0).to(device)  # (T)
                else:  # Multi-channel, average to mono
                    segment_input = segment_waveform.mean(0).to(device)  # (T)
                
                # Process through model
                s_estimates = model(segment_input)  # Returns list of tensors [speech, music]
                
                # Extract speech and music estimates
                speech_est = s_estimates[0]  # Speech (first element)
                music_est = s_estimates[1]   # Music (second element)
                
                # Add channel dimension and copy to output
                speech_output[:, start_idx:end_idx] = speech_est.unsqueeze(0).cpu()
                music_output[:, start_idx:end_idx] = music_est.unsqueeze(0).cpu()
                
                print(f"Processed segment {i+1}/{num_segments}")
        
        else:
            # Process entire file at once
            # ConvTasNet expects (B, T) or (T)
            if mixture.shape[0] == 1:  # Mono
                input_tensor = mixture.squeeze(0).to(device)  # (T)
            else:  # Multi-channel, average to mono
                input_tensor = mixture.mean(0).to(device)  # (T)
            
            # Process through model
            s_estimates = model(input_tensor)  # Returns list of tensors [speech, music]
            
            # Extract speech and music estimates
            speech_output = s_estimates[0].unsqueeze(0).cpu()  # Add channel dimension
            music_output = s_estimates[1].unsqueeze(0).cpu()   # Add channel dimension
    
    return speech_output, music_output


def batch_separate(dataset_dir, output_dir=None, checkpoint_path=None, split="test",
                  segment=None, sr=8000, batch_size=1, num_workers=0, max_files=None):
    """
    Batch process all triplet files in the dataset
    
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
            'best_model.pth'
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
    # Use a very large segment length (3600 seconds = 1 hour) instead of None
    # to ensure we process entire files while avoiding the None multiplication error
    dataset = FolderTripletDataset(Path(dataset_dir), split=split, segment=10)
    
    # Create a subset of the dataset if max_files is specified
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if max_files is not None and max_files < dataset_size:
        indices = indices[:max_files]
        print(f"Limiting to {max_files} files (out of {dataset_size} found)")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices) if max_files else None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    start_time = time.time()
    
    for i, (mixture, speech_gt, music_gt) in enumerate(dataloader):
        # Generate a unique file ID based on index
        file_id = f"sample_{i:04d}"
        print(f"\nProcessing file {i+1}/{len(indices) if max_files else dataset_size}: {file_id}")
        
        # Create output directory for this file
        sample_dir = os.path.join(output_dir, file_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Convert to torch tensor and ensure correct shape
        mixture = mixture.squeeze(0)  # Remove batch dimension if batch_size=1
        speech_gt = speech_gt.squeeze(0)  # Remove batch dimension if batch_size=1
        music_gt = music_gt.squeeze(0)  # Remove batch dimension if batch_size=1
        
        # Process audio through model
        speech_output, music_output = process_audio(
            mixture, 
            model, 
            segment=segment,
            sample_rate=sr,
            device=device
        )
        
        # Save original mixture and ground truths
        save_audio(mixture, os.path.join(sample_dir, 'gt_mixture.wav'), sr)
        save_audio(speech_gt, os.path.join(sample_dir, 'gt_speech.wav'), sr)
        save_audio(music_gt, os.path.join(sample_dir, 'gt_music.wav'), sr)
        
        # Save estimated outputs
        save_audio(speech_output, os.path.join(sample_dir, 'est_speech.wav'), sr)
        save_audio(music_output, os.path.join(sample_dir, 'est_music.wav'), sr)
        
        print(f"Processed {file_id}: Separation complete")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Batch Processing Complete - {len(indices) if max_files else dataset_size} files")
    print(f"Results saved to: {output_dir}")
    print(f"Total processing time: {time.time() - start_time:.1f} seconds")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch separate speech and music from a dataset")
    parser.add_argument('dataset_dir', type=str, help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save output files')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (defaults to automatic detection)')
    parser.add_argument('--split', type=str, default="test",
                        help='Dataset split to use (test, val, train)')
    parser.add_argument('--segment', type=float, default=None,
                        help='Max segment length in seconds for processing long files (default: process entire file)')
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