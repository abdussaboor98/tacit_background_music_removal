# train.py (Modified for HTDemucs with L1 Loss)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import argparse
import time
import os
import numpy as np
from tqdm import tqdm # For progress bars
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random

# Import HTDemucs model instead of MSHybridNet
from demucs.htdemucs import HTDemucs
from audio_datasets import FolderTripletDataset
# Using torchmetrics for SI-SNR, PESQ, STOI
import torchmetrics

# --- Loss Functions ---

class HTDemucsL1Loss(nn.Module):
    """
    L1 loss for HTDemucs model with two sources (speech and music).
    Computes L1 loss between each estimated source and its target.
    """
    def __init__(self, speech_target_index=0, music_target_index=1):
        super().__init__()
        self.speech_target_index = speech_target_index
        self.music_target_index = music_target_index
        self.loss_fn = nn.L1Loss()
        
    def forward(self, estimates, targets):
        """
        Args:
            estimates: Tensor of shape (batch, n_sources, channels, time)
            targets: Tensor of shape (batch, n_sources, channels, time)
        Returns:
            total_loss: Sum of L1 losses for all sources
            speech_loss: L1 loss for speech
            music_loss: L1 loss for music
        """
        batch_size = estimates.shape[0]
        device = estimates.device
        
        # Ensure all inputs are on the same device
        targets = targets.to(device)
        
        # Extract speech and music components
        speech_est = estimates[:, self.speech_target_index]
        speech_tgt = targets[:, self.speech_target_index]
        music_est = estimates[:, self.music_target_index]
        music_tgt = targets[:, self.music_target_index]
        
        # Calculate L1 losses
        speech_loss = self.loss_fn(speech_est, speech_tgt)
        music_loss = self.loss_fn(music_est, music_tgt)
        
        # Total loss is the sum of both source losses
        total_loss = speech_loss + music_loss
        
        return total_loss, speech_loss, music_loss

# Function for SI-SNR calculation (used for evaluation metrics only)
def si_snr_loss(estimate, target, epsilon=1e-8):
    """Calculate negative SI-SNR loss for evaluation metrics"""
    # Ensure inputs have a batch dimension
    if estimate.ndim == 1: estimate = estimate.unsqueeze(0)
    if target.ndim == 1: target = target.unsqueeze(0)
    
    # Move to same device
    device = estimate.device
    target = target.to(device)
    
    # Handle length mismatches
    min_len = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_len]
    target = target[..., :min_len]
    
    # Zero mean normalization
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # SI-SNR calculation
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + epsilon
    scale = torch.sum(estimate * target, dim=-1, keepdim=True) / target_energy
    scaled_target = scale * target
    
    noise = estimate - scaled_target
    target_power = torch.sum(scaled_target**2, dim=-1) + epsilon
    noise_power = torch.sum(noise**2, dim=-1) + epsilon
    
    si_snr = -10 * torch.log10(target_power / noise_power)
    return si_snr

# --- Evaluation Function (Modified for HTDemucs with L1 Loss) ---
def evaluate(model, dataloader, loss_fn_eval, metrics_dict, device, desc="Evaluating", max_batches=0, sample_rate=8000):
    """ Evaluate HTDemucs model on a dataloader """
    model.eval()
    total_loss = 0.0
    total_speech_loss = 0.0
    total_music_loss = 0.0
    num_samples = 0

    # Reset metrics
    for metric in metrics_dict.values():
        metric.reset()

    # Accumulators for per-source SI-SNR metrics
    accumulated_speech_sisnr_metric = 0.0
    accumulated_music_sisnr_metric = 0.0

    with torch.no_grad():
        total_batches = min(len(dataloader), max_batches) if max_batches > 0 else len(dataloader)
        eval_pbar = tqdm(dataloader, desc=desc, total=total_batches)
        for batch_idx, (mixture, target_speech, target_music) in enumerate(eval_pbar):
            # Limit number of batches if max_batches > 0
            if max_batches > 0 and batch_idx >= max_batches:
                break

            mixture_input = mixture.to(device)           # (B, C, T)

            # Targets: Stack and ensure correct shape (B, N_src, C, T)
            if target_speech.ndim == 3: # Already (B, C, T)
                targets = torch.stack([target_speech, target_music], dim=1).to(device)
            else: # Assume (B, T), add channel dim
                targets = torch.stack([target_speech.unsqueeze(1), target_music.unsqueeze(1)], dim=1).to(device)

            batch_size = mixture_input.shape[0]
            num_samples += batch_size

            # Model forward pass -> estimates (B, N_src, C, T)
            estimates = model(mixture_input)
            
            # Calculate loss using the provided loss function instance
            batch_loss, batch_speech_loss, batch_music_loss = loss_fn_eval(estimates, targets)
            
            total_loss += batch_loss.item() * batch_size
            total_speech_loss += batch_speech_loss.item() * batch_size
            total_music_loss += batch_music_loss.item() * batch_size
            
            # Calculate SI-SNR metrics (not used for training but for evaluation)
            speech_est = estimates[:, loss_fn_eval.speech_target_index]   # (B, C, T)
            music_est = estimates[:, loss_fn_eval.music_target_index]     # (B, C, T)
            speech_tgt = targets[:, loss_fn_eval.speech_target_index]     # (B, C, T)
            music_tgt = targets[:, loss_fn_eval.music_target_index]       # (B, C, T)
            
            # Calculate SI-SNR for speech and music
            speech_est_mono = speech_est.squeeze(1) if speech_est.shape[1] == 1 else torch.mean(speech_est, dim=1)
            speech_tgt_mono = speech_tgt.squeeze(1) if speech_tgt.shape[1] == 1 else torch.mean(speech_tgt, dim=1)
            music_est_mono = music_est.squeeze(1) if music_est.shape[1] == 1 else torch.mean(music_est, dim=1)
            music_tgt_mono = music_tgt.squeeze(1) if music_tgt.shape[1] == 1 else torch.mean(music_tgt, dim=1)
            
            for i in range(batch_size):
                speech_sisnr = -si_snr_loss(speech_est_mono[i], speech_tgt_mono[i]).item()
                music_sisnr = -si_snr_loss(music_est_mono[i], music_tgt_mono[i]).item()
                accumulated_speech_sisnr_metric += speech_sisnr
                accumulated_music_sisnr_metric += music_sisnr
            
            # Update progress bar with current metrics
            if num_samples > 0:
                eval_pbar.set_postfix(
                    speech=f"{accumulated_speech_sisnr_metric / num_samples:.2f}dB",
                    music=f"{accumulated_music_sisnr_metric / num_samples:.2f}dB"
                )

            # Update PESQ/STOI for speech only
            min_len = min(speech_est_mono.shape[-1], speech_tgt_mono.shape[-1])
            speech_est_trim = speech_est_mono[..., :min_len]
            speech_tgt_trim = speech_tgt_mono[..., :min_len]

            valid_indices = torch.sum(torch.abs(speech_tgt_trim), dim=-1) > 1e-5
            if torch.any(valid_indices):
                estimates_valid = speech_est_trim[valid_indices].to(device)
                targets_valid = speech_tgt_trim[valid_indices].to(device)
                # Get sample rate from loss function or fall back to default
                sample_rate = 8000  # Default sample rate
                pesq_key = f"PESQ-{'WB' if sample_rate == 16000 else 'NB'}"
                if pesq_key in metrics_dict and estimates_valid.numel() > 0:
                    try: metrics_dict[pesq_key].update(estimates_valid, targets_valid)
                    except Exception as e: print(f"Warning: PESQ calculation failed: {e}")
                if "STOI" in metrics_dict and estimates_valid.numel() > 0:
                    try: metrics_dict["STOI"].update(estimates_valid, targets_valid)
                    except Exception as e: print(f"Warning: STOI calculation failed: {e}")

    # Compute final metric values
    if num_samples > 0:
        results = {name: metric.compute().item() for name, metric in metrics_dict.items()}
        avg_loss = total_loss / num_samples
        avg_speech_loss = total_speech_loss / num_samples
        avg_music_loss = total_music_loss / num_samples
        
        results["Loss_Total"] = avg_loss
        results["Loss_Speech(L1)"] = avg_speech_loss
        results["Loss_Music(L1)"] = avg_music_loss
        
        # Add separate speech and music SI-SNR metric values (for reporting only)
        results["Speech_SI-SNR_Eval"] = accumulated_speech_sisnr_metric / num_samples
        results["Music_SI-SNR_Eval"] = accumulated_music_sisnr_metric / num_samples
    else:
        # Handle case where no samples were processed
        results = {}
        # Default sample rate for PESQ key
        pesq_key = f"PESQ-{'WB' if sample_rate == 16000 else 'NB'}"
        # Add default values for all necessary metrics
        results["SI-SNR"] = 0.0
        results["Loss_Total"] = 0.0
        results["Loss_Speech(L1)"] = 0.0
        results["Loss_Music(L1)"] = 0.0
        results["Speech_SI-SNR_Eval"] = 0.0
        results["Music_SI-SNR_Eval"] = 0.0
        results[pesq_key] = 0.0
        results["STOI"] = 0.0

    return results

# --- Helper Functions ---
# Keep save_audio, plot_losses, plot_si_snr, find_latest_checkpoint, train_sample_index
# definitions EXACTLY as provided in the uploaded train.py file.
# Make sure plot_si_snr can handle the new keys in the checkpoint history if added.

def save_audio(tensor, filepath, sample_rate=8000):
    """Save audio tensor to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if tensor.device != torch.device('cpu'): tensor = tensor.cpu()
    if tensor.ndim == 4: tensor = tensor[0, 0] # (B, N, C, T) -> (C, T)
    elif tensor.ndim == 3: tensor = tensor[0] # (B, C, T) or (N, C, T) -> (C, T)
    elif tensor.ndim == 1: tensor = tensor.unsqueeze(0) # (T) -> (C=1, T)
    if tensor.ndim != 2 or tensor.shape[0] > 16 : # Basic check for sensible shape
        print(f"Warning: Unexpected tensor shape for audio saving: {tensor.shape}. Forcing to mono.")
        tensor = tensor.mean(dim=0, keepdim=True) # Average channels if multi-channel
        tensor = tensor.reshape(1, -1) # Force to (1, Time)
    max_val = torch.max(torch.abs(tensor));
    if max_val > 0.999: tensor = tensor / (max_val * 1.05)
    torchaudio.save(filepath, tensor, sample_rate); print(f"Saved audio to {filepath}")

def save_spectrogram(tensor, filepath, sample_rate=8000, n_fft=512, hop_length=128, title=None):
    """Generate and save a spectrogram visualization from an audio tensor"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Move to CPU if on GPU
    if tensor.device != torch.device('cpu'): tensor = tensor.cpu()
    
    # Handle different tensor shapes to get a flat (Time) or (1, Time) tensor
    if tensor.ndim == 4: tensor = tensor[0, 0] # (B, N, C, T) -> (C, T)
    elif tensor.ndim == 3: tensor = tensor[0] # (B, C, T) or (N, C, T) -> (C, T)
    if tensor.ndim == 2 and tensor.shape[0] > 1: # Multi-channel audio
        tensor = tensor.mean(dim=0) # Average channels
    if tensor.ndim == 2: tensor = tensor.squeeze(0) # (1, T) -> (T)
    
    # Create spectrogram using torchaudio
    specgram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    
    # Calculate spectrogram
    spec = specgram_transform(tensor)  # (1, Freq, Time) or (Freq, Time)
    if spec.ndim == 3: spec = spec.squeeze(0)  # (Freq, Time)
    
    # Convert to dB scale with appropriate handling of zeros
    spec_db = 10 * torch.log10(spec + 1e-10)
    
    # Plot spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap='inferno')
    
    # Add labels and title
    plt.xlabel('Frames')
    plt.ylabel('Frequency Bins')
    if title:
        plt.title(title)
    
    # Add colorbar
    plt.colorbar(format='%+2.0f dB')
    
    # Save figure and close
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved spectrogram to {filepath}")

def plot_losses(train_losses, val_losses, save_path, train_losses_components=None, val_losses_components=None):
    """
    Plot training and validation losses as separate plots.
    Creates individual figures for:
    1. Overall loss
    2. Speech L1 loss
    3. Music L1 loss
    
    Args:
        train_losses: List of total training losses
        val_losses: List of total validation losses
        save_path: Path to save the figure
        train_losses_components: Dict of component losses for training (speech_l1, music_l1)
        val_losses_components: Dict of component losses for validation
    """
    if not train_losses or not val_losses:
        print("Warning: Loss history is empty, skipping plotting.")
        return None
    
    epochs = range(1, len(train_losses) + 1)
    
    # Make sure we have component losses
    has_components = False
    if train_losses_components and val_losses_components:
        epochs_len = len(epochs)
        train_speech_l1 = train_losses_components.get('speech_l1', [0] * epochs_len)[:epochs_len]
        val_speech_l1 = val_losses_components.get('speech_l1', [0] * epochs_len)[:epochs_len]
        train_music_l1 = train_losses_components.get('music_l1', [0] * epochs_len)[:epochs_len]
        val_music_l1 = val_losses_components.get('music_l1', [0] * epochs_len)[:epochs_len]
        has_components = True
    
    # Create a multiple-plot figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 1. Overall Loss Plot
    # -------------------
    ax = axs[0]
    line1, = ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Total Loss')
    line2, = ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Total Loss')
    
    ax.set_title('Overall Loss')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Annotate minimum values
    min_train_epoch = np.argmin(train_losses) + 1
    min_val_epoch = np.argmin(val_losses) + 1
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    
    ax.annotate(f'Min Train: {min_train_loss:.4f} (Ep {min_train_epoch})', 
                xy=(min_train_epoch, min_train_loss), 
                xytext=(min_train_epoch, min_train_loss * 1.3), 
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    ax.annotate(f'Min Val: {min_val_loss:.4f} (Ep {min_val_epoch})', 
                xy=(min_val_epoch, min_val_loss), 
                xytext=(min_val_epoch, min_val_loss * 1.3), 
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    # 2. Speech L1 Loss Plot
    # ---------------------
    ax = axs[1]
    
    if has_components:
        line1, = ax.plot(epochs, train_speech_l1, 'b-', linewidth=2, label='Train Speech L1')
        line2, = ax.plot(epochs, val_speech_l1, 'r-', linewidth=2, label='Val Speech L1')
        
        ax.set_title('Speech L1 Loss')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        
        # Annotate minimum values
        min_train_speech = np.argmin(train_speech_l1) + 1
        min_val_speech = np.argmin(val_speech_l1) + 1
        min_train_speech_loss = min(train_speech_l1)
        min_val_speech_loss = min(val_speech_l1)
        
        ax.annotate(f'Min Train: {min_train_speech_loss:.4f} (Ep {min_train_speech})', 
                    xy=(min_train_speech, min_train_speech_loss), 
                    xytext=(min_train_speech, min_train_speech_loss * 1.3), 
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
        
        ax.annotate(f'Min Val: {min_val_speech_loss:.4f} (Ep {min_val_speech})', 
                    xy=(min_val_speech, min_val_speech_loss), 
                    xytext=(min_val_speech, min_val_speech_loss * 1.3), 
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
    else:
        ax.text(0.5, 0.5, 'Speech L1 loss data not available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    
    # 3. Music L1 Loss Plot
    # -------------------
    ax = axs[2]
    
    if has_components:
        line1, = ax.plot(epochs, train_music_l1, 'b-', linewidth=2, label='Train Music L1')
        line2, = ax.plot(epochs, val_music_l1, 'r-', linewidth=2, label='Val Music L1')
        
        ax.set_title('Music L1 Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')
        
        # Annotate minimum values
        min_train_music = np.argmin(train_music_l1) + 1
        min_val_music = np.argmin(val_music_l1) + 1
        min_train_music_loss = min(train_music_l1)
        min_val_music_loss = min(val_music_l1)
        
        ax.annotate(f'Min Train: {min_train_music_loss:.4f} (Ep {min_train_music})', 
                    xy=(min_train_music, min_train_music_loss), 
                    xytext=(min_train_music, min_train_music_loss * 1.3), 
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
        
        ax.annotate(f'Min Val: {min_val_music_loss:.4f} (Ep {min_val_music})', 
                    xy=(min_val_music, min_val_music_loss), 
                    xytext=(min_val_music, min_val_music_loss * 1.3), 
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
    else:
        ax.text(0.5, 0.5, 'Music L1 loss data not available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")
    return fig


def plot_si_snr(epochs, train_speech, train_music, val_speech, val_music, train_overall, val_overall, save_path):
    """
    Plot SI-SNR metrics as separate plots:
    1. Speech SI-SNR 
    2. Music SI-SNR
    3. Overall SI-SNR
    """
    if not epochs or not train_speech or not val_speech:
        print("Warning: SI-SNR history is empty or incomplete, skipping plotting.")
        return
    
    # Ensure lists have same length as epochs for plotting
    min_len = len(epochs)
    train_speech = train_speech[:min_len]
    train_music = train_music[:min_len]
    val_speech = val_speech[:min_len]
    val_music = val_music[:min_len]
    train_overall = train_overall[:min_len] if train_overall else []
    val_overall = val_overall[:min_len] if val_overall else []
    
    # Create a 3-subplot figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Speech SI-SNR
    # -------------------
    ax = axs[0]
    
    line1, = ax.plot(epochs, train_speech, 'b-', linewidth=2, label='Train Speech SI-SNR')
    line2, = ax.plot(epochs, val_speech, 'r-', linewidth=2, label='Val Speech SI-SNR')
    
    ax.set_title('Speech SI-SNR Metrics')
    ax.set_ylabel('SI-SNR (dB)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    
    # Annotate maximum values (higher SI-SNR is better)
    max_train_speech_idx = np.argmax(train_speech)
    max_val_speech_idx = np.argmax(val_speech)
    
    ax.annotate(f'Max Train: {train_speech[max_train_speech_idx]:.2f} dB (Ep {epochs[max_train_speech_idx]})', 
                xy=(epochs[max_train_speech_idx], train_speech[max_train_speech_idx]), 
                xytext=(epochs[max_train_speech_idx], train_speech[max_train_speech_idx] + 1),
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    ax.annotate(f'Max Val: {val_speech[max_val_speech_idx]:.2f} dB (Ep {epochs[max_val_speech_idx]})', 
                xy=(epochs[max_val_speech_idx], val_speech[max_val_speech_idx]), 
                xytext=(epochs[max_val_speech_idx], val_speech[max_val_speech_idx] + 1),
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    # Plot 2: Music SI-SNR
    # ------------------
    ax = axs[1]
    
    line1, = ax.plot(epochs, train_music, 'b-', linewidth=2, label='Train Music SI-SNR')
    line2, = ax.plot(epochs, val_music, 'r-', linewidth=2, label='Val Music SI-SNR')
    
    ax.set_title('Music SI-SNR Metrics')
    ax.set_ylabel('SI-SNR (dB)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    
    # Annotate maximum values
    max_train_music_idx = np.argmax(train_music)
    max_val_music_idx = np.argmax(val_music)
    
    ax.annotate(f'Max Train: {train_music[max_train_music_idx]:.2f} dB (Ep {epochs[max_train_music_idx]})', 
                xy=(epochs[max_train_music_idx], train_music[max_train_music_idx]), 
                xytext=(epochs[max_train_music_idx], train_music[max_train_music_idx] + 1),
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    ax.annotate(f'Max Val: {val_music[max_val_music_idx]:.2f} dB (Ep {epochs[max_val_music_idx]})', 
                xy=(epochs[max_val_music_idx], val_music[max_val_music_idx]), 
                xytext=(epochs[max_val_music_idx], val_music[max_val_music_idx] + 1),
                ha='center',
                arrowprops=dict(arrowstyle='->'))
    
    # Plot 3: Overall SI-SNR (PIT)
    # --------------------------
    ax = axs[2]
    
    if train_overall and val_overall:
        line1, = ax.plot(epochs, train_overall, 'b-', linewidth=2, label='Train Overall SI-SNR (PIT)')
        line2, = ax.plot(epochs, val_overall, 'r-', linewidth=2, label='Val Overall SI-SNR (PIT)')
        
        # Also show average of speech and music SI-SNR for comparison
        train_avg = [(s + m) / 2 for s, m in zip(train_speech, train_music)]
        val_avg = [(s + m) / 2 for s, m in zip(val_speech, val_music)]
        
        line3, = ax.plot(epochs, train_avg, 'b--', alpha=0.7, linewidth=1, label='Train Avg SI-SNR')
        line4, = ax.plot(epochs, val_avg, 'r--', alpha=0.7, linewidth=1, label='Val Avg SI-SNR')
        
        ax.set_title('Overall SI-SNR')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('SI-SNR (dB)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        
        # Annotate maximum values
        max_train_overall_idx = np.argmax(train_overall)
        max_val_overall_idx = np.argmax(val_overall)
        
        ax.annotate(f'Max Train: {train_overall[max_train_overall_idx]:.2f} dB (Ep {epochs[max_train_overall_idx]})', 
                    xy=(epochs[max_train_overall_idx], train_overall[max_train_overall_idx]), 
                    xytext=(epochs[max_train_overall_idx], train_overall[max_train_overall_idx] + 1),
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
        
        ax.annotate(f'Max Val: {val_overall[max_val_overall_idx]:.2f} dB (Ep {epochs[max_val_overall_idx]})', 
                    xy=(epochs[max_val_overall_idx], val_overall[max_val_overall_idx]), 
                    xytext=(epochs[max_val_overall_idx], val_overall[max_val_overall_idx] + 1),
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
    else:
        ax.text(0.5, 0.5, 'Overall SI-SNR (PIT) Data Not Available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel('Epochs')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"SI-SNR plot saved to {save_path}")
    return fig


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file based on epoch number."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints: return None
    epoch_nums = []
    for checkpoint in checkpoints:
        try:
            epoch_str = checkpoint.stem.split('_')[-1]
            epoch_num = int(epoch_str)
            epoch_nums.append((epoch_num, checkpoint))
        except (ValueError, IndexError): continue
    if not epoch_nums: return None
    latest_epoch, latest_checkpoint = sorted(epoch_nums, key=lambda x: x[0])[-1]
    print(f"Found {len(epoch_nums)} checkpoints. Latest is epoch {latest_epoch}.")
    return latest_checkpoint, latest_epoch

def train_sample_index(dataset, sample_index=-1):
     """Helper function to get a sample from a dataset."""
     if sample_index < 0 or sample_index >= len(dataset):
          sample_index = np.random.randint(0, len(dataset))
     mixture, target_speech, target_music = dataset[sample_index]
     return mixture, target_speech, target_music, sample_index


# --- Training Setup ---

def print_metrics_table(metrics, title="Evaluation Metrics"):
    """Print metrics in a nicely formatted table"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)
    
    # Determine the longest metric name for formatting
    max_len = max([len(k) for k in metrics.keys()])
    
    # Print each metric
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            # Add dB suffix for SI-SNR metrics
            if "SI-SNR" in key or "SDR" in key:
                print(f"{key.ljust(max_len)} : {value:.2f} dB")
            # Format PESQ and STOI differently
            elif "PESQ" in key:
                print(f"{key.ljust(max_len)} : {value:.2f} / 5.0")
            elif "STOI" in key:
                print(f"{key.ljust(max_len)} : {value:.3f} (0-1)")
            # Loss values
            elif "Loss" in key:
                print(f"{key.ljust(max_len)} : {value:.4f}")
            else:
                print(f"{key.ljust(max_len)} : {value:.4f}")
    
    print("=" * 60)

def train(args):
    """Main training loop (Modified for HTDemucs with L1 Loss)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    root_path = Path(args.root_dir)
    train_dataset = FolderTripletDataset(root_path, split="train", segment=args.segment, sr=args.sr)
    # Check for different validation folder naming conventions
    if (root_path / "val").exists():
        val_split_name = "val"
    elif (root_path / "validation").exists():
        val_split_name = "validation"
    elif (root_path / "valid").exists():
        val_split_name = "valid"
    else:
        print("Warning: No validation folder found. Defaulting to 'val'")
        val_split_name = "val"
    val_dataset = FolderTripletDataset(root_path, split=val_split_name, segment=args.segment, sr=args.sr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Setup directories ---
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard setup
    tb_log_dir = output_dir / 'tensorboard_logs'
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to {tb_log_dir}")
    print(f"Run 'tensorboard --logdir={tb_log_dir}' to view training progress")

    # --- Setup for saving audio samples and spectrograms ---
    samples_dir = os.path.join(args.save_dir, 'audio_samples')
    spectrograms_dir = os.path.join(args.save_dir, 'spectrograms')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(spectrograms_dir, exist_ok=True)
    save_sample_flag = False
    try:
        fixed_example_mixture, fixed_example_speech, fixed_example_music, sample_idx_to_use = train_sample_index(val_dataset, args.sample_index)
        # Make sure examples are on CPU initially to avoid memory issues
        fixed_example_mixture = fixed_example_mixture.cpu()
        fixed_example_speech = fixed_example_speech.cpu()
        fixed_example_music = fixed_example_music.cpu()
        print(f"Using validation sample at index {sample_idx_to_use} for audio monitoring")

        # Save ground truth audio
        save_audio(fixed_example_mixture, os.path.join(samples_dir, 'gt_mixture.wav'), args.sr)
        save_audio(fixed_example_speech, os.path.join(samples_dir, 'gt_speech.wav'), args.sr)
        save_audio(fixed_example_music, os.path.join(samples_dir, 'gt_music.wav'), args.sr)

        # Save ground truth spectrograms
        save_spectrogram(
            fixed_example_mixture, 
            os.path.join(spectrograms_dir, 'gt_mixture_spec.png'), 
            args.sr, args.n_fft, args.hop_length,
            title="Ground Truth Mixture"
        )
        save_spectrogram(
            fixed_example_speech, 
            os.path.join(spectrograms_dir, 'gt_speech_spec.png'),
            args.sr, args.n_fft, args.hop_length,
            title="Ground Truth Speech"
        )
        save_spectrogram(
            fixed_example_music, 
            os.path.join(spectrograms_dir, 'gt_music_spec.png'),
            args.sr, args.n_fft, args.hop_length,
            title="Ground Truth Music"
        )

        save_sample_flag = True
    except Exception as e:
        print(f"Could not get fixed validation sample for saving: {e}")

    # --- Model ---
    # Define sources for HTDemucs
    sources = ["speech", "music"]
    
    # Initialize HTDemucs model
    model = HTDemucs(
        sources=sources,
        audio_channels=1,  # Force mono for consistency
        samplerate=args.sr,
        segment=args.segment,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss Function (L1 for source separation) ---
    loss_fn = HTDemucsL1Loss(
        speech_target_index=0,  # speech is first source
        music_target_index=1     # music is second source
    ).to(device)

    # --- Metrics for Validation/Test ---
    pesq_mode = 'wb' if args.sr == 16000 else 'nb'
    pesq_key = f"PESQ-{pesq_mode.upper()}"

    eval_metric_list = {} # Initialize empty
    
    # Fix SI-SNR import
    try:
        # Try current import structure
        eval_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    except (AttributeError, ImportError):
        try:
            # Try alternate import path
            from torchmetrics.functional.audio.sdr import scale_invariant_signal_noise_ratio
            # Create a wrapper metric
            class SISNRMetric(torchmetrics.Metric):
                def __init__(self):
                    super().__init__()
                    self.add_state("sisnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            score = scale_invariant_signal_noise_ratio(pred, tgt)
                            self.sisnr_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"SI-SNR calculation failed: {e}")
                
                def compute(self):
                    return self.sisnr_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["SI-SNR"] = SISNRMetric().to(device)
        except (AttributeError, ImportError):
            print(f"Warning: SI-SNR metric not available. Using custom implementation.")
            # Use our custom SI-SNR implementation
            class CustomSISNRMetric(torchmetrics.Metric):
                def __init__(self):
                    super().__init__()
                    self.add_state("sisnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            # Use the custom si_snr_loss function but negate it (since it returns negative SI-SNR)
                            score = -si_snr_loss(pred, tgt).mean()
                            self.sisnr_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"Custom SI-SNR calculation failed: {e}")
                
                def compute(self):
                    return self.sisnr_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["SI-SNR"] = CustomSISNRMetric().to(device)

    if pesq_mode:
        try:
            # Try the current import structure
            eval_metric_list[pesq_key] = torchmetrics.audio.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
        except (AttributeError, ImportError):
            try:
                # Try alternate import path
                from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
                # Create a wrapper metric since we can't import the class directly
                class PESQMetric(torchmetrics.Metric):
                    def __init__(self, fs, mode):
                        super().__init__()
                        self.fs = fs
                        self.mode = mode
                        self.add_state("pesq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                    
                    def update(self, preds, target):
                        for pred, tgt in zip(preds, target):
                            try:
                                score = perceptual_evaluation_speech_quality(pred, tgt, self.fs, self.mode)
                                self.pesq_sum += score
                                self.total += 1
                            except Exception as e:
                                print(f"PESQ calculation failed: {e}")
                    
                    def compute(self):
                        return self.pesq_sum / self.total if self.total > 0 else torch.tensor(0.0)
                
                eval_metric_list[pesq_key] = PESQMetric(args.sr, pesq_mode).to(device)
            except (AttributeError, ImportError):
                print(f"Warning: PESQ metric not available. Skipping PESQ evaluation.")
    
    # Fix STOI import
    try:
        # Try current import structure
        eval_metric_list["STOI"] = torchmetrics.audio.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
    except (AttributeError, ImportError):
        try:
            # Try alternate import path
            from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
            # Create a wrapper metric
            class STOIMetric(torchmetrics.Metric):
                def __init__(self, fs, extended=False):
                    super().__init__()
                    self.fs = fs
                    self.extended = extended
                    self.add_state("stoi_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            score = short_time_objective_intelligibility(pred, tgt, self.fs, self.extended)
                            self.stoi_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"STOI calculation failed: {e}")
                
                def compute(self):
                    return self.stoi_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["STOI"] = STOIMetric(args.sr, extended=False).to(device)
        except (AttributeError, ImportError):
            print(f"Warning: STOI metric not available. Skipping STOI evaluation.")
    eval_metrics = torchmetrics.MetricCollection(eval_metric_list).to(device)

    # --- Training Loop ---
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize History Lists ---
    train_losses_total = []
    val_losses_total = []
    train_speech_l1_loss = []
    val_speech_l1_loss = []
    train_music_l1_loss = []
    val_music_l1_loss = []
    val_pesq_hist = []  # For plotting metrics
    val_stoi_hist = []
    train_speech_si_snr = []  # For monitoring SI-SNR metrics
    val_speech_si_snr = []
    train_music_si_snr = []
    val_music_si_snr = []
    train_overall_si_snr = []
    val_overall_si_snr = []

    # --- Initialize Torchmetrics for Training SI-SNR ---
    train_sisnr_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)

    # Initialize epoch counter and best validation loss
    start_epoch = 0
    best_val_loss = float("inf")

    latest_checkpoint_info = find_latest_checkpoint(output_dir)
    if latest_checkpoint_info:
        checkpoint_path, latest_epoch = latest_checkpoint_info
        print(f"\nResuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # Load histories
            train_losses_total = checkpoint.get('train_losses_total', [])
            val_losses_total = checkpoint.get('val_losses_total', [])
            train_speech_l1_loss = checkpoint.get('train_speech_l1_loss', [])
            
            # Ensure val_speech_l1_loss is always a list
            val_speech_l1_loss_data = checkpoint.get('val_speech_l1_loss', [])
            if isinstance(val_speech_l1_loss_data, float):
                val_speech_l1_loss = [val_speech_l1_loss_data]
            else:
                val_speech_l1_loss = val_speech_l1_loss_data
                
            train_music_l1_loss = checkpoint.get('train_music_l1_loss', [])
            
            # Ensure val_music_l1_loss is always a list
            val_music_l1_loss_data = checkpoint.get('val_music_l1_loss', [])
            if isinstance(val_music_l1_loss_data, float):
                val_music_l1_loss = [val_music_l1_loss_data]
            else:
                val_music_l1_loss = val_music_l1_loss_data
                
            val_pesq_hist = checkpoint.get('val_pesq_hist', [])
            val_stoi_hist = checkpoint.get('val_stoi_hist', [])
            train_speech_si_snr = checkpoint.get('train_speech_si_snr', [])
            val_speech_si_snr = checkpoint.get('val_speech_si_snr', [])
            train_music_si_snr = checkpoint.get('train_music_si_snr', [])
            val_music_si_snr = checkpoint.get('val_music_si_snr', [])
            train_overall_si_snr = checkpoint.get('train_overall_si_snr', [])
            val_overall_si_snr = checkpoint.get('val_overall_si_snr', [])

            print(f"Resumed from epoch {start_epoch}. Best validation loss so far: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint state: {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float("inf")
            # Reset all history lists
            train_losses_total = []
            val_losses_total = []
            train_speech_l1_loss = []
            val_speech_l1_loss = []
            train_music_l1_loss = []
            val_music_l1_loss = []
            val_pesq_hist = []
            val_stoi_hist = []
            train_speech_si_snr = []
            val_speech_si_snr = []
            train_music_si_snr = []
            val_music_si_snr = []
            train_overall_si_snr = []
            val_overall_si_snr = []

    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses_total.append(0.0)
        train_speech_l1_loss.append(0.0)
        train_music_l1_loss.append(0.0)
        train_speech_si_snr.append(0.0)
        train_music_si_snr.append(0.0)
        train_overall_si_snr.append(0.0)
        
        # Initialize the progress bar for training with max_batches limit
        total_train_batches = min(len(train_loader), args.max_batches) if args.max_batches > 0 else len(train_loader)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, total=total_train_batches)

        for batch_idx, (mixture, target_speech, target_music) in enumerate(train_pbar):
            # Stop if we've reached max_batches
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
                
            mixture_input = mixture.to(device)           # (B, C, T)
            target_speech = target_speech.to(device)
            target_music = target_music.to(device)

            # Model forward pass -> estimates (B, N_src, C, T)
            estimates = model(mixture_input)
            
            # Calculate loss using the provided loss function instance
            batch_loss, batch_speech_loss, batch_music_loss = loss_fn(estimates, torch.stack([target_speech, target_music], dim=1))

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_losses_total[-1] += batch_loss.item() * mixture_input.shape[0]
            train_speech_l1_loss[-1] += batch_speech_loss.item() * mixture_input.shape[0]
            train_music_l1_loss[-1] += batch_music_loss.item() * mixture_input.shape[0]

            # Calculate SI-SNR for training monitoring (not used for training)
            with torch.no_grad():
                # Get speech and music estimates
                speech_est = estimates[:, loss_fn.speech_target_index]  # (B, C, T)
                music_est = estimates[:, loss_fn.music_target_index]    # (B, C, T)
                
                # Convert to mono if needed
                speech_est_mono = speech_est.squeeze(1) if speech_est.shape[1] == 1 else torch.mean(speech_est, dim=1)
                speech_tgt_mono = target_speech.squeeze(1) if target_speech.shape[1] == 1 else torch.mean(target_speech, dim=1)
                music_est_mono = music_est.squeeze(1) if music_est.shape[1] == 1 else torch.mean(music_est, dim=1)
                music_tgt_mono = target_music.squeeze(1) if target_music.shape[1] == 1 else torch.mean(target_music, dim=1)
                
                # Calculate batch SI-SNR
                batch_speech_sisnr = 0.0
                batch_music_sisnr = 0.0
                for i in range(speech_est_mono.shape[0]):
                    batch_speech_sisnr += (-si_snr_loss(speech_est_mono[i], speech_tgt_mono[i])).item()
                    batch_music_sisnr += (-si_snr_loss(music_est_mono[i], music_tgt_mono[i])).item()
                
                batch_speech_sisnr /= speech_est_mono.shape[0]
                batch_music_sisnr /= music_est_mono.shape[0]
                
                # Accumulate for epoch average
                train_speech_si_snr[-1] += batch_speech_sisnr * mixture_input.shape[0]
                train_music_si_snr[-1] += batch_music_sisnr * mixture_input.shape[0]

            # Update progress bar with current metrics
            train_pbar.set_postfix(
                loss=f"{batch_loss.item():.4f}",
                speech_L1=f"{batch_speech_loss.item():.4f}",
                music_L1=f"{batch_music_loss.item():.4f}",
                speech_SISNR=f"{batch_speech_sisnr:.2f}dB",
                music_SISNR=f"{batch_music_sisnr:.2f}dB"
            )

        # Train summary calculation
        total_train_samples = min(len(train_loader), args.max_batches) * train_loader.batch_size if args.max_batches > 0 else len(train_dataset)
        if total_train_samples > 0:
            train_losses_total[-1] /= total_train_samples
            train_speech_l1_loss[-1] /= total_train_samples
            train_music_l1_loss[-1] /= total_train_samples
            train_speech_si_snr[-1] /= total_train_samples
            train_music_si_snr[-1] /= total_train_samples
            # Calculate overall SI-SNR as average of speech and music
            train_overall_si_snr[-1] = (train_speech_si_snr[-1] + train_music_si_snr[-1]) / 2.0

        # Calculate validation loss and metrics
        val_metrics = evaluate(model, val_loader, loss_fn, eval_metrics, device, max_batches=args.max_batches, sample_rate=args.sr)
        val_loss = torch.tensor(val_metrics["Loss_Total"])
        val_speech_loss = torch.tensor(val_metrics["Loss_Speech(L1)"])
        val_music_loss = torch.tensor(val_metrics["Loss_Music(L1)"])
        val_losses_total.append(val_loss.item())
        val_speech_l1_loss.append(val_speech_loss.item())
        val_music_l1_loss.append(val_music_loss.item())
        
        # Validation set size (accounting for max_batches)
        val_samples = min(len(val_loader), args.max_batches) * val_loader.batch_size if args.max_batches > 0 else len(val_dataset)
        
        # Track SI-SNR metrics from validation
        val_speech_si_snr.append(val_metrics["Speech_SI-SNR_Eval"])
        val_music_si_snr.append(val_metrics["Music_SI-SNR_Eval"])
        
        # Initialize progress bar for validation (using max_batches)
        total_val_batches = min(len(val_loader), args.max_batches) if args.max_batches > 0 else len(val_loader)
        val_pbar = tqdm(val_loader, desc=f"Validation", leave=False, total=total_val_batches)
        val_pbar.set_postfix(
            loss=f"{val_loss.item():.4f}",
            speech_L1=f"{val_speech_loss.item():.4f}",
            music_L1=f"{val_music_loss.item():.4f}",
            speech_SISNR=f"{val_speech_si_snr[-1]:.2f}dB",
            music_SISNR=f"{val_music_si_snr[-1]:.2f}dB"
        )

        # Log training and validation losses
        writer.add_scalar('Loss/Train/Total', train_losses_total[-1], epoch)
        writer.add_scalar('Loss/Train/Speech_L1', train_speech_l1_loss[-1], epoch)
        writer.add_scalar('Loss/Train/Music_L1', train_music_l1_loss[-1], epoch)
        writer.add_scalar('SI-SNR/Train/Speech', train_speech_si_snr[-1], epoch)
        writer.add_scalar('SI-SNR/Train/Music', train_music_si_snr[-1], epoch)
        writer.add_scalar('SI-SNR/Train/Overall', train_overall_si_snr[-1], epoch)

        # Log validation loss and metrics
        writer.add_scalar('Loss/Val/Total', val_loss.item(), epoch)
        writer.add_scalar('Loss/Val/Speech_L1', val_speech_loss.item(), epoch)
        writer.add_scalar('Loss/Val/Music_L1', val_music_loss.item(), epoch)
        writer.add_scalar('SI-SNR/Val/Speech', val_speech_si_snr[-1], epoch)
        writer.add_scalar('SI-SNR/Val/Music', val_music_si_snr[-1], epoch)
        if len(val_overall_si_snr) > 0 and epoch < len(val_overall_si_snr):
            writer.add_scalar('SI-SNR/Val/Overall', val_overall_si_snr[epoch], epoch)

        # Save checkpoint for every epoch
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss.item(),
                "best_val_loss": best_val_loss,
                "train_losses_total": train_losses_total,
                "val_losses_total": val_losses_total,
                "train_speech_l1_loss": train_speech_l1_loss,
                "val_speech_l1_loss": val_speech_l1_loss,  # Save the whole list
                "train_music_l1_loss": train_music_l1_loss,
                "val_music_l1_loss": val_music_l1_loss,    # Save the whole list
                "val_pesq_hist": val_pesq_hist,
                "val_stoi_hist": val_stoi_hist,
                "train_speech_si_snr": train_speech_si_snr,
                "val_speech_si_snr": val_speech_si_snr,
                "train_music_si_snr": train_music_si_snr,
                "val_music_si_snr": val_music_si_snr,
                "train_overall_si_snr": train_overall_si_snr,
                "val_overall_si_snr": val_overall_si_snr,
                "val_metrics": val_metrics,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

        # Track and save best model separately
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # Save best model checkpoint
            best_checkpoint_path = output_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss.item(),
                    "best_val_loss": best_val_loss,
                    "train_losses_total": train_losses_total,
                    "val_losses_total": val_losses_total,
                    "train_speech_l1_loss": train_speech_l1_loss,
                    "val_speech_l1_loss": val_speech_l1_loss,
                    "train_music_l1_loss": train_music_l1_loss,
                    "val_music_l1_loss": val_music_l1_loss,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            print(f"✓ New best model saved to {best_checkpoint_path} (val_loss: {val_loss.item():.4f})")

        # Generate and save plots for current epoch
        fig = plot_losses(train_losses_total, val_losses_total, output_dir / f'loss_plot_epoch_{epoch+1}.png', {
            'speech_l1': train_speech_l1_loss,
            'music_l1': train_music_l1_loss
        }, {
            'speech_l1': val_speech_l1_loss,
            'music_l1': val_music_l1_loss
        })
        writer.add_figure('Plots/loss', fig, epoch+1)
        
        fig = plot_si_snr(
            list(range(1, epoch+2)), 
            train_speech_si_snr, train_music_si_snr, 
            val_speech_si_snr, val_music_si_snr, 
            train_overall_si_snr, val_overall_si_snr, 
            output_dir / f'si_snr_plot_epoch_{epoch+1}.png'
        )
        writer.add_figure('Plots/si_snr', fig, epoch+1)

        # --- Save Audio Sample and Spectrograms ---
        if save_sample_flag:
            model.eval()
            with torch.no_grad():
                mixture_tensor = fixed_example_mixture.clone().to(device) # Use stored sample
                # Model expects (B, C, T), dataset gives (C, T) -> add Batch dim
                if mixture_tensor.ndim == 2: mixture_tensor = mixture_tensor.unsqueeze(0) # Add Batch dim
                print(f"Saving sample audio and spectrograms for epoch {epoch+1}, input shape: {mixture_tensor.shape}")

                # Get separated sources
                estimates = model(mixture_tensor)

                # Save audio files
                save_audio(estimates[:, 0], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_speech_est.wav'), args.sr)
                save_audio(estimates[:, 1], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_music_est.wav'), args.sr)

                # Save spectrograms
                save_spectrogram(
                    estimates[:, 0], 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Estimated Speech (Epoch {epoch+1})"
                )
                save_spectrogram(
                    estimates[:, 1], 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Estimated Music (Epoch {epoch+1})"
                )

                # Log audio to TensorBoard
                try:
                    # Make sure tensor is in the right format (batch-first) and on CPU
                    speech_est_cpu = estimates[:, 0].cpu()
                    music_est_cpu = estimates[:, 1].cpu()

                    # Original mixture for comparison
                    orig_mixture_cpu = fixed_example_mixture.cpu()
                    if orig_mixture_cpu.ndim == 2:  # Add batch dimension if needed
                        orig_mixture_cpu = orig_mixture_cpu.unsqueeze(0)

                    # Log original ground truth
                    writer.add_audio(
                        'Audio/original_mixture', 
                        orig_mixture_cpu.squeeze(),
                        global_step=epoch+1, 
                        sample_rate=args.sr
                    )

                    # Log separated audio
                    writer.add_audio(
                        'Audio/estimated_speech', 
                        speech_est_cpu.squeeze(), 
                        global_step=epoch+1, 
                        sample_rate=args.sr
                    )
                    writer.add_audio(
                        'Audio/estimated_music', 
                        music_est_cpu.squeeze(), 
                        global_step=epoch+1, 
                        sample_rate=args.sr
                    )

                    # Also log spectrograms as images
                    # Use the paths of the saved images
                    speech_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'))
                    music_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'))

                    writer.add_image('Spectrograms/speech', speech_spec_img, epoch+1, dataformats='HWC')
                    writer.add_image('Spectrograms/music', music_spec_img, epoch+1, dataformats='HWC')
                except Exception as e:
                    print(f"Warning: Could not log audio to TensorBoard: {e}")

    # --- Plotting after training ---
    epochs_ran = list(range(1, len(train_losses_total) + 1)) # Use actual length of history
    if len(train_losses_total) == len(epochs_ran): # Ensure history length matches epochs
        fig = plot_losses(
            train_losses_total,
            val_losses_total,
            output_dir / "loss_plot.png",
            {
                "speech_l1": train_speech_l1_loss,
                "music_l1": train_music_l1_loss,
            },
            {
                "speech_l1": val_speech_l1_loss,
                "music_l1": val_music_l1_loss,
            }
        )
        writer.add_figure('Plots/loss', fig, 0)
        
        fig = plot_si_snr(epochs_ran, train_speech_si_snr, train_music_si_snr, 
                            val_speech_si_snr, val_music_si_snr, 
                            train_overall_si_snr, val_overall_si_snr, 
                            output_dir / 'si_snr_plot.png')
        writer.add_figure('Plots/si_snr', fig, 0)
    else:
        print("Warning: History length is zero, skipping plotting.")

    # Get final evaluation metrics by running one more evaluation on the validation set
    final_metrics = evaluate(model, val_loader, loss_fn, eval_metrics, device, desc="Final evaluation", max_batches=args.max_batches, sample_rate=args.sr)
    
    # Print final metrics
    print_metrics_table(final_metrics, title="Final Evaluation Metrics")
    
    # Log final metrics to TensorBoard
    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, (int, float)):
            writer.add_scalar(f'Final/{metric_name}', metric_value, 0)
    
    # Make a bar chart of final metrics for TensorBoard
    try:
        # Filter for key metrics (SI-SNR, PESQ, STOI)
        si_snr_metrics = {k: v for k, v in final_metrics.items() if "SI-SNR" in k and isinstance(v, (int, float))}
        other_metrics = {k: v for k, v in final_metrics.items() if any(m in k for m in ["PESQ", "STOI"]) and isinstance(v, (int, float))}
        
        # Plot SI-SNR metrics as a bar chart
        if si_snr_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(list(si_snr_metrics.keys()), list(si_snr_metrics.values()))
            ax.set_ylabel('dB')
            ax.set_title('Final SI-SNR Metrics')
            for i, v in enumerate(si_snr_metrics.values()):
                ax.text(i, v + 0.1, f"{v:.2f} dB", ha='center')
            plt.tight_layout()
            writer.add_figure('Final/SI-SNR_Metrics', fig, 0)
        
        # Plot other metrics as a bar chart if they exist
        if other_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(list(other_metrics.keys()), list(other_metrics.values()))
            ax.set_title('Final PESQ/STOI Metrics')
            for i, v in enumerate(other_metrics.values()):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
            plt.tight_layout()
            writer.add_figure('Final/Other_Metrics', fig, 0)
    except Exception as e:
        print(f"Warning: Could not create final metrics visualization: {e}")
    
    # Close TensorBoard writer
    writer.close()
    
    return final_metrics

def main():
    """
    Main entry point for training HTDemucs models from command line.
    Example usage:
        python train.py --root_dir data --save_dir checkpoints --sr 8000 --batch_size 4 --epochs 100
    """
    parser = argparse.ArgumentParser(description="Train HTDemucs for Music/Speech Separation with L1 Loss")

    # --- Data Arguments ---
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--segment', type=float, default=1.0, help='Duration of audio segments in seconds')
    parser.add_argument('--sr', type=int, default=8000, help='Sample rate (PESQ requires 8k or 16k)')

    # --- Training Arguments ---
    parser.add_argument('--save_dir', type=str, default='checkpoints_htdemucs', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N batches')
    parser.add_argument('--max_batches', type=int, default=0, help='Maximum number of batches per epoch (0 for all)')

    # --- Model Hyperparameters for HTDemucs ---
    parser.add_argument('--n_fft', type=int, default=4096, help='FFT size for STFT')
    parser.add_argument('--hop_length', type=int, default=1024, help='Hop length for STFT')
    parser.add_argument('--channels', type=int, default=48, help='Initial number of hidden channels')
    parser.add_argument('--growth', type=int, default=2, help='Increase channels by this factor each layer')
    parser.add_argument('--depth', type=int, default=6, help='Number of layers in the encoder and decoder')
    parser.add_argument('--kernel_size', type=int, default=8, help='Kernel size for encoder and decoder')
    parser.add_argument('--stride', type=int, default=4, help='Stride for encoder and decoder')
    
    # --- Eval/Debug Arguments ---
    parser.add_argument('--sample_index', type=int, default=-1, help='Index of validation sample to save audio for (-1 for random)')
    parser.add_argument('--test_samples', type=int, default=20, help='Number of test samples to save during final evaluation')
    parser.add_argument('--test_samples_random', action='store_true', help='Use random samples for test audio instead of sequential')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only run test evaluation')
    parser.add_argument('--test_checkpoint', type=str, default='', help='Specific checkpoint to use for testing (if empty, uses best checkpoint)')

    args = parser.parse_args()

    # --- Basic Validation ---
    if args.sr not in [8000, 16000]:
        raise ValueError("PESQ metric requires sr=8000 or sr=16000")
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
        
    # Print summary of arguments
    print("=" * 70)
    print("HTDemucs Training Configuration:")
    print("=" * 70)
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("=" * 70)

    # Setup output directory
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for test set
    root_path = Path(args.root_dir)
    has_test_set = False
    
    for test_dir_name in ["test", "testing"]:
        if (root_path / test_dir_name).exists():
            has_test_set = True
            test_split_name = test_dir_name
            break
    
    if not has_test_set:
        print("Warning: No test directory found. Will use validation set for final evaluation.")
        test_split_name = "val" if (root_path / "val").exists() else "validation"
    
    # If not skipping training, run the training process
    final_metrics = {}
    if not args.skip_train:
        print("\n=== STARTING TRAINING ===\n")
        final_metrics = train(args)
    
    # Setup for test evaluation
    print("\n=== STARTING TEST EVALUATION ===\n")
    
    # Load model
    # Define sources for HTDemucs
    sources = ["speech", "music"]
    
    # Initialize HTDemucs model
    model = HTDemucs(
        sources=sources,
        audio_channels=1,  # Force mono for consistency
        samplerate=args.sr,
        segment=args.segment,
    ).to(device)
    
    # Load the best checkpoint
    if args.skip_train:
        if args.test_checkpoint:
            checkpoint_path = Path(args.test_checkpoint)
        else:
            checkpoint_info = find_latest_checkpoint(output_dir)
            if checkpoint_info:
                checkpoint_path, _ = checkpoint_info
            else:
                raise ValueError("No checkpoint found for testing. Please provide --test_checkpoint.")
                
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and dataloader
    test_dataset = FolderTripletDataset(root_path, split=test_split_name, segment=args.segment, sr=args.sr)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(test_dataset)}")
    
    # Setup metrics for test evaluation
    # --- Metrics for Test Evaluation ---
    pesq_mode = 'wb' if args.sr == 16000 else 'nb'
    pesq_key = f"PESQ-{pesq_mode.upper()}"

    eval_metric_list = {} # Initialize empty
    
    # Use the same metrics setup as in the train function
    try:
        eval_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    except (AttributeError, ImportError):
        try:
            from torchmetrics.functional.audio.sdr import scale_invariant_signal_noise_ratio
            class SISNRMetric(torchmetrics.Metric):
                def __init__(self):
                    super().__init__()
                    self.add_state("sisnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            score = scale_invariant_signal_noise_ratio(pred, tgt)
                            self.sisnr_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"SI-SNR calculation failed: {e}")
                
                def compute(self):
                    return self.sisnr_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["SI-SNR"] = SISNRMetric().to(device)
        except (AttributeError, ImportError):
            print(f"Warning: SI-SNR metric not available. Using custom implementation.")
            class CustomSISNRMetric(torchmetrics.Metric):
                def __init__(self):
                    super().__init__()
                    self.add_state("sisnr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            score = -si_snr_loss(pred, tgt).mean()
                            self.sisnr_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"Custom SI-SNR calculation failed: {e}")
                
                def compute(self):
                    return self.sisnr_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["SI-SNR"] = CustomSISNRMetric().to(device)

    if pesq_mode:
        try:
            eval_metric_list[pesq_key] = torchmetrics.audio.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
        except (AttributeError, ImportError):
            try:
                from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
                class PESQMetric(torchmetrics.Metric):
                    def __init__(self, fs, mode):
                        super().__init__()
                        self.fs = fs
                        self.mode = mode
                        self.add_state("pesq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                    
                    def update(self, preds, target):
                        for pred, tgt in zip(preds, target):
                            try:
                                score = perceptual_evaluation_speech_quality(pred, tgt, self.fs, self.mode)
                                self.pesq_sum += score
                                self.total += 1
                            except Exception as e:
                                print(f"PESQ calculation failed: {e}")
                    
                    def compute(self):
                        return self.pesq_sum / self.total if self.total > 0 else torch.tensor(0.0)
                
                eval_metric_list[pesq_key] = PESQMetric(args.sr, pesq_mode).to(device)
            except (AttributeError, ImportError):
                print(f"Warning: PESQ metric not available. Skipping PESQ evaluation.")
    
    try:
        eval_metric_list["STOI"] = torchmetrics.audio.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
    except (AttributeError, ImportError):
        try:
            from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
            class STOIMetric(torchmetrics.Metric):
                def __init__(self, fs, extended=False):
                    super().__init__()
                    self.fs = fs
                    self.extended = extended
                    self.add_state("stoi_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
                    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
                
                def update(self, preds, target):
                    for pred, tgt in zip(preds, target):
                        try:
                            score = short_time_objective_intelligibility(pred, tgt, self.fs, self.extended)
                            self.stoi_sum += score
                            self.total += 1
                        except Exception as e:
                            print(f"STOI calculation failed: {e}")
                
                def compute(self):
                    return self.stoi_sum / self.total if self.total > 0 else torch.tensor(0.0)
            
            eval_metric_list["STOI"] = STOIMetric(args.sr, extended=False).to(device)
        except (AttributeError, ImportError):
            print(f"Warning: STOI metric not available. Skipping STOI evaluation.")
    
    eval_metrics = torchmetrics.MetricCollection(eval_metric_list).to(device)
    
    # Setup loss function
    loss_fn = HTDemucsL1Loss(
        speech_target_index=0,  # speech is first source
        music_target_index=1     # music is second source
    ).to(device)
    
    # Set up test output directories
    test_output_dir = output_dir / "test_results"
    test_output_dir.mkdir(exist_ok=True)
    test_audio_dir = test_output_dir / "audio_samples"
    test_audio_dir.mkdir(exist_ok=True)
    test_spec_dir = test_output_dir / "spectrograms"
    test_spec_dir.mkdir(exist_ok=True)
    
    # Create TensorBoard writer for test results
    test_tb_dir = test_output_dir / "tensorboard"
    test_tb_dir.mkdir(exist_ok=True)
    test_writer = SummaryWriter(log_dir=test_tb_dir)
    
    # Evaluate model on test set
    model.eval()
    test_metrics = evaluate(model, test_loader, loss_fn, eval_metrics, device, desc="Test Evaluation", max_batches=args.max_batches, sample_rate=args.sr)
    
    # Print test metrics
    print_metrics_table(test_metrics, title="Test Set Evaluation Metrics")
    
    # Log test metrics to TensorBoard
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, (int, float)):
            test_writer.add_scalar(f'Test/{metric_name}', metric_value, 0)
    
    # Save test metrics to file
    test_metrics_file = test_output_dir / "test_metrics.txt"
    try:
        with open(test_metrics_file, "w") as f:
            f.write("HTDemucs Test Evaluation Metrics\n")
            f.write("=" * 40 + "\n")
            for k, v in test_metrics.items():
                if isinstance(v, (int, float)):
                    if "SI-SNR" in k or "SDR" in k:
                        f.write(f"{k}: {v:.2f} dB\n")
                    elif "PESQ" in k:
                        f.write(f"{k}: {v:.2f} / 5.0\n")
                    elif "STOI" in k:
                        f.write(f"{k}: {v:.3f} (0-1)\n")
                    else:
                        f.write(f"{k}: {v:.4f}\n")
            f.write("\n")
            f.write(f"Tested on {len(test_dataset)} samples\n")
            f.write(f"Sample rate: {args.sr} Hz\n")
            
        print(f"\nTest metrics saved to: {test_metrics_file}")
    except Exception as e:
        print(f"Warning: Could not save test metrics to file: {e}")
    
    # Save example predictions from test set
    print("\nGenerating example predictions from test set...")
    
    # Decide which samples to use - select samples that are spread out across the dataset
    num_test_samples = min(args.test_samples, len(test_dataset))
    
    if args.test_samples_random:
        # Spread samples evenly across the dataset instead of completely random
        dataset_size = len(test_dataset)
        
        if num_test_samples <= 1:
            test_indices = [random.randint(0, dataset_size - 1)]
        else:
            # Calculate step size to spread samples
            step = dataset_size // num_test_samples
            
            # Add some randomization within each segment
            test_indices = []
            for i in range(num_test_samples):
                # Define segment boundaries
                start = i * step
                end = min((i + 1) * step - 1, dataset_size - 1)
                
                # Pick a random sample within this segment
                if start == end:
                    index = start
                else:
                    index = random.randint(start, end)
                    
                test_indices.append(index)
                
            # Shuffle the indices to avoid processing in sequential order
            random.shuffle(test_indices)
    else:
        # Use evenly spaced samples rather than just the first N
        if num_test_samples <= 1:
            test_indices = [0]  # Just use the first sample
        else:
            # Calculate step size for even spacing
            step = len(test_dataset) // num_test_samples
            test_indices = [i * step for i in range(num_test_samples)]
    
    print(f"Selected {len(test_indices)} test samples at indices: {test_indices}")
    
    # Process each selected test sample
    with torch.no_grad():
        for idx, sample_idx in enumerate(test_indices):
            mixture, target_speech, target_music = test_dataset[sample_idx]
            
            # Create sample directory
            sample_dir = test_audio_dir / f"sample_{idx+1}"
            sample_dir.mkdir(exist_ok=True)
            spec_dir = test_spec_dir / f"sample_{idx+1}"
            spec_dir.mkdir(exist_ok=True)
            
            # Convert to batch and move to device
            mixture_batch = mixture.unsqueeze(0).to(device)
            
            # Generate prediction
            estimated_sources = model(mixture_batch)
            
            # Save ground truth audio
            save_audio(mixture, sample_dir / "mixture.wav", args.sr)
            save_audio(target_speech, sample_dir / "speech_target.wav", args.sr)
            save_audio(target_music, sample_dir / "music_target.wav", args.sr)
            
            # Save predicted audio
            save_audio(estimated_sources[0, 0], sample_dir / "speech_predicted.wav", args.sr)
            save_audio(estimated_sources[0, 1], sample_dir / "music_predicted.wav", args.sr)
            
            # Save spectrograms
            save_spectrogram(mixture, spec_dir / "mixture.png", args.sr, args.n_fft, args.hop_length, "Mixture")
            save_spectrogram(target_speech, spec_dir / "speech_target.png", args.sr, args.n_fft, args.hop_length, "Target Speech")
            save_spectrogram(target_music, spec_dir / "music_target.png", args.sr, args.n_fft, args.hop_length, "Target Music")
            save_spectrogram(estimated_sources[0, 0], spec_dir / "speech_predicted.png", args.sr, args.n_fft, args.hop_length, "Predicted Speech")
            save_spectrogram(estimated_sources[0, 1], spec_dir / "music_predicted.png", args.sr, args.n_fft, args.hop_length, "Predicted Music")
            
            # Calculate and save metrics for this sample
            try:
                # Speech SI-SNR
                speech_sisnr = -si_snr_loss(estimated_sources[0, 0].cpu(), target_speech).item()
                
                # Music SI-SNR
                music_sisnr = -si_snr_loss(estimated_sources[0, 1].cpu(), target_music).item()
                
                # Write metrics to file
                with open(sample_dir / "metrics.txt", "w") as f:
                    f.write(f"Speech SI-SNR: {speech_sisnr:.2f} dB\n")
                    f.write(f"Music SI-SNR: {music_sisnr:.2f} dB\n")
                
                # Log to TensorBoard
                test_writer.add_audio(f"Sample_{idx+1}/Mixture", mixture.unsqueeze(0), global_step=0, sample_rate=args.sr)
                test_writer.add_audio(f"Sample_{idx+1}/Target_Speech", target_speech.unsqueeze(0), global_step=0, sample_rate=args.sr)
                test_writer.add_audio(f"Sample_{idx+1}/Target_Music", target_music.unsqueeze(0), global_step=0, sample_rate=args.sr)
                test_writer.add_audio(f"Sample_{idx+1}/Predicted_Speech", estimated_sources[0, 0].cpu().unsqueeze(0), global_step=0, sample_rate=args.sr)
                test_writer.add_audio(f"Sample_{idx+1}/Predicted_Music", estimated_sources[0, 1].cpu().unsqueeze(0), global_step=0, sample_rate=args.sr)
                
                # Add metrics to TensorBoard
                test_writer.add_scalar(f"Sample_{idx+1}/Speech_SISNR", speech_sisnr, 0)
                test_writer.add_scalar(f"Sample_{idx+1}/Music_SISNR", music_sisnr, 0)
                
                print(f"Processed test sample {idx+1}/{num_test_samples} (SI-SNR: Speech {speech_sisnr:.2f}dB, Music {music_sisnr:.2f}dB)")
            except Exception as e:
                print(f"Error processing metrics for sample {idx+1}: {e}")
    
    # Close test writer
    test_writer.close()
    
    print(f"\nTest evaluation completed. Results saved to {test_output_dir}")
    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
