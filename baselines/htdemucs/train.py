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

from demucs.htdemucs import HTDemucs 
from windowed_audio_datasets import FolderTripletDataset
import torchmetrics


#######
# Based on the code from https://github.com/facebookresearch/demucs
#######

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
        
        # --- Handle Channel Dimension (Assuming Mono for Loss Calculation) ---
        if estimates.shape[2] == 1:
            estimates_mono = estimates.squeeze(2) # (B, N_src, T)
            targets_mono = targets.squeeze(2)     # (B, N_src, T)
        else: # Average if multi-channel
            print("Warning: Multi-channel detected, averaging channels for loss calculation.")
            estimates_mono = torch.mean(estimates, dim=2)
            targets_mono = torch.mean(targets, dim=2)
        
        # Extract speech and music components
        speech_est = estimates_mono[:, self.speech_target_index]
        speech_tgt = targets_mono[:, self.speech_target_index]
        music_est = estimates_mono[:, self.music_target_index]
        music_tgt = targets_mono[:, self.music_target_index]
        
        # Calculate L1 losses
        speech_loss = self.loss_fn(speech_est, speech_tgt)
        music_loss = self.loss_fn(music_est, music_tgt)
        
        # Total loss is the sum of both source losses
        total_loss = speech_loss + music_loss
        
        return total_loss, speech_loss, music_loss

# --- Evaluation Function (Modified for HTDemucs L1 Loss & Max Batches) ---
def evaluate(model, dataloader, loss_fn_eval, metrics_dict, device, desc="Evaluating", max_batches=0, sample_rate=8000):
    """ Evaluate HTDemucs model on a dataloader """
    model.eval()
    # Updated loss accumulators
    total_loss_l1_speech = 0.0
    total_loss_l1_music = 0.0
    num_samples = 0

    # Reset metrics
    for metric in metrics_dict.values():
        metric.reset()

    # Accumulators for per-source SI-SNR metrics (calculated manually for reporting)
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
            _, batch_loss_l1_speech, batch_loss_l1_music = loss_fn_eval(estimates, targets)
            
            # Accumulate L1 losses
            total_loss_l1_speech += batch_loss_l1_speech.item() * batch_size
            total_loss_l1_music += batch_loss_l1_music.item() * batch_size
            
            # --- Calculate SI-SNR metrics (Manual for reporting, TorchMetrics below) ---
            speech_est_mono = estimates[:, loss_fn_eval.speech_target_index].squeeze(2) # (B, T)
            music_est_mono = estimates[:, loss_fn_eval.music_target_index].squeeze(2)   # (B, T)
            speech_tgt_mono = targets[:, loss_fn_eval.speech_target_index].squeeze(2).to(device) # (B, T)
            music_tgt_mono = targets[:, loss_fn_eval.music_target_index].squeeze(2).to(device)   # (B, T)
            
            batch_speech_sisnr = -torch.mean(si_snr_loss_manual(speech_est_mono, speech_tgt_mono)).item()
            batch_music_sisnr = -torch.mean(si_snr_loss_manual(music_est_mono, music_tgt_mono)).item()
            accumulated_speech_sisnr_metric += batch_speech_sisnr * batch_size
            accumulated_music_sisnr_metric += batch_music_sisnr * batch_size
            
            # Update progress bar with current SI-SNR metrics
            if num_samples > 0:
                eval_pbar.set_postfix(
                    speech=f"{accumulated_speech_sisnr_metric / num_samples:.2f}dB",
                    music=f"{accumulated_music_sisnr_metric / num_samples:.2f}dB"
                )

            # --- Calculate TorchMetrics SI-SNR, PESQ, STOI ---
            if "SI-SNR" in metrics_dict:
                try:
                    # Make sure shapes are correct (expects shape [B, C, T] or [B, N_src, C, T])
                    metrics_dict["SI-SNR"].update(estimates.to(device), targets.to(device))
                except Exception as e:
                    print(f"Warning: SI-SNR calculation failed: {e}")

            # Update PESQ/STOI for speech only (using mono versions)
            min_len = min(speech_est_mono.shape[-1], speech_tgt_mono.shape[-1])
            speech_est_trim = speech_est_mono[..., :min_len]
            speech_tgt_trim = speech_tgt_mono[..., :min_len]

            valid_indices = torch.sum(torch.abs(speech_tgt_trim), dim=-1) > 1e-5
            if torch.any(valid_indices):
                estimates_valid = speech_est_trim[valid_indices].to(device)
                targets_valid = speech_tgt_trim[valid_indices].to(device)
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
        avg_loss_l1_speech = total_loss_l1_speech / num_samples
        avg_loss_l1_music = total_loss_l1_music / num_samples
        
        # Update results dictionary with L1 loss names
        results["Loss_Speech(L1)"] = avg_loss_l1_speech
        results["Loss_Music(L1)"] = avg_loss_l1_music
        
        # Total loss is sum of L1 losses
        results["Loss_Total"] = avg_loss_l1_speech + avg_loss_l1_music
        
        # Add separate speech and music SI-SNR metric values (from manual calculation)
        results["Speech_SI-SNR_Eval"] = accumulated_speech_sisnr_metric / num_samples
        results["Music_SI-SNR_Eval"] = accumulated_music_sisnr_metric / num_samples
        
    else:
        # Handle case where no samples were processed
        results = {}
        results["SI-SNR"] = 0.0
        results["Loss_Speech(L1)"] = 0.0
        results["Loss_Music(L1)"] = 0.0
        results["Loss_Total"] = 0.0
        results["Speech_SI-SNR_Eval"] = 0.0
        results["Music_SI-SNR_Eval"] = 0.0
        pesq_key = f"PESQ-{'WB' if sample_rate == 16000 else 'NB'}"
        results[pesq_key] = 0.0
        results["STOI"] = 0.0

    return results

# --- Helper Functions --- 
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

def plot_si_snr(epochs, train_speech, train_music, val_speech, val_music, train_overall, val_overall, save_path):
    """Plot SI-SNR metrics for training and validation and save the figure"""
    if not epochs or not train_speech or not val_speech:
        print("Warning: SI-SNR history is empty or incomplete, skipping plotting.")
        return None # Return None instead of fig
    # Ensure lists have same length as epochs for plotting
    min_len = len(epochs)
    train_speech = train_speech[:min_len] if train_speech else [0]*min_len
    train_music = train_music[:min_len] if train_music else [0]*min_len
    val_speech = val_speech[:min_len] if val_speech else [0]*min_len
    val_music = val_music[:min_len] if val_music else [0]*min_len
    train_overall = train_overall[:min_len] if train_overall else [0]*min_len
    val_overall = val_overall[:min_len] if val_overall else [0]*min_len
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(12, 15))
    
    # Speech SI-SNR subplot
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_speech, 'b-o', markersize=4, label='Train Speech SI-SNR')
    plt.plot(epochs, val_speech, 'r-o', markersize=4, label='Val Speech SI-SNR')
    plt.title('Speech SI-SNR (dB)'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    if train_speech: # Check if list is not empty
        max_train_idx = np.argmax(train_speech); 
        plt.annotate(f'Max: {train_speech[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_speech[max_train_idx]), ha='center', va='bottom')
    if val_speech: # Check if list is not empty
        max_val_idx = np.argmax(val_speech);
        plt.annotate(f'Max: {val_speech[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_speech[max_val_idx]), ha='center', va='bottom', color='red')

    # Music SI-SNR subplot
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_music, 'b-o', markersize=4, label='Train Music SI-SNR')
    plt.plot(epochs, val_music, 'r-o', markersize=4, label='Val Music SI-SNR')
    plt.title('Music SI-SNR (dB)'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    if train_music:
        max_train_idx = np.argmax(train_music); 
        plt.annotate(f'Max: {train_music[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_music[max_train_idx]), ha='center', va='bottom')
    if val_music:
        max_val_idx = np.argmax(val_music);
        plt.annotate(f'Max: {val_music[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_music[max_val_idx]), ha='center', va='bottom', color='red')
    
    # Overall SI-SNR subplot (PIT)
    plt.subplot(3, 1, 3)
    if train_overall and val_overall and any(train_overall) and any(val_overall): # Check if lists exist and are not all zeros
        plt.plot(epochs, train_overall, 'b-o', markersize=4, label='Train Overall SI-SNR (PIT)')
        plt.plot(epochs, val_overall, 'r-o', markersize=4, label='Val Overall SI-SNR (PIT)')
        plt.title('Overall SI-SNR (PIT) (dB)'); plt.xlabel('Epochs'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
        max_train_idx = np.argmax(train_overall); 
        plt.annotate(f'Max: {train_overall[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_overall[max_train_idx]), ha='center', va='bottom')
        max_val_idx = np.argmax(val_overall);
        plt.annotate(f'Max: {val_overall[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_overall[max_val_idx]), ha='center', va='bottom', color='red')
    else:
        plt.text(0.5, 0.5, 'No Overall SI-SNR (PIT) Data Available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.xlabel('Epochs')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path); 
    plt.close(); 
    print(f"SI-SNR plot saved to {save_path}")
    return fig

def train_sample_index(dataset, sample_index=-1):
     """Helper function to get a sample from a dataset."""
     if sample_index < 0 or sample_index >= len(dataset):
          sample_index = np.random.randint(0, len(dataset))
     mixture, target_speech, target_music = dataset[sample_index]
     return mixture, target_speech, target_music, sample_index

def si_snr_loss_manual(estimate, target, epsilon=1e-8):
    """ Calculates negative SI-SNR loss. Input shapes: (Batch, Time) """
    # Ensure inputs have a batch dimension
    if estimate.ndim == 1: estimate = estimate.unsqueeze(0)
    if target.ndim == 1: target = target.unsqueeze(0)

    # Ensure both tensors are on the same device
    device = estimate.device
    target = target.to(device)

    # Handle potential length mismatches
    min_len = min(estimate.shape[-1], target.shape[-1])
    estimate = estimate[..., :min_len]
    target = target[..., :min_len]

    # Zero mean adjustments
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # Calculate scale factor
    target_dot_estimate = torch.sum(target * estimate, dim=-1, keepdim=True)
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + epsilon
    alpha = target_dot_estimate / target_energy

    # Calculate scaled target and noise
    target_scaled = alpha * target
    noise = estimate - target_scaled

    # Calculate powers
    target_power = torch.sum(target_scaled**2, dim=-1) + epsilon
    noise_power = torch.sum(noise**2, dim=-1) + epsilon

    # Calculate negative SI-SNR
    neg_si_snr = -10 * torch.log10(target_power / noise_power)
    neg_si_snr = torch.nan_to_num(neg_si_snr, nan=40.0, posinf=40.0, neginf=-50.0).to(device)
    return neg_si_snr # Shape: (Batch,)

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

# Updated plot_losses function
def plot_losses(train_losses, val_losses, save_path, train_losses_components=None, val_losses_components=None):
    """
    Plot training and validation losses and save the figure.
    Plots Speech L1 and Music L1 components if provided.
    
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
    epochs_len = len(epochs)
    has_components = False
    if train_losses_components and val_losses_components:
        train_speech_l1 = train_losses_components.get('speech_l1', [0] * epochs_len)[:epochs_len]
        val_speech_l1 = val_losses_components.get('speech_l1', [0] * epochs_len)[:epochs_len]
        train_music_l1 = train_losses_components.get('music_l1', [0] * epochs_len)[:epochs_len]
        val_music_l1 = val_losses_components.get('music_l1', [0] * epochs_len)[:epochs_len]
        if any(train_speech_l1) or any(val_speech_l1) or any(train_music_l1) or any(val_music_l1):
            has_components = True
            
    num_plots = 1 + (2 if has_components else 0)
    fig = plt.figure(figsize=(12, 5 * num_plots))
    plot_idx = 1
    
    # Total loss plot
    plt.subplot(num_plots, 1, plot_idx)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Total Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Annotations for min values
    min_train_epoch = np.argmin(train_losses) + 1
    min_val_epoch = np.argmin(val_losses) + 1
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    plt.annotate(f'Min Train: {min_train_loss:.4f} (Ep {min_train_epoch})', 
                 xy=(min_train_epoch, min_train_loss), xytext=(min_train_epoch, min_train_loss + (max(train_losses)-min_train_loss)*0.1), ha='center')
    plt.annotate(f'Min Val: {min_val_loss:.4f} (Ep {min_val_epoch})', 
                 xy=(min_val_epoch, min_val_loss), xytext=(min_val_epoch, min_val_loss + (max(val_losses)-min_val_loss)*0.15), ha='center')
    plot_idx += 1
    
    if has_components:
        # Speech L1 Loss
        plt.subplot(num_plots, 1, plot_idx)
        plt.plot(epochs, train_speech_l1, 'b-', label='Train Speech L1 Loss')
        plt.plot(epochs, val_speech_l1, 'r-', label='Val Speech L1 Loss')
        plt.title('Speech L1 Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_idx += 1

        # Music L1 Loss
        plt.subplot(num_plots, 1, plot_idx)
        plt.plot(epochs, train_music_l1, 'b-', label='Train Music L1 Loss')
        plt.plot(epochs, val_music_l1, 'r-', label='Val Music L1 Loss')
        plt.title('Music L1 Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_idx += 1
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")
    return fig

# --- Training Setup ---

def train(args):
    """Main training loop (Modified for HTDemucs L1 Loss)"""
    # Define device
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs, using GPU 2")
        device = torch.device("cuda:2")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    root_path = Path(args.root_dir)
    # Use windowed dataset with segment_length_sec and hop_length_sec
    train_dataset = FolderTripletDataset(root_path, split="train", 
                                         segment_length_sec=args.segment, 
                                         hop_length_sec=args.hop_length_sec, 
                                         sr=args.sr)
    # Check for different validation folder naming conventions
    if (root_path / "val").exists(): val_split_name = "val"
    elif (root_path / "validation").exists(): val_split_name = "validation"
    elif (root_path / "valid").exists(): val_split_name = "valid"
    else: print("Warning: No validation folder found. Defaulting to 'val'"); val_split_name = "val"
    val_dataset = FolderTripletDataset(root_path, split=val_split_name, 
                                       segment_length_sec=args.segment, 
                                       hop_length_sec=args.hop_length_sec, 
                                       sr=args.sr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Setup directories --- (Keep as is)
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = output_dir / 'tensorboard_logs'
    tb_log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to {tb_log_dir}")
    print(f"Run 'tensorboard --logdir={tb_log_dir}' to view training progress")

    # --- Setup for saving audio samples and spectrograms --- (Keep as is, but note HTDemucs output)
    samples_dir = os.path.join(args.save_dir, 'audio_samples')
    spectrograms_dir = os.path.join(args.save_dir, 'spectrograms')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(spectrograms_dir, exist_ok=True)
    save_sample_flag = False
    try:
        fixed_example_mixture, fixed_example_speech, fixed_example_music, sample_idx_to_use = train_sample_index(val_dataset, args.sample_index)
        fixed_example_mixture = fixed_example_mixture.cpu()
        fixed_example_speech = fixed_example_speech.cpu()
        fixed_example_music = fixed_example_music.cpu()
        print(f"Using validation sample at index {sample_idx_to_use} for audio monitoring")
        # Save GT audio
        save_audio(fixed_example_mixture, os.path.join(samples_dir, 'gt_mixture.wav'), args.sr)
        save_audio(fixed_example_speech, os.path.join(samples_dir, 'gt_speech.wav'), args.sr)
        save_audio(fixed_example_music, os.path.join(samples_dir, 'gt_music.wav'), args.sr)
        # Save GT spectrograms
        save_spectrogram(fixed_example_mixture, os.path.join(spectrograms_dir, 'gt_mixture_spec.png'), args.sr, args.n_fft, args.hop_length, title="Ground Truth Mixture")
        save_spectrogram(fixed_example_speech, os.path.join(spectrograms_dir, 'gt_speech_spec.png'), args.sr, args.n_fft, args.hop_length, title="Ground Truth Speech")
        save_spectrogram(fixed_example_music, os.path.join(spectrograms_dir, 'gt_music_spec.png'), args.sr, args.n_fft, args.hop_length, title="Ground Truth Music")
        save_sample_flag = True
    except Exception as e:
        print(f"Could not get fixed validation sample for saving: {e}")

    # --- Model --- (Instantiate HTDemucs)
    sources = ["speech", "music"]
    model = HTDemucs(
        sources=sources,
        audio_channels=1,
        samplerate=args.sr,
        segment=args.segment,
    ).to(device)
    print(f"Model: HTDemucs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = HTDemucsL1Loss(speech_target_index=0, music_target_index=1).to(device)
    
    # --- Metrics for Validation/Test --- 
    pesq_mode = 'wb' if args.sr == 16000 else 'nb'
    pesq_key = f"PESQ-{pesq_mode.upper()}"
    eval_metric_list = {} # Initialize empty
    
    try: eval_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    except AttributeError: print("Warning: SI-SNR metric not found in torchmetrics.audio")
    if pesq_mode:
        try: eval_metric_list[pesq_key] = torchmetrics.audio.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
        except AttributeError: print(f"Warning: PESQ metric ({pesq_key}) not found in torchmetrics.audio")
    try: eval_metric_list["STOI"] = torchmetrics.audio.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
    except AttributeError: print("Warning: STOI metric not found in torchmetrics.audio")
    eval_metrics = torchmetrics.MetricCollection(eval_metric_list).to(device)

    # --- Training Loop --- 
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize History Lists 
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses_total, val_losses_total = [], []
    train_l1_s_loss, val_l1_s_loss = [], [] 
    train_l1_m_loss, val_l1_m_loss = [], [] 
    val_pesq_hist, val_stoi_hist = [], []
    train_speech_si_snr, val_speech_si_snr = [], [] # For positive SI-SNR metrics (reporting)
    train_music_si_snr, val_music_si_snr = [], [] 
    train_overall_si_snr, val_overall_si_snr = [], [] # For torchmetrics SI-SNR

    # --- Checkpoint Loading 
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
            # Load histories (updated keys)
            train_losses_total = checkpoint.get('train_losses_total', [])
            val_losses_total = checkpoint.get('val_losses_total', [])
            train_l1_s_loss = checkpoint.get('train_l1_s_loss', []) # New
            val_l1_s_loss = checkpoint.get('val_l1_s_loss', [])     # New
            train_l1_m_loss = checkpoint.get('train_l1_m_loss', []) # New
            val_l1_m_loss = checkpoint.get('val_l1_m_loss', [])     # New
            # Removed mel loss history loading
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
            start_epoch = 0; best_val_loss = float("inf")
            # Reset histories
            train_losses_total, val_losses_total = [], []; train_l1_s_loss, val_l1_s_loss = [], []
            train_l1_m_loss, val_l1_m_loss = [], []; val_pesq_hist, val_stoi_hist = [], []
            train_speech_si_snr, val_speech_si_snr = [], []; train_music_si_snr, val_music_si_snr = [], []
            train_overall_si_snr, val_overall_si_snr = [], []
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Epoch Loop ---
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        # --- Reset epoch loss accumulators (Updated for L1) ---
        epoch_total_loss = 0.0
        epoch_loss_l1_speech = 0.0
        epoch_loss_l1_music = 0.0
        # Accumulators for SI-SNR values (for reporting, calculated manually)
        epoch_neg_sisnr_speech_report = 0.0
        epoch_neg_sisnr_music_report = 0.0
        num_train_samples = 0
        
        # Reset training metrics
        if "SI-SNR" in eval_metrics: eval_metrics["SI-SNR"].reset()
        
        # --- Training Batch Loop (Updated for L1 loss) ---
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train",
                          total=min(len(train_loader), args.max_batches) if args.max_batches > 0 else len(train_loader))
        for i, (mixture, target_speech, target_music) in enumerate(train_pbar):
            if args.max_batches > 0 and i >= args.max_batches: break # Apply max_batches

            mixture_input = mixture.to(device) # (B, C, T)
            # Targets [speech, music] order: (B, N_src, C, T)
            if target_speech.ndim == 3: 
                targets = torch.stack([target_speech, target_music], dim=1).to(device)
            else: 
                targets = torch.stack([target_speech.unsqueeze(1), target_music.unsqueeze(1)], dim=1).to(device)

            batch_size = mixture_input.shape[0]

            optimizer.zero_grad()
            estimates = model(mixture_input) # estimates:(B,N,C,T)
            # Calculate batch losses using loss function instance
            batch_total_loss, batch_l1_speech, batch_l1_music = loss_fn(estimates, targets)

            # Update torchmetrics SI-SNR (handles multiple sources internally with PIT)
            if "SI-SNR" in eval_metrics:
                 try: eval_metrics["SI-SNR"].update(estimates.to(device), targets.to(device))
                 except Exception as e: print(f"Metric update error: {e}")

            batch_total_loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Accumulate per-sample losses/metrics
            epoch_total_loss += batch_total_loss.item() * batch_size
            epoch_loss_l1_speech += batch_l1_speech.item() * batch_size 
            epoch_loss_l1_music += batch_l1_music.item() * batch_size

            # Calculate and accumulate SI-SNR manually for reporting
            with torch.no_grad():
                speech_est_mono = estimates[:, loss_fn.speech_target_index].squeeze(2) # (B, T)
                music_est_mono = estimates[:, loss_fn.music_target_index].squeeze(2)   # (B, T)
                speech_tgt_mono = targets[:, loss_fn.speech_target_index].squeeze(2).to(device) # (B, T)
                music_tgt_mono = targets[:, loss_fn.music_target_index].squeeze(2).to(device)   # (B, T)
                batch_neg_sisnr_speech = torch.mean(si_snr_loss_manual(speech_est_mono, speech_tgt_mono)).item()
                batch_neg_sisnr_music = torch.mean(si_snr_loss_manual(music_est_mono, music_tgt_mono)).item()
                epoch_neg_sisnr_speech_report += batch_neg_sisnr_speech * batch_size
                epoch_neg_sisnr_music_report += batch_neg_sisnr_music * batch_size
            
            num_train_samples += batch_size

            # Update tqdm postfix (average over samples processed so far in epoch)
            speech_sisnr_report = -epoch_neg_sisnr_speech_report / num_train_samples if num_train_samples > 0 else 0
            music_sisnr_report = -epoch_neg_sisnr_music_report / num_train_samples if num_train_samples > 0 else 0
            train_pbar.set_postfix(
                loss=f"{epoch_total_loss / num_train_samples:.4f}",
                speech=f"{speech_sisnr_report:.2f}dB",
                music=f"{music_sisnr_report:.2f}dB",
                batch=f"{i+1}/{len(train_pbar)}"
            )

        # Calculate average epoch losses/metrics
        avg_epoch_total_loss = epoch_total_loss / num_train_samples if num_train_samples > 0 else 0
        avg_epoch_loss_l1_speech = epoch_loss_l1_speech / num_train_samples if num_train_samples > 0 else 0
        avg_epoch_loss_l1_music = epoch_loss_l1_music / num_train_samples if num_train_samples > 0 else 0
        avg_epoch_neg_sisnr_s_report = epoch_neg_sisnr_speech_report / num_train_samples if num_train_samples > 0 else 0
        avg_epoch_neg_sisnr_m_report = epoch_neg_sisnr_music_report / num_train_samples if num_train_samples > 0 else 0
        
        train_losses_total.append(avg_epoch_total_loss)
        train_l1_s_loss.append(avg_epoch_loss_l1_speech)
        train_l1_m_loss.append(avg_epoch_loss_l1_music)
        # Removed mel loss history append
        train_speech_si_snr.append(-avg_epoch_neg_sisnr_s_report) # Use manually calculated reporting SI-SNR
        train_music_si_snr.append(-avg_epoch_neg_sisnr_m_report)
        
        # Compute and store the torchmetrics SI-SNR (with PIT)
        overall_sisnr = 0.0
        if "SI-SNR" in eval_metrics and num_train_samples > 0:
            try: overall_sisnr = eval_metrics["SI-SNR"].compute().item()
            except Exception as e: print(f"Error computing overall SI-SNR: {e}")
        train_overall_si_snr.append(overall_sisnr)

        # --- Validation --- (Use updated evaluate function)
        print("\n=> Running validation...")
        val_results = evaluate(model, val_loader, loss_fn, eval_metrics, device,
                               desc=f"Epoch {epoch+1}/{args.epochs} Val",
                               max_batches=args.max_batches, sample_rate=args.sr) # Pass sr
        avg_val_total_loss = val_results["Loss_Total"]
        val_losses_total.append(avg_val_total_loss)
        val_l1_s_loss.append(val_results['Loss_Speech(L1)'])
        val_l1_m_loss.append(val_results['Loss_Music(L1)'])
        # Removed mel loss append
        val_speech_si_snr.append(val_results['Speech_SI-SNR_Eval'])
        val_music_si_snr.append(val_results['Music_SI-SNR_Eval'])
        val_overall_si_snr.append(val_results.get('SI-SNR', 0.0)) # Use .get for safety 
        if pesq_key and pesq_key in val_results: val_pesq_hist.append(val_results[pesq_key])
        if "STOI" in val_results: val_stoi_hist.append(val_results['STOI'])

        epoch_time = time.time() - start_time

        # --- Logging (Updated for L1) ---
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch+1}/{args.epochs} SUMMARY | Time: {epoch_time:.2f}s")
        print(f"{'-' * 60}")
        # Train metrics
        print(f"TRAIN | Loss: {avg_epoch_total_loss:.4f}")
        print(f"       Speech L1: {avg_epoch_loss_l1_speech:.4f} | Music L1: {avg_epoch_loss_l1_music:.4f}")
        print(f"       Speech SI-SNR (Report): {-avg_epoch_neg_sisnr_s_report:.2f} dB | Music SI-SNR (Report): {-avg_epoch_neg_sisnr_m_report:.2f} dB")
        print(f"       Overall SI-SNR (PIT): {overall_sisnr:.2f} dB")
        # Validation metrics
        print(f"{'-' * 60}")
        print(f"VAL   | Loss: {avg_val_total_loss:.4f}" + (" (NEW BEST) ✓" if avg_val_total_loss < best_val_loss else ""))
        print(f"       Speech L1: {val_results['Loss_Speech(L1)']:.4f} | Music L1: {val_results['Loss_Music(L1)']:.4f}")
        print(f"       Speech SI-SNR: {val_results['Speech_SI-SNR_Eval']:.2f} dB | Music SI-SNR: {val_results['Music_SI-SNR_Eval']:.2f} dB")
        print(f"       Overall SI-SNR (PIT): {val_results.get('SI-SNR', 0.0):.2f} dB")
        if pesq_key in val_results: print(f"       {pesq_key}: {val_results[pesq_key]:.2f}")
        if "STOI" in val_results: print(f"       STOI: {val_results['STOI']:.3f}")
        print(f"{'=' * 60}")

        # --- TensorBoard Logging (Updated for L1) ---
        writer.add_scalar('Training/learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('Loss/train_total', avg_epoch_total_loss, epoch+1)
        writer.add_scalar('Loss/train_speech_l1', avg_epoch_loss_l1_speech, epoch+1)
        writer.add_scalar('Loss/train_music_l1', avg_epoch_loss_l1_music, epoch+1)
        # Removed mel loss log
        writer.add_scalar('Metrics/train_speech_sisnr', -avg_epoch_neg_sisnr_s_report, epoch+1)
        writer.add_scalar('Metrics/train_music_sisnr', -avg_epoch_neg_sisnr_m_report, epoch+1)
        writer.add_scalar('Metrics/train_overall_sisnr', overall_sisnr, epoch+1)
        # Log validation metrics
        writer.add_scalar('Loss/val_total', avg_val_total_loss, epoch+1)
        writer.add_scalar('Loss/val_speech_l1', val_results['Loss_Speech(L1)'], epoch+1)
        writer.add_scalar('Loss/val_music_l1', val_results['Loss_Music(L1)'], epoch+1)
        # Removed mel loss log
        writer.add_scalar('Metrics/val_speech_sisnr', val_results['Speech_SI-SNR_Eval'], epoch+1)
        writer.add_scalar('Metrics/val_music_sisnr', val_results['Music_SI-SNR_Eval'], epoch+1)
        writer.add_scalar('Metrics/val_overall_sisnr', val_results.get('SI-SNR', 0.0), epoch+1)
        if pesq_key in val_results: writer.add_scalar(f'Metrics/val_{pesq_key}', val_results[pesq_key], epoch+1)
        if "STOI" in val_results: writer.add_scalar('Metrics/val_STOI', val_results['STOI'], epoch+1)

        # Generate and save plots for current epoch (Updated for L1)
        fig = plot_losses(train_losses_total, val_losses_total, output_dir / f'loss_plot_epoch_{epoch+1}.png', {
            'speech_l1': train_l1_s_loss,
            'music_l1': train_l1_m_loss
        }, {
            'speech_l1': val_l1_s_loss,
            'music_l1': val_l1_m_loss
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

        # --- Save Audio Sample and Spectrograms (Updated for HTDemucs output) ---
        if save_sample_flag:
            model.eval()
            with torch.no_grad():
                mixture_tensor = fixed_example_mixture.clone().to(device)
                if mixture_tensor.ndim == 2: mixture_tensor = mixture_tensor.unsqueeze(0)
                print(f"Saving sample audio/spectrograms epoch {epoch+1}, input shape: {mixture_tensor.shape}")
                estimates = model(mixture_tensor) # Shape (B=1, N_src=2, C=1, T)
                # Save audio files (speech, music estimates)
                save_audio(estimates[:, 0], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_speech_est.wav'), args.sr)
                save_audio(estimates[:, 1], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_music_est.wav'), args.sr)
                # Removed mixture reconstruction saving
                # Save spectrograms (speech, music estimates)
                save_spectrogram(estimates[:, 0], os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'), args.sr, args.n_fft, args.hop_length, title=f"Estimated Speech (Epoch {epoch+1})")
                save_spectrogram(estimates[:, 1], os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'), args.sr, args.n_fft, args.hop_length, title=f"Estimated Music (Epoch {epoch+1})")
                # Removed mixture reconstruction spectrogram saving
                # Log audio to TensorBoard (Updated)
                try:
                    speech_est_cpu = estimates[:, 0].cpu()
                    music_est_cpu = estimates[:, 1].cpu()
                    orig_mixture_cpu = fixed_example_mixture.cpu()
                    if orig_mixture_cpu.ndim == 2: orig_mixture_cpu = orig_mixture_cpu.unsqueeze(0)
                    writer.add_audio('Audio/original_mixture', orig_mixture_cpu.squeeze(), global_step=epoch+1, sample_rate=args.sr)
                    writer.add_audio('Audio/estimated_speech', speech_est_cpu.squeeze(), global_step=epoch+1, sample_rate=args.sr)
                    writer.add_audio('Audio/estimated_music', music_est_cpu.squeeze(), global_step=epoch+1, sample_rate=args.sr)
                    # Removed reconstructed mixture audio log
                    # Log spectrograms as images (Updated)
                    speech_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'))
                    music_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'))
                    writer.add_image('Spectrograms/speech', speech_spec_img, epoch+1, dataformats='HWC')
                    writer.add_image('Spectrograms/music', music_spec_img, epoch+1, dataformats='HWC')
                    # Removed mixture spectrogram log
                except Exception as e:
                    print(f"Warning: Could not log audio/images to TensorBoard: {e}")

        # --- Save Checkpoint (Updated for L1 losses) ---
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_total_loss,
                "best_val_loss": best_val_loss,
                "train_losses_total": train_losses_total,
                "val_losses_total": val_losses_total,
                "train_l1_s_loss": train_l1_s_loss, # Added
                "val_l1_s_loss": val_l1_s_loss,     # Added
                "train_l1_m_loss": train_l1_m_loss, # Added
                "val_l1_m_loss": val_l1_m_loss,     # Added
                # Removed mel history saving
                "val_pesq_hist": val_pesq_hist,
                "val_stoi_hist": val_stoi_hist,
                "train_speech_si_snr": train_speech_si_snr,
                "val_speech_si_snr": val_speech_si_snr,
                "train_music_si_snr": train_music_si_snr,
                "val_music_si_snr": val_music_si_snr,
                "train_overall_si_snr": train_overall_si_snr,
                "val_overall_si_snr": val_overall_si_snr,
                "val_metrics": val_results,
                "args": vars(args),
            }, checkpoint_path)
        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

        # --- Save Best Model (Based on Validation Loss) ---
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            best_model_save_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_total_loss,
                'val_metrics': val_results,
                'args': vars(args)
            }, best_model_save_path)
            print(f"\n★ SAVED BEST MODEL ★ to {best_model_save_path} (Loss: {best_val_loss:.4f})")

    # --- Plotting after training (Updated for L1) ---
    epochs_ran = list(range(1, len(train_losses_total) + 1))
    if len(train_losses_total) == len(epochs_ran) and len(epochs_ran) > 0:
        fig = plot_losses(
            train_losses_total, val_losses_total, output_dir / "loss_plot.png",
            {"speech_l1": train_l1_s_loss, "music_l1": train_l1_m_loss},
            {"speech_l1": val_l1_s_loss, "music_l1": val_l1_m_loss}
        )
        writer.add_figure('Plots/loss', fig, 0)
        fig = plot_si_snr(epochs_ran, train_speech_si_snr, train_music_si_snr, val_speech_si_snr, val_music_si_snr, train_overall_si_snr, val_overall_si_snr, output_dir / 'si_snr_plot.png')
        writer.add_figure('Plots/si_snr', fig, 0)
    else:
        print("Warning: History length mismatch or zero, skipping final plotting.")

    # --- Final Testing Section (Updated for HTDemucs) ---
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE - STARTING FINAL TESTING ".center(70, "="))
    print("=" * 70)

    best_model_path = output_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        saved_args_dict = checkpoint.get('args', {}) 
        current_args_dict = vars(args)
        final_model_args = {**current_args_dict, **saved_args_dict}

        test_model = HTDemucs(
            sources=sources, 
            audio_channels=1,
            samplerate=final_model_args['sr'],
            segment=final_model_args['segment']
        ).to(device)
        
        try: # Outer try for the entire testing phase
            try:
                test_model.load_state_dict(checkpoint['model_state_dict'])
                print("Successfully loaded model state dict for testing.")
            except Exception as e:
                print(f"Error loading state dict into HTDemucs: {e}. Skipping test.")
                writer.close()
                return

            # Find test split
            test_split_name = ""
            for name in ["test", "testing"]:
                if (root_path / name).exists():
                    test_split_name = name
                    break
            if not test_split_name:
                 print("Warning: No 'test' or 'testing' directory found. Using validation set for final testing.")
                 if (root_path / "val").exists(): test_split_name = "val"
                 elif (root_path / "validation").exists(): test_split_name = "validation"
                 elif (root_path / "valid").exists(): test_split_name = "valid"
                 else: raise FileNotFoundError("No test or validation split found for final evaluation.")
                 
            test_dataset = FolderTripletDataset(root_path, split=test_split_name, 
                                              segment_length_sec=args.segment, 
                                              hop_length_sec=args.hop_length_sec,
                                              sr=args.sr)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
            print(f"Using test split: '{test_split_name}' with {len(test_dataset)} samples")

            test_metric_list = {}
            try: test_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
            except AttributeError: pass
            if pesq_mode:
                try: test_metric_list[pesq_key] = torchmetrics.audio.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
                except AttributeError: pass
            try: test_metric_list["STOI"] = torchmetrics.audio.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
            except AttributeError: pass
            test_metrics_collection = torchmetrics.MetricCollection(test_metric_list).to(device)

            test_loss_fn = HTDemucsL1Loss(speech_target_index=0, music_target_index=1).to(device)

            test_results = evaluate(test_model, test_loader, test_loss_fn, test_metrics_collection, device, 
                                   desc="Testing (best model)", max_batches=0, sample_rate=args.sr)

            print("\n" + "=" * 70)
            print(" FINAL TEST RESULTS ".center(70, "="))
            print("=" * 70)
            print(f"Model: HTDemucs @ Epoch {checkpoint.get('epoch', 'N/A')}")
            print(f"\nLOSS METRICS:")
            print(f"  Total Loss:       {test_results['Loss_Total']:.4f}")
            print(f"  Speech L1:        {test_results['Loss_Speech(L1)']:.4f}")
            print(f"  Music L1:         {test_results['Loss_Music(L1)']:.4f}")
            print(f"\nPERFORMANCE METRICS:")
            print(f"  SI-SNR (Overall PIT): {test_results.get('SI-SNR', 0.0):.2f} dB")
            print(f"  Speech SI-SNR (Eval): {test_results['Speech_SI-SNR_Eval']:.2f} dB")
            print(f"  Music SI-SNR (Eval):  {test_results['Music_SI-SNR_Eval']:.2f} dB")
            if pesq_key in test_results: print(f"  {pesq_key}:             {test_results[pesq_key]:.2f}")
            if "STOI" in test_results: print(f"  STOI:                 {test_results['STOI']:.3f}")
            print("=" * 70)

            writer.add_scalar('Test/Loss_Total', test_results['Loss_Total'], 0)
            writer.add_scalar('Test/Loss_Speech_L1', test_results['Loss_Speech(L1)'], 0)
            writer.add_scalar('Test/Loss_Music_L1', test_results['Loss_Music(L1)'], 0)
            writer.add_scalar('Test/SI-SNR_Overall', test_results.get('SI-SNR', 0.0), 0)
            writer.add_scalar('Test/Speech_SI-SNR_Eval', test_results['Speech_SI-SNR_Eval'], 0)
            writer.add_scalar('Test/Music_SI-SNR_Eval', test_results['Music_SI-SNR_Eval'], 0)
            if pesq_key in test_results: writer.add_scalar(f'Test/{pesq_key}', test_results[pesq_key], 0)
            if "STOI" in test_results: writer.add_scalar('Test/STOI', test_results['STOI'], 0)

            test_summary = (
                f"Loss Total: {test_results['Loss_Total']:.4f}\n"
                f"Speech L1: {test_results['Loss_Speech(L1)']:.4f}\n"
                f"Music L1: {test_results['Loss_Music(L1)']:.4f}\n"
                f"SI-SNR Overall: {test_results.get('SI-SNR', 0.0):.2f} dB\n"
                f"Speech SI-SNR: {test_results['Speech_SI-SNR_Eval']:.2f} dB\n"
                f"Music SI-SNR: {test_results['Music_SI-SNR_Eval']:.2f} dB\n"
            )
            if pesq_key in test_results: test_summary += f"{pesq_key}: {test_results[pesq_key]:.2f}\n"
            if "STOI" in test_results: test_summary += f"STOI: {test_results['STOI']:.3f}\n"
            writer.add_text('Test/Summary', test_summary, 0)

            num_test_samples = args.test_samples
            if num_test_samples > 0:
                test_audio_dir = output_dir / 'test_audio_samples'
                test_spec_dir = output_dir / 'test_spectrograms'
                test_audio_dir.mkdir(exist_ok=True)
                test_spec_dir.mkdir(exist_ok=True)
                print(f"\nSelecting {num_test_samples} test samples for saving... ({'Random' if args.test_samples_random else 'Sequential'}) ")
                total_samples = len(test_dataset)
                selected_indices = []
                if args.test_samples_random:
                     if total_samples <= num_test_samples: selected_indices = list(range(total_samples))
                     else: selected_indices = random.sample(range(total_samples), num_test_samples)
                else:
                     if total_samples <= num_test_samples: selected_indices = list(range(total_samples))
                     else: step = total_samples // num_test_samples; selected_indices = [i * step for i in range(num_test_samples)]
                print(f"Selected indices: {sorted(selected_indices)}")

                test_model.eval()
                with torch.no_grad():
                    for i, sample_idx in enumerate(selected_indices):
                        mixture, target_speech, target_music = test_dataset[sample_idx]
                        mixture_input = mixture.unsqueeze(0).to(device)
                        estimates = test_model(mixture_input)
                        
                        sample_dir = test_audio_dir / f'sample_{i+1:02d}'
                        sample_spec_dir = test_spec_dir / f'sample_{i+1:02d}'
                        sample_dir.mkdir(exist_ok=True)
                        sample_spec_dir.mkdir(exist_ok=True)

                        save_audio(mixture, sample_dir / 'gt_mixture.wav', args.sr)
                        save_audio(target_speech, sample_dir / 'gt_speech.wav', args.sr)
                        save_audio(target_music, sample_dir / 'gt_music.wav', args.sr)
                        save_audio(estimates[:, 0], sample_dir / 'est_speech.wav', args.sr)
                        save_audio(estimates[:, 1], sample_dir / 'est_music.wav', args.sr)
                        
                        save_spectrogram(mixture, sample_spec_dir / 'gt_mixture_spec.png', args.sr, args.n_fft, args.hop_length, f"GT Mixture (Sample {i+1})")
                        save_spectrogram(target_speech, sample_spec_dir / 'gt_speech_spec.png', args.sr, args.n_fft, args.hop_length, f"GT Speech (Sample {i+1})")
                        save_spectrogram(target_music, sample_spec_dir / 'gt_music_spec.png', args.sr, args.n_fft, args.hop_length, f"GT Music (Sample {i+1})")
                        save_spectrogram(estimates[:, 0], sample_spec_dir / 'est_speech_spec.png', args.sr, args.n_fft, args.hop_length, f"Est Speech (Sample {i+1})")
                        save_spectrogram(estimates[:, 1], sample_spec_dir / 'est_music_spec.png', args.sr, args.n_fft, args.hop_length, f"Est Music (Sample {i+1})")
                        
                        speech_est_mono = estimates[:, 0].squeeze()
                        music_est_mono = estimates[:, 1].squeeze()
                        speech_tgt_mono = target_speech.to(device).squeeze()
                        music_tgt_mono = target_music.to(device).squeeze()
                        speech_si_snr = -si_snr_loss_manual(speech_est_mono, speech_tgt_mono).item()
                        music_si_snr = -si_snr_loss_manual(music_est_mono, music_tgt_mono).item()
                        
                        with open(sample_dir / 'metrics.txt', 'w') as f:
                            f.write(f"Sample {i+1} (Dataset index: {sample_idx})\n")
                            f.write(f"Speech SI-SNR: {speech_si_snr:.2f} dB\n")
                            f.write(f"Music SI-SNR: {music_si_snr:.2f} dB\n")
                        print(f"Saved test sample {i+1}/{len(selected_indices)} (idx {sample_idx}): Speech SI-SNR {speech_si_snr:.2f}dB, Music SI-SNR {music_si_snr:.2f}dB")

                print(f"\nSaved {len(selected_indices)} test samples with audio and spectrograms to:")
                print(f"  Audio: {test_audio_dir}")
                print(f"  Spectrograms: {test_spec_dir}")
            else:
                 print("\nSkipping test sample saving as --test_samples is 0.")

        except (ValueError, FileNotFoundError, RuntimeError) as e:
             print(f"Could not create/run test dataset/loader or error during testing: {e}")
             print("Skipping test evaluation steps after error.")
        except Exception as e: # Catch any other unexpected error during testing
             print(f"An unexpected error occurred during testing: {e}")
             print("Skipping test evaluation steps after error.")
             
    else:
        print("No best model checkpoint found. Skipping final testing.")
    
    writer.close()
    print("\nTraining and evaluation process finished.")


# --- Argument Parser (Updated for HTDemucs) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HTDemucs with L1 Reconstruction Loss")

    # --- Data Arguments ---
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--segment', type=float, default=3.0, help='Duration of audio segments in seconds (passed to model and dataset)')
    parser.add_argument('--hop_length_sec', type=float, default=1.5, help='Hop length between audio segments in seconds (for windowed dataset)') # Added
    parser.add_argument('--sr', type=int, default=8000, help='Sample rate (PESQ requires 8k or 16k)')

    # --- Training Arguments ---
    parser.add_argument('--save_dir', type=str, default='checkpoints_htdemucs_l1', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size') # Adjusted default
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N batches (not used in current tqdm setup)')
    parser.add_argument('--max_batches', type=int, default=0, help='Maximum number of batches per epoch (0 for all)')

    # --- Loss Arguments (Removed Mel/Mixer specific args) ---
    # L1 loss is now default and only loss type
    parser.add_argument('--n_fft', type=int, default=512, help='FFT size for STFT (used for spectrogram saving)')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length for STFT (used for spectrogram saving)')
    # parser.add_argument('--n_mels', type=int, default=80, help='Number of Mel bins (used for spectrogram saving if needed)')

    # --- Model Hyperparameters (Removed MSHybridNet specific args) ---
    # Add HTDemucs specific args here if needed, e.g.:
    # parser.add_argument('--channels', type=int, default=48, help='Initial hidden channels for HTDemucs')
    # parser.add_argument('--growth', type=int, default=2, help='Channel growth factor for HTDemucs')
    # parser.add_argument('--depth', type=int, default=6, help='Depth for HTDemucs')
    # parser.add_argument('--kernel_size', type=int, default=8, help='Conv kernel size for HTDemucs')
    # parser.add_argument('--stride', type=int, default=4, help='Conv stride for HTDemucs')
    
    # --- Eval/Debug Arguments ---
    parser.add_argument('--sample_index', type=int, default=-1, help='Index of validation sample to save audio for (-1 for random)')
    parser.add_argument('--test_samples', type=int, default=20, help='Number of test samples to save during final evaluation (0 to disable)') # Adjusted default
    parser.add_argument('--test_samples_random', action='store_true', help='Use random samples for test audio instead of sequential/spaced')

    args = parser.parse_args()

    # --- Basic Validation ---
    if args.sr not in [8000, 16000]:
        raise ValueError("PESQ metric requires sr=8000 or sr=16000")
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
        
    # Print summary of arguments
    print("=" * 70)
    print("HTDemucs L1 Training Configuration:")
    print("=" * 70)
    for arg, value in vars(args).items():
        print(f"{arg:20}: {value}")
    print("=" * 70)

    train(args)
