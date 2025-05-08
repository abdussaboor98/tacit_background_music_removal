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

# Assuming your model and dataset are in these files
from model import MSHybridNet
from audio_datasets import FolderTripletDataset
# Using torchmetrics for SI-SNR, PESQ, STOI
import torchmetrics

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Multiple GPUs detected ({torch.cuda.device_count()}). Setting to use GPU 2.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
# --- Loss Functions ---

class MelSpectrogramLoss(nn.Module):
    """Calculates L1 loss between Mel Spectrograms. Simplified device handling."""
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=128, n_mels=80,
                 power=2.0, normalized=True, reduction='mean'):
        super().__init__()
        # Store transform parameters, create transform in forward pass
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.normalized = normalized
        # Create loss function here
        self.loss_fn = nn.L1Loss(reduction=reduction)
        # No transform created here initially
        self.mel_transform = None
        self.transform_device = None
        print(f"Initialized MelSpectrogramLoss stub with sr={sample_rate}, n_fft={n_fft}, hop={hop_length}, n_mels={n_mels}")

    def _create_or_move_transform(self, target_device):
        """Creates transform or moves it if device changed."""
        if self.mel_transform is None or self.transform_device != target_device:
            # print(f"MelSpectrogramLoss: Moving/Creating transform for device {target_device}") # Optional debug print
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=self.power,
                normalized=self.normalized,
                center=True,
                pad_mode="reflect"
            ).to(target_device)
            self.transform_device = target_device

    def to(self, *args, **kwargs):
        # Move the loss_fn parameters if any, handle internal transform move
        new_self = super().to(*args, **kwargs)
        # Ensure device is parsed correctly from args/kwargs
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        new_self._create_or_move_transform(device) # Ensure transform moves with module
        return new_self

    def forward(self, estimate, target):
        target_device = estimate.device
        # Ensure the transform exists and is on the correct device
        self._create_or_move_transform(target_device)

        # --- Handle Channel Dimension ---
        if estimate.ndim == 3 and estimate.shape[1] == 1: estimate = estimate.squeeze(1)
        if target.ndim == 3 and target.shape[1] == 1: target = target.squeeze(1)
        if estimate.ndim == 1: estimate = estimate.unsqueeze(0)
        if target.ndim == 1: target = target.unsqueeze(0)
        # Now inputs should be (Batch, Time)

        # Calculate Mel Spectrograms
        mel_est = self.mel_transform(estimate)
        mel_tgt = self.mel_transform(target)

        # Calculate L1 loss on log Mel Spectrograms
        log_mel_est = torch.log(mel_est + 1e-8)
        log_mel_tgt = torch.log(mel_tgt + 1e-8)

        loss = self.loss_fn(log_mel_est, log_mel_tgt)
        return loss


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

class MSHybridAEDirectLoss(nn.Module):
    """
    Loss for MSHybrid Autoencoder with WFAE-inspired approach WITHOUT PIT.
    Assumes fixed order: Source 0 = Speech+Ambient, Source 1 = Music+Vocals.
    Uses SI-SNR for speech (and optionally music), mixture reconstruction L1 loss, 
    and optional Mel Spectrogram loss for speech.
    Designed specifically for the MSHybridNet autoencoder architecture.
    """
    def __init__(self, mix_recon_weight=0.1, mel_loss_weight=0.1,
                 speech_target_index=0, music_target_index=1, # Fixed indices
                 include_music_loss=False, # Flag to include music loss in backprop
                 sample_rate=8000, n_fft=512, hop_length=128, n_mels=80):
        super().__init__()
        self.mix_recon_weight = mix_recon_weight
        self.mel_loss_weight = mel_loss_weight
        self.speech_target_index = speech_target_index
        self.music_target_index = music_target_index
        self.include_music_loss = include_music_loss
        self.sample_rate = sample_rate  # Store sample_rate as an instance attribute

        if self.mel_loss_weight > 0:
            self.mel_loss_fn = MelSpectrogramLoss(sample_rate=sample_rate, n_fft=n_fft,
                                                  hop_length=hop_length, n_mels=n_mels)
        else:
            self.mel_loss_fn = None
        self.mix_recon_loss_fn = nn.L1Loss() # Using L1 for mixture reconstruction

    def to(self, device):
        super().to(device)
        if self.mel_loss_fn is not None:
            self.mel_loss_fn = self.mel_loss_fn.to(device)
        return self

    def forward(self, s_estimates, targets, mix_estimate, mix_target):
        # s_estimates: (Batch, NumSources, Channels, Time)
        # targets: (Batch, NumSources, Channels, Time)
        # mix_estimate: (Batch, Channels, Time)
        # mix_target: (Batch, Channels, Time)
        batch_size, num_sources, num_channels, _ = s_estimates.shape
        device = s_estimates.device
        
        # Ensure all inputs are on the same device
        targets = targets.to(device)
        mix_estimate = mix_estimate.to(device)
        mix_target = mix_target.to(device)

        # --- Handle Channel Dimension (Assuming Mono for Loss Calculation) ---
        if num_channels == 1:
            s_estimates_mono = s_estimates.squeeze(2) # (B, N_src, T)
            targets_mono = targets.squeeze(2)         # (B, N_src, T)
            mix_estimate_mono = mix_estimate.squeeze(1) # (B, T)
            mix_target_mono = mix_target.squeeze(1)     # (B, T)
        else: # Average if multi-channel
            print("Warning: Multi-channel detected, averaging channels for loss calculation.")
            s_estimates_mono = torch.mean(s_estimates, dim=2)
            targets_mono = torch.mean(targets, dim=2)
            mix_estimate_mono = torch.mean(mix_estimate, dim=1)
            mix_target_mono = torch.mean(mix_target, dim=1)


        # --- Calculate Losses Directly (NO PIT) ---
        speech_est = s_estimates_mono[:, self.speech_target_index, :]
        speech_tgt = targets_mono[:, self.speech_target_index, :]
        music_est = s_estimates_mono[:, self.music_target_index, :]
        music_tgt = targets_mono[:, self.music_target_index, :]

        # 1. Speech SI-SNR Loss (potentially for gradient)
        batch_loss_sisnr_speech = si_snr_loss_manual(speech_est, speech_tgt) # (B,)
        avg_loss_sisnr_speech = torch.mean(batch_loss_sisnr_speech) # Scalar mean

        # 2. Music SI-SNR Calculation (potentially for gradient or monitoring)
        batch_loss_sisnr_music = si_snr_loss_manual(music_est, music_tgt) # (B,)
        avg_loss_sisnr_music = torch.mean(batch_loss_sisnr_music) # Scalar mean

        # 3. Mixture Reconstruction Loss
        min_len_mix = min(mix_estimate_mono.shape[-1], mix_target_mono.shape[-1])
        loss_mix_recon = self.mix_recon_loss_fn(
            mix_estimate_mono[..., :min_len_mix],
            mix_target_mono[..., :min_len_mix]
        ) # Scalar

        # 4. Mel Loss (Speech only)
        loss_mel_speech = torch.tensor(0.0).to(device)
        if self.mel_loss_fn is not None and self.mel_loss_weight > 0:
            self.mel_loss_fn = self.mel_loss_fn.to(device) # Ensure device
            loss_mel_speech = self.mel_loss_fn(speech_est, speech_tgt) # Scalar

        # 5. --- Combine Losses for Backpropagation ---
        total_loss = avg_loss_sisnr_speech # Start with speech SI-SNR loss
        if self.include_music_loss:
            total_loss += avg_loss_sisnr_music # Add music SI-SNR loss if requested
        total_loss += self.mix_recon_weight * loss_mix_recon
        total_loss += self.mel_loss_weight * loss_mel_speech

        # Return total loss for backprop and components for logging/evaluation
        return total_loss, avg_loss_sisnr_speech, loss_mix_recon, loss_mel_speech, avg_loss_sisnr_music


# --- Evaluation Function (Modified for Direct Loss & Max Batches) ---
def evaluate(model, dataloader, loss_fn_eval, metrics_dict, device, desc="Evaluating", max_batches=0):
    """ Evaluate model on a dataloader (Adapted for Direct WFAE Loss & Max Batches) """
    model.eval()
    total_loss_sisnr_speech = 0.0
    total_loss_sisnr_music = 0.0
    total_loss_mix_recon = 0.0
    total_loss_mel = 0.0
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
            mixture_target_mono = mixture.squeeze(1).to(device) if mixture.shape[1] == 1 else torch.mean(mixture, dim=1).to(device) # (B, T) Target for recon loss

            # Targets: Stack and ensure correct shape (B, N_src, C, T)
            if target_speech.ndim == 3: # Already (B, C, T)
                targets = torch.stack([target_speech, target_music], dim=1).to(device)
            else: # Assume (B, T), add channel dim
                targets = torch.stack([target_speech.unsqueeze(1), target_music.unsqueeze(1)], dim=1).to(device)

            batch_size = mixture_input.shape[0]
            num_samples += batch_size

            # Model forward pass -> s_estimates (B, N_src, C, T), x_mix_recon (B, C, T)
            s_estimates, x_mix_recon = model(mixture_input)

            # Calculate loss components using the loss function for reporting
            # Loss expects channel dim squeezed if mono
            # Ensure all inputs are on the correct device
            s_estimates = s_estimates.to(device)
            targets = targets.to(device)
            x_mix_recon = x_mix_recon.to(device)
            mixture_for_loss = mixture.to(device)
            
            # Calculate loss using the provided loss function instance
            _, batch_avg_loss_sisnr_speech, batch_loss_mix_recon, batch_loss_mel_speech, batch_avg_loss_sisnr_music = loss_fn_eval(
                s_estimates, targets, x_mix_recon, mixture_for_loss # Pass original mixture for loss
            )
            total_loss_sisnr_speech += batch_avg_loss_sisnr_speech.item() * batch_size
            total_loss_sisnr_music += batch_avg_loss_sisnr_music.item() * batch_size
            total_loss_mix_recon += batch_loss_mix_recon.item() * batch_size
            total_loss_mel += batch_loss_mel_speech.item() * batch_size
            accumulated_music_sisnr_metric += (-batch_avg_loss_sisnr_music.item()) * batch_size # SI-SNR metric is neg of loss
            accumulated_speech_sisnr_metric += (-batch_avg_loss_sisnr_speech.item()) * batch_size # SI-SNR metric is neg of loss

            # Update progress bar with current metrics
            if num_samples > 0:
                eval_pbar.set_postfix(
                    speech=f"{accumulated_speech_sisnr_metric / num_samples:.2f}dB",
                    music=f"{accumulated_music_sisnr_metric / num_samples:.2f}dB"
                )

            # --- Calculate Metrics on Separated Sources (NO PIT) ---
            speech_est = s_estimates[:, loss_fn_eval.speech_target_index]   # (B, C, T)
            music_est = s_estimates[:, loss_fn_eval.music_target_index]     # (B, C, T)
            speech_tgt = targets[:, loss_fn_eval.speech_target_index]       # (B, C, T)
            music_tgt = targets[:, loss_fn_eval.music_target_index]         # (B, C, T)


            if "SI-SNR" in metrics_dict:
                try:
                    # Make sure shapes are correct for torchmetrics SI-SNR (expects shape [B, C, T])
                    metrics_dict["SI-SNR"].update(s_estimates.to(device), targets.to(device))
                except Exception as e:
                    print(f"Warning: SI-SNR calculation failed: {e}")

            # Update PESQ/STOI for speech only
            speech_est_mono = speech_est.squeeze(1) if speech_est.shape[1] == 1 else torch.mean(speech_est, dim=1)
            speech_tgt_mono = speech_tgt.squeeze(1) if speech_tgt.shape[1] == 1 else torch.mean(speech_tgt, dim=1)

            min_len = min(speech_est_mono.shape[-1], speech_tgt_mono.shape[-1])
            speech_est_trim = speech_est_mono[..., :min_len]
            speech_tgt_trim = speech_tgt_mono[..., :min_len]

            valid_indices = torch.sum(torch.abs(speech_tgt_trim), dim=-1) > 1e-5
            if torch.any(valid_indices):
                estimates_valid = speech_est_trim[valid_indices].to(device)
                targets_valid = speech_tgt_trim[valid_indices].to(device)
                # Get sample rate from loss function or fall back to default
                sample_rate = getattr(loss_fn_eval, 'sample_rate', 8000)
                pesq_key = f"PESQ-{'WB' if sample_rate == 16000 else 'NB'}" # Access sr from loss
                if pesq_key in metrics_dict and estimates_valid.numel() > 0:
                    try: metrics_dict[pesq_key].update(estimates_valid, targets_valid)
                    except Exception as e: print(f"Warning: PESQ calculation failed: {e}")
                if "STOI" in metrics_dict and estimates_valid.numel() > 0:
                    try: metrics_dict["STOI"].update(estimates_valid, targets_valid)
                    except Exception as e: print(f"Warning: STOI calculation failed: {e}")


    # Compute final metric values
    if num_samples > 0:
        results = {name: metric.compute().item() for name, metric in metrics_dict.items()}
        avg_loss_sisnr_speech = total_loss_sisnr_speech / num_samples
        avg_loss_sisnr_music = total_loss_sisnr_music / num_samples
        avg_loss_mix_recon = total_loss_mix_recon / num_samples
        avg_loss_mel = total_loss_mel / num_samples
        results["Loss_Speech(SI-SNR)"] = avg_loss_sisnr_speech
        results["Loss_Mix(L1)"] = avg_loss_mix_recon
        results["Loss_Mel"] = avg_loss_mel
        # Calculate total loss based on eval set averages (more stable)
        results["Loss_Total"] = avg_loss_sisnr_speech + loss_fn_eval.mix_recon_weight * avg_loss_mix_recon + loss_fn_eval.mel_loss_weight * avg_loss_mel
        if loss_fn_eval.include_music_loss:
             results["Loss_Total"] += avg_loss_sisnr_music
             results["Loss_Music(SI-SNR)"] = avg_loss_sisnr_music

        # Add separate speech and music SI-SNR metric values
        results["Speech_SI-SNR_Eval"] = accumulated_speech_sisnr_metric / num_samples
        results["Music_SI-SNR_Eval"] = accumulated_music_sisnr_metric / num_samples
        
    else:
        # Handle case where no samples were processed
        results = {}
        # Add default values for all necessary metrics
        results["SI-SNR"] = 0.0
        results["Loss_Speech(SI-SNR)"] = 0.0
        results["Loss_Music(SI-SNR)"] = 0.0
        results["Loss_Mix(L1)"] = 0.0
        results["Loss_Mel"] = 0.0
        results["Loss_Total"] = 0.0
        results["Speech_SI-SNR_Eval"] = 0.0
        results["Music_SI-SNR_Eval"] = 0.0
        if pesq_key:
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

def plot_losses(train_losses, val_losses, save_path, train_losses_components=None, val_losses_components=None, include_music_loss=False):
    """
    Plot training and validation losses and save the figure.
    Also plots individual loss components if provided.
    
    Args:
        train_losses: List of total training losses
        val_losses: List of total validation losses
        save_path: Path to save the figure
        train_losses_components: Dict of component losses for training (speech_sisnr, music_sisnr, mix_recon, mel)
        val_losses_components: Dict of component losses for validation
        include_music_loss: Whether music loss is included in training
    """
    if not train_losses or not val_losses:
        print("Warning: Loss history is empty, skipping plotting.")
        return None
    
    # Set up a multi-subplot figure based on what components we have
    component_plots = 0
    if train_losses_components and val_losses_components:
        component_plots = 4 if include_music_loss else 3
    
    fig = plt.figure(figsize=(12, 6 + 3 * component_plots))
    
    # Total loss plot (always included)
    plt.subplot(1 + component_plots, 1, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Total Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    min_train_epoch = np.argmin(train_losses) + 1
    min_val_epoch = np.argmin(val_losses) + 1
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    plt.annotate(f'Min Train: {min_train_loss:.4f} (Ep {min_train_epoch})', 
                 xy=(min_train_epoch, min_train_loss), 
                 xytext=(min_train_epoch, min_train_loss + (max(train_losses)-min_train_loss)*0.1), 
                 ha='center')
    plt.annotate(f'Min Val: {min_val_loss:.4f} (Ep {min_val_epoch})', 
                 xy=(min_val_epoch, min_val_loss), 
                 xytext=(min_val_epoch, min_val_loss + (max(val_losses)-min_val_loss)*0.15), 
                 ha='center')
    
    # If we have component losses, add subplots for each
    if component_plots > 0 and train_losses_components and val_losses_components:
        # Ensure all lists have the proper length
        epochs_len = len(epochs)
        
        # Get loss components, ensuring proper length
        train_speech_sisnr = train_losses_components.get('speech_sisnr', [0] * epochs_len)[:epochs_len]
        val_speech_sisnr = val_losses_components.get('speech_sisnr', [0] * epochs_len)[:epochs_len]
        train_mix_recon = train_losses_components.get('mix_recon', [0] * epochs_len)[:epochs_len]
        val_mix_recon = val_losses_components.get('mix_recon', [0] * epochs_len)[:epochs_len]
        train_mel = train_losses_components.get('mel', [0] * epochs_len)[:epochs_len]
        val_mel = val_losses_components.get('mel', [0] * epochs_len)[:epochs_len]
        
        # Speech SI-SNR Loss
        plt.subplot(1 + component_plots, 1, 2)
        plt.plot(epochs, train_speech_sisnr, 'b-', label='Train Speech SI-SNR Loss')
        plt.plot(epochs, val_speech_sisnr, 'r-', label='Val Speech SI-SNR Loss')
        plt.title('Speech SI-SNR Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Mix Reconstruction Loss
        plt.subplot(1 + component_plots, 1, 3)
        plt.plot(epochs, train_mix_recon, 'b-', label='Train Mix Recon Loss')
        plt.plot(epochs, val_mix_recon, 'r-', label='Val Mix Recon Loss')
        plt.title('Mixture Reconstruction Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Mel Spectrogram Loss
        plt.subplot(1 + component_plots, 1, 4)
        plt.plot(epochs, train_mel, 'b-', label='Train Mel Loss')
        plt.plot(epochs, val_mel, 'r-', label='Val Mel Loss')
        plt.title('Mel Spectrogram Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Music SI-SNR Loss (if included)
        if include_music_loss and 'music_sisnr' in train_losses_components and 'music_sisnr' in val_losses_components:
            train_music_sisnr = train_losses_components.get('music_sisnr', [0] * epochs_len)[:epochs_len]
            val_music_sisnr = val_losses_components.get('music_sisnr', [0] * epochs_len)[:epochs_len]
            
            plt.subplot(1 + component_plots, 1, 5)
            plt.plot(epochs, train_music_sisnr, 'b-', label='Train Music SI-SNR Loss')
            plt.plot(epochs, val_music_sisnr, 'r-', label='Val Music SI-SNR Loss')
            plt.title('Music SI-SNR Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")
    return fig


def plot_si_snr(epochs, train_speech, train_music, val_speech, val_music, train_overall, val_overall, save_path):
    """Plot SI-SNR metrics for training and validation and save the figure"""
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
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(12, 15))
    
    # Speech SI-SNR subplot
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_speech, 'b-o', markersize=4, label='Train Speech SI-SNR')
    plt.plot(epochs, val_speech, 'r-o', markersize=4, label='Val Speech SI-SNR')
    plt.title('Speech SI-SNR (dB)'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    max_train_idx = np.argmax(train_speech); max_val_idx = np.argmax(val_speech)
    plt.annotate(f'Max: {train_speech[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_speech[max_train_idx]), ha='center', va='bottom')
    plt.annotate(f'Max: {val_speech[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_speech[max_val_idx]), ha='center', va='bottom', color='red')

    # Music SI-SNR subplot
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_music, 'b-o', markersize=4, label='Train Music SI-SNR')
    plt.plot(epochs, val_music, 'r-o', markersize=4, label='Val Music SI-SNR')
    plt.title('Music SI-SNR (dB)'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
    max_train_idx = np.argmax(train_music); max_val_idx = np.argmax(val_music)
    plt.annotate(f'Max: {train_music[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_music[max_train_idx]), ha='center', va='bottom')
    plt.annotate(f'Max: {val_music[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_music[max_val_idx]), ha='center', va='bottom', color='red')
    
    # Overall SI-SNR subplot (PIT)
    if train_overall and val_overall:
        plt.subplot(3, 1, 3)
        plt.plot(epochs, train_overall, 'b-o', markersize=4, label='Train Overall SI-SNR')
        plt.plot(epochs, val_overall, 'r-o', markersize=4, label='Val Overall SI-SNR')
        plt.title('Overall SI-SNR (dB)'); plt.xlabel('Epochs'); plt.grid(True, linestyle='--', alpha=0.6); plt.legend()
        max_train_idx = np.argmax(train_overall); max_val_idx = np.argmax(val_overall)
        plt.annotate(f'Max: {train_overall[max_train_idx]:.2f}', xy=(epochs[max_train_idx], train_overall[max_train_idx]), ha='center', va='bottom')
        plt.annotate(f'Max: {val_overall[max_val_idx]:.2f}', xy=(epochs[max_val_idx], val_overall[max_val_idx]), ha='center', va='bottom', color='red')
    else:
        plt.subplot(3, 1, 3)
        plt.text(0.5, 0.5, 'No Overall SI-SNR Data Available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.xlabel('Epochs')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path); 
    plt.close(); 
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

def train(args):
    """Main training loop (Modified for WFAE + Checkpoints + Max Batches)"""
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
    model = MSHybridNet(
        channels=1, # Force mono channel for now
        enc_kernel_size=args.enc_kernel_size, enc_stride=args.enc_stride, enc_features=args.features,
        num_blocks=args.num_blocks,
        tcn_hidden_channels=args.tcn_hidden_channels, tcn_kernel_size=args.tcn_kernel_size, tcn_layers_per_block=args.tcn_layers_per_block, tcn_dilation_base=args.tcn_dilation_base,
        conformer_dim=args.conformer_dim, conformer_heads=args.conformer_heads, conformer_kernel_size=args.conformer_kernel_size,
        conformer_ffn_expansion=args.conformer_ffn_expansion, conformer_dropout=args.conformer_dropout
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss Function (Instantiate the WFAE loss WITHOUT PIT) ---
    # Determine mel loss weight based on args
    mel_weight_to_use = 0.0
    if args.disable_mel_loss:
        mel_weight_to_use = 0.0
    elif args.mel_weight_decay:
        mel_weight_to_use = args.mel_weight_initial  # Start with initial value
    else:
        mel_weight_to_use = args.mel_weight  # Use fixed value
        
    loss_fn = MSHybridAEDirectLoss(
        mix_recon_weight=args.mix_recon_weight,
        mel_loss_weight=mel_weight_to_use,
        speech_target_index=0, # Dataset returns mix, speech, music
        music_target_index=1,
        include_music_loss=args.include_music_loss, # Use new argument
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    ).to(device)
    
    # Explicitly ensure the mixer loss function and any mel loss is also on the correct device
    if hasattr(loss_fn, 'mix_recon_loss_fn'):
        loss_fn.mix_recon_loss_fn = loss_fn.mix_recon_loss_fn.to(device)
    if hasattr(loss_fn, 'mel_loss_fn') and loss_fn.mel_loss_fn is not None:
        loss_fn.mel_loss_fn = loss_fn.mel_loss_fn.to(device)

    # --- Metrics for Validation/Test ---
    pesq_mode = 'wb' if args.sr == 16000 else 'nb'
    pesq_key = f"PESQ-{pesq_mode.upper()}"

    eval_metric_list = {} # Initialize empty
    eval_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
    if pesq_mode:
        eval_metric_list[pesq_key] = torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
    eval_metric_list["STOI"] = torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
    eval_metrics = torchmetrics.MetricCollection(eval_metric_list).to(device)

    # --- Training Loop ---
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    # --- Initialize History Lists ---
    train_losses_total, val_losses_total = [], []
    train_sisnr_s_loss, val_sisnr_s_loss = [], []
    train_sisnr_m_loss, val_sisnr_m_loss = [], []
    train_mix_recon_loss, val_mix_recon_loss = [], []
    train_mel_loss, val_mel_loss = [], []
    val_pesq_hist, val_stoi_hist = [], [] # For plotting metrics
    train_speech_si_snr, val_speech_si_snr = [], [] # For positive SI-SNR metrics
    train_music_si_snr, val_music_si_snr = [], [] # For positive SI-SNR metrics
    train_overall_si_snr, val_overall_si_snr = [], [] # For torchmetrics SI-SNR

    # --- Initialize Torchmetrics for Training SI-SNR ---
    train_sisnr_metric = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)

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
            train_sisnr_s_loss = checkpoint.get('train_sisnr_s_loss', [])
            val_sisnr_s_loss = checkpoint.get('val_sisnr_s_loss', [])
            train_sisnr_m_loss = checkpoint.get('train_sisnr_m_loss', [])
            val_sisnr_m_loss = checkpoint.get('val_sisnr_m_loss', [])
            train_mix_recon_loss = checkpoint.get('train_mix_recon_loss', [])
            val_mix_recon_loss = checkpoint.get('val_mix_recon_loss', [])
            train_mel_loss = checkpoint.get('train_mel_loss', [])
            val_mel_loss = checkpoint.get('val_mel_loss', [])
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
            train_losses_total, val_losses_total = [], []  # Reset history
            train_sisnr_s_loss, val_sisnr_s_loss = [], []
            train_sisnr_m_loss, val_sisnr_m_loss = [], []
            train_mix_recon_loss, val_mix_recon_loss = [], []
            train_mel_loss, val_mel_loss = [], []
            val_pesq_hist, val_stoi_hist = [], []
            train_speech_si_snr, val_speech_si_snr = [], []
            train_music_si_snr, val_music_si_snr = [], []
            train_overall_si_snr, val_overall_si_snr = [], []

    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Epoch Loop ---
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # --- Update mel loss weight if using decay ---
        if args.mel_weight_decay and not args.disable_mel_loss:
            # Linear decay from initial value to 0 over mel_weight_decay_epochs
            if epoch < args.mel_weight_decay_epochs:
                current_mel_weight = args.mel_weight_initial * (1 - epoch / args.mel_weight_decay_epochs)
            else:
                current_mel_weight = 0.0
                
            # Update loss function's mel weight
            loss_fn.mel_loss_weight = current_mel_weight
            print(f"Epoch {epoch+1}: Mel loss weight decayed to {current_mel_weight:.4f}")
        
        model.train()
        # --- Reset epoch loss accumulators ---
        epoch_total_loss = 0.0
        epoch_loss_sisnr_speech = 0.0
        epoch_loss_sisnr_music = 0.0
        epoch_loss_mix_recon = 0.0
        epoch_loss_mel = 0.0
        num_train_samples = 0
        
        # Reset training metrics
        train_sisnr_metric.reset()
        
        # --- Training Batch Loop ---
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train",
                          total=min(len(train_loader), args.max_batches) if args.max_batches > 0 else len(train_loader))
        for i, (mixture, target_speech, target_music) in enumerate(train_pbar):
            if args.max_batches > 0 and i >= args.max_batches: break # Apply max_batches

            mixture_input = mixture.to(device) # (B, C, T)
            mixture_target = mixture.squeeze(1).to(device) if mixture.shape[1] == 1 else torch.mean(mixture, dim=1).to(device) # (B, T)

            # Targets [speech, music] order: (B, N_src, C, T)
            if target_speech.ndim == 3: 
                targets = torch.stack([target_speech, target_music], dim=1).to(device)
            else: 
                targets = torch.stack([target_speech.unsqueeze(1), target_music.unsqueeze(1)], dim=1).to(device)

            batch_size = mixture_input.shape[0]

            optimizer.zero_grad()
            s_estimates, x_mix_recon = model(mixture_input) # s_est:(B,N,C,T), mix_recon:(B,C,T)
            # Calculate batch losses using loss function instance
            batch_total_loss, batch_avg_loss_sisnr_speech, batch_loss_mix_recon, batch_loss_mel_speech, batch_avg_loss_sisnr_music = loss_fn(
                s_estimates, targets, x_mix_recon, mixture_target
            )

            # Update torchmetrics SI-SNR (handles multiple sources internally with PIT)
            # Need to convert shapes for torchmetrics format if needed and ensure device
            s_est_for_metric = s_estimates.to(device)
            targets_for_metric = targets.to(device)
            train_sisnr_metric.update(s_est_for_metric, targets_for_metric)

            batch_total_loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Accumulate per-sample losses/metrics
            epoch_total_loss += batch_total_loss.item() * batch_size
            epoch_loss_sisnr_speech += batch_avg_loss_sisnr_speech.item() * batch_size # This is neg SI-SNR loss
            epoch_loss_sisnr_music += batch_avg_loss_sisnr_music.item() * batch_size # This is neg SI-SNR loss
            epoch_loss_mix_recon += batch_loss_mix_recon.item() * batch_size
            epoch_loss_mel += batch_loss_mel_speech.item() * batch_size
            num_train_samples += batch_size

            # Update tqdm postfix (average over samples processed so far in epoch)
            speech_sisnr = -epoch_loss_sisnr_speech / num_train_samples
            music_sisnr = -epoch_loss_sisnr_music / num_train_samples
            train_pbar.set_postfix(
                loss=f"{epoch_total_loss / num_train_samples:.4f}",
                speech=f"{speech_sisnr:.2f}dB",
                music=f"{music_sisnr:.2f}dB",
                batch=f"{i+1}/{len(train_loader)}"
            )

        # Calculate average epoch losses/metrics
        avg_epoch_total_loss = epoch_total_loss / num_train_samples
        avg_epoch_loss_sisnr_speech = epoch_loss_sisnr_speech / num_train_samples
        avg_epoch_loss_sisnr_music = epoch_loss_sisnr_music / num_train_samples
        avg_epoch_loss_mix_recon = epoch_loss_mix_recon / num_train_samples
        avg_epoch_loss_mel = epoch_loss_mel / num_train_samples
        train_losses_total.append(avg_epoch_total_loss)
        train_sisnr_s_loss.append(avg_epoch_loss_sisnr_speech) # History list name kept for consistency with checkpoint loading
        train_sisnr_m_loss.append(avg_epoch_loss_sisnr_music) # History list name kept for consistency with checkpoint loading
        train_mix_recon_loss.append(avg_epoch_loss_mix_recon)
        train_mel_loss.append(avg_epoch_loss_mel)
        train_speech_si_snr.append(-avg_epoch_loss_sisnr_speech) # Convert loss to positive SI-SNR metric
        train_music_si_snr.append(-avg_epoch_loss_sisnr_music)
        
        # Compute and store the torchmetrics SI-SNR (with PIT)
        # Only compute if we've processed at least one batch
        if num_train_samples > 0:
            overall_sisnr = train_sisnr_metric.compute().item()
            train_overall_si_snr.append(overall_sisnr)
        else:
            # If no batches processed, use a default value
            train_overall_si_snr.append(0.0)
            overall_sisnr = 0.0

        # --- Validation ---
        print("\n=> Running validation...")
        val_results = evaluate(model, val_loader, loss_fn, eval_metrics, device,
                               desc=f"Epoch {epoch+1}/{args.epochs} Val",
                               max_batches=args.max_batches)
        avg_val_total_loss = val_results["Loss_Total"]
        val_losses_total.append(avg_val_total_loss)
        val_sisnr_s_loss.append(val_results['Loss_Speech(SI-SNR)'])
        val_mix_recon_loss.append(val_results['Loss_Mix(L1)'])
        val_mel_loss.append(val_results['Loss_Mel'])
        val_speech_si_snr.append(val_results['Speech_SI-SNR_Eval'])
        val_music_si_snr.append(val_results['Music_SI-SNR_Eval'])
        val_overall_si_snr.append(val_results['SI-SNR']) 
        if pesq_key and pesq_key in val_results: val_pesq_hist.append(val_results[pesq_key])
        if "STOI" in val_results: val_stoi_hist.append(val_results['STOI'])
        if args.include_music_loss and 'Loss_Music(SI-SNR)' in val_results:
            val_sisnr_m_loss.append(val_results['Loss_Music(SI-SNR)'])

        epoch_time = time.time() - start_time

        # --- Logging ---
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch+1}/{args.epochs} SUMMARY | Time: {epoch_time:.2f}s")
        print(f"{'-' * 60}")

        # Train metrics
        print(f"TRAIN | Loss: {avg_epoch_total_loss:.4f}")
        print(f"       Mix Recon: {avg_epoch_loss_mix_recon:.4f} | Mel: {avg_epoch_loss_mel:.4f}")
        print(f"       Speech SI-SNR: {-avg_epoch_loss_sisnr_speech:.2f} dB | Music SI-SNR: {-avg_epoch_loss_sisnr_music:.2f} dB")
        print(f"       Overall SI-SNR: {overall_sisnr:.2f} dB")

        # Validation metrics
        print(f"{'-' * 60}")
        print(f"VAL   | Loss: {avg_val_total_loss:.4f}" + (" (NEW BEST) ✓" if avg_val_total_loss < best_val_loss else ""))
        print(f"       Speech SI-SNR: {val_results['Speech_SI-SNR_Eval']:.2f} dB | Music SI-SNR: {val_results['Music_SI-SNR_Eval']:.2f} dB")
        print(f"       Mix Recon: {val_results['Loss_Mix(L1)']:.4f} | Mel: {val_results['Loss_Mel']:.4f}")
        print(f"       Overall SI-SNR: {val_results['SI-SNR']:.2f} dB")

        if pesq_key in val_results: 
            print(f"       {pesq_key}: {val_results[pesq_key]:.2f}")
        if "STOI" in val_results: 
            print(f"       STOI: {val_results['STOI']:.3f}")

        print(f"{'=' * 60}")

        # --- TensorBoard Logging ---
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Training/learning_rate', current_lr, epoch+1)
        
        # Log mel spectrogram weight if using decay
        if args.mel_weight_decay and not args.disable_mel_loss:
            writer.add_scalar('Training/mel_weight', loss_fn.mel_loss_weight, epoch+1)

        # Log training metrics
        writer.add_scalar('Loss/train_total', avg_epoch_total_loss, epoch+1)
        writer.add_scalar('Loss/train_speech_sisnr', avg_epoch_loss_sisnr_speech, epoch+1)
        writer.add_scalar('Loss/train_mix_recon', avg_epoch_loss_mix_recon, epoch+1)
        writer.add_scalar('Loss/train_mel', avg_epoch_loss_mel, epoch+1)

        # If music loss is included in backpropagation, log it as a loss component
        if args.include_music_loss:
            writer.add_scalar('Loss/train_music_sisnr', avg_epoch_loss_sisnr_music, epoch+1)

        # Log performance metrics (positive values for SI-SNR)
        writer.add_scalar('Metrics/train_speech_sisnr', -avg_epoch_loss_sisnr_speech, epoch+1)
        writer.add_scalar('Metrics/train_music_sisnr', -avg_epoch_loss_sisnr_music, epoch+1)
        writer.add_scalar('Metrics/train_overall_sisnr', overall_sisnr, epoch+1)

        # Log validation metrics
        writer.add_scalar('Loss/val_total', avg_val_total_loss, epoch+1)
        writer.add_scalar('Loss/val_speech_sisnr', val_results['Loss_Speech(SI-SNR)'], epoch+1)
        writer.add_scalar('Loss/val_mix_recon', val_results['Loss_Mix(L1)'], epoch+1)
        writer.add_scalar('Loss/val_mel', val_results['Loss_Mel'], epoch+1)

        # Log music SI-SNR loss if included in training
        if args.include_music_loss and 'Loss_Music(SI-SNR)' in val_results:
            writer.add_scalar('Loss/val_music_sisnr', val_results['Loss_Music(SI-SNR)'], epoch+1)

        # Log performance metrics
        writer.add_scalar('Metrics/val_speech_sisnr', val_results['Speech_SI-SNR_Eval'], epoch+1)
        writer.add_scalar('Metrics/val_music_sisnr', val_results['Music_SI-SNR_Eval'], epoch+1)
        writer.add_scalar('Metrics/val_overall_sisnr', val_results['SI-SNR'], epoch+1)

        # Log additional metrics if available
        if pesq_key in val_results:
            writer.add_scalar(f'Metrics/val_{pesq_key}', val_results[pesq_key], epoch+1)
        if "STOI" in val_results:
            writer.add_scalar('Metrics/val_STOI', val_results['STOI'], epoch+1)

        # Generate and save plots for current epoch
        fig = plot_losses(train_losses_total, val_losses_total, output_dir / f'loss_plot_epoch_{epoch+1}.png', {
            'speech_sisnr': train_sisnr_s_loss, 
            'music_sisnr': train_sisnr_m_loss if train_sisnr_m_loss else [],
            'mix_recon': train_mix_recon_loss, 
            'mel': train_mel_loss
        }, {
            'speech_sisnr': val_sisnr_s_loss, 
            'music_sisnr': val_sisnr_m_loss if val_sisnr_m_loss else [],
            'mix_recon': val_mix_recon_loss, 
            'mel': val_mel_loss
        }, args.include_music_loss)
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

                # Get both separated sources and reconstructed mixture
                s_est_sample, x_mix_recon_sample = model(mixture_tensor)

                # Save audio files
                save_audio(s_est_sample[:, 0], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_speech_est.wav'), args.sr)
                save_audio(s_est_sample[:, 1], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_music_est.wav'), args.sr)
                save_audio(x_mix_recon_sample, os.path.join(samples_dir, f'epoch_{epoch+1:03d}_mixture_recon.wav'), args.sr)

                # Save spectrograms
                save_spectrogram(
                    s_est_sample[:, 0], 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Estimated Speech (Epoch {epoch+1})"
                )
                save_spectrogram(
                    s_est_sample[:, 1], 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Estimated Music (Epoch {epoch+1})"
                )
                save_spectrogram(
                    x_mix_recon_sample, 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_mixture_recon_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Reconstructed Mixture (Epoch {epoch+1})"
                )

                # Log audio to TensorBoard
                try:
                    # Make sure tensor is in the right format (batch-first) and on CPU
                    speech_est_cpu = s_est_sample[:, 0].cpu()
                    music_est_cpu = s_est_sample[:, 1].cpu()
                    mix_recon_cpu = x_mix_recon_sample.cpu()

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
                    writer.add_audio(
                        'Audio/reconstructed_mixture', 
                        mix_recon_cpu.squeeze(), 
                        global_step=epoch+1, 
                        sample_rate=args.sr
                    )

                    # Also log spectrograms as images
                    # Use the paths of the saved images
                    speech_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'))
                    music_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_music_est_spec.png'))
                    mix_spec_img = plt.imread(os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_mixture_recon_spec.png'))

                    writer.add_image('Spectrograms/speech', speech_spec_img, epoch+1, dataformats='HWC')
                    writer.add_image('Spectrograms/music', music_spec_img, epoch+1, dataformats='HWC')
                    writer.add_image('Spectrograms/mixture', mix_spec_img, epoch+1, dataformats='HWC')
                except Exception as e:
                    print(f"Warning: Could not log audio to TensorBoard: {e}")

        # --- Save Checkpoint (Every Epoch) ---
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_val_total_loss,  # Use average total val loss for checkpoint loss value
                "best_val_loss": best_val_loss,
                "train_losses_total": train_losses_total,
                "val_losses_total": val_losses_total,
                "train_sisnr_s_loss": train_sisnr_s_loss,
                "val_sisnr_s_loss": val_sisnr_s_loss,
                "train_sisnr_m_loss": train_sisnr_m_loss,
                "val_sisnr_m_loss": val_sisnr_m_loss,
                "train_mix_recon_loss": train_mix_recon_loss,
                "val_mix_recon_loss": val_mix_recon_loss,
                "train_mel_loss": train_mel_loss,
                "val_mel_loss": val_mel_loss,
                "val_pesq_hist": val_pesq_hist,
                "val_stoi_hist": val_stoi_hist,
                "train_speech_si_snr": train_speech_si_snr,
                "val_speech_si_snr": val_speech_si_snr,
                "train_music_si_snr": train_music_si_snr,
                "val_music_si_snr": val_music_si_snr,
                "train_overall_si_snr": train_overall_si_snr,
                "val_overall_si_snr": val_overall_si_snr,
                "val_metrics": val_results,  # Save last validation metrics dict
                "args": vars(args),
            },
            checkpoint_path,
        )
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
            print(f"\n★ SAVED BEST MODEL ★ to {best_model_save_path}")
            print(f"  New best validation loss: {avg_val_total_loss:.4f} (previous: {best_val_loss:.4f})")

    # --- Plotting after training ---
    epochs_ran = list(range(1, len(train_losses_total) + 1)) # Use actual length of history
    if len(train_losses_total) == len(epochs_ran): # Ensure history length matches epochs
        fig = plot_losses(
            train_losses_total,
            val_losses_total,
            output_dir / "loss_plot.png",
            {
                "speech_sisnr": train_sisnr_s_loss,
                "music_sisnr": train_sisnr_m_loss if train_sisnr_m_loss else [],
                "mix_recon": train_mix_recon_loss,
                "mel": train_mel_loss,
            },
            {
                "speech_sisnr": val_sisnr_s_loss,
                "music_sisnr": val_sisnr_m_loss if val_sisnr_m_loss else [],
                "mix_recon": val_mix_recon_loss,
                "mel": val_mel_loss,
            },
            args.include_music_loss,
        )
        writer.add_figure('Plots/loss', fig, 0)
        # Need to adapt plot_si_snr to use new history lists
        fig = plot_si_snr(epochs_ran, train_speech_si_snr, train_music_si_snr, val_speech_si_snr, val_music_si_snr, train_overall_si_snr, val_overall_si_snr, output_dir / 'si_snr_plot.png')
        writer.add_figure('Plots/si_snr', fig, 0)
    else:
        print("Warning: History length is zero, skipping plotting.")

    # --- Final Testing ---
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE ".center(70, "="))
    print("=" * 70)

    print("\nStarting final testing...")
    best_model_path = output_dir / "best_model.pth" # Reusing variable name from save
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model_args_dict = checkpoint.get('args', vars(args))
        # Ensure all args are available, fallback to current args if necessary
        current_args_dict = vars(args)
        final_model_args = {**current_args_dict, **model_args_dict} # Prioritize args from checkpoint

        test_model = MSHybridNet(
            channels=1,  # Assuming mono
            enc_kernel_size=final_model_args["enc_kernel_size"],
            enc_stride=final_model_args["enc_stride"],
            enc_features=final_model_args["features"],
            num_blocks=final_model_args["num_blocks"],
            tcn_hidden_channels=final_model_args["tcn_hidden_channels"],
            tcn_kernel_size=final_model_args["tcn_kernel_size"],
            tcn_layers_per_block=final_model_args["tcn_layers_per_block"],
            tcn_dilation_base=final_model_args["tcn_dilation_base"],
            conformer_dim=final_model_args["conformer_dim"],
            conformer_heads=final_model_args["conformer_heads"],
            conformer_kernel_size=final_model_args["conformer_kernel_size"],
            conformer_ffn_expansion=final_model_args["conformer_ffn_expansion"],
            conformer_dropout=final_model_args["conformer_dropout"],
         ).to(device)
        test_model.load_state_dict(checkpoint['model_state_dict'])

        try:
            test_dataset = FolderTripletDataset(root_path, split="test", segment=args.segment, sr=args.sr)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) # Batch size 1 for test
            print(f"Test samples: {len(test_dataset)}")

            # Recreate metrics for testing using the same pesq_mode and pesq_key
            test_metric_list = {}
            test_metric_list["SI-SNR"] = torchmetrics.audio.ScaleInvariantSignalNoiseRatio().to(device)
            if pesq_mode:
                test_metric_list[pesq_key] = torchmetrics.audio.pesq.PerceptualEvaluationSpeechQuality(args.sr, pesq_mode).to(device)
            test_metric_list["STOI"] = torchmetrics.audio.stoi.ShortTimeObjectiveIntelligibility(args.sr, extended=False).to(device)
            test_metrics = torchmetrics.MetricCollection(test_metric_list).to(device)

            # Recreate loss function for reporting loss on test set
            test_loss_fn = MSHybridAEDirectLoss(
                mix_recon_weight=args.mix_recon_weight,
                mel_loss_weight=loss_fn.mel_loss_weight if 'loss_fn' in locals() else (0.0 if args.disable_mel_loss else args.mel_weight),
                speech_target_index=0, # Speech=0, Music=1
                music_target_index=1,
                include_music_loss=args.include_music_loss,
                sample_rate=args.sr,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels
             ).to(device)

            # Run evaluation on test set using the test dataloader
            test_results = evaluate(test_model, test_loader, test_loss_fn, test_metrics, device, 
                                   desc="Testing (best model)", max_batches=0) # Pass test_loader

            print("\n" + "=" * 70)
            print(" FINAL TEST RESULTS ".center(70, "="))
            print("=" * 70)
            print(f"\nModel: MSHybridNet @ Epoch {checkpoint['epoch']}")
            print(f"\nLOSS METRICS:")
            print(f"  Total Loss:       {test_results['Loss_Total']:.4f}")
            print(f"  Speech SI-SNR:    {test_results['Loss_Speech(SI-SNR)']:.4f}")
            print(f"  Mix Recon (L1):   {test_results['Loss_Mix(L1)']:.4f}")
            print(f"  Mel Loss:         {test_results['Loss_Mel']:.4f}")
            if args.include_music_loss: 
                print(f"  Music SI-SNR:     {test_results['Loss_Music(SI-SNR)']:.4f}")

            print(f"\nPERFORMANCE METRICS:")
            print(f"  SI-SNR (Overall PIT): {test_results['SI-SNR']:.2f} dB")
            print(f"  Speech SI-SNR:        {test_results['Speech_SI-SNR_Eval']:.2f} dB")
            print(f"  Music SI-SNR:         {test_results['Music_SI-SNR_Eval']:.2f} dB")
            if pesq_key in test_results: 
                print(f"  {pesq_key}:             {test_results[pesq_key]:.2f}")
            if "STOI" in test_results: 
                print(f"  STOI:                 {test_results['STOI']:.3f}")
            print("=" * 70)

            # Log test results to TensorBoard
            writer.add_scalar('Test/Loss_Total', test_results['Loss_Total'], 0)
            writer.add_scalar('Test/Loss_Speech_SISNR', test_results['Loss_Speech(SI-SNR)'], 0)
            writer.add_scalar('Test/Loss_Mix_L1', test_results['Loss_Mix(L1)'], 0)
            writer.add_scalar('Test/Loss_Mel', test_results['Loss_Mel'], 0)

            writer.add_scalar('Test/SI-SNR_Overall', test_results['SI-SNR'], 0)
            writer.add_scalar('Test/Speech_SI-SNR', test_results['Speech_SI-SNR_Eval'], 0)
            writer.add_scalar('Test/Music_SI-SNR', test_results['Music_SI-SNR_Eval'], 0)

            if pesq_key in test_results:
                writer.add_scalar(f'Test/{pesq_key}', test_results[pesq_key], 0)
            if "STOI" in test_results:
                writer.add_scalar('Test/STOI', test_results['STOI'], 0)

            # Add text summary of test results
            test_summary = (
                f"Loss Total: {test_results['Loss_Total']:.4f}\n"
                f"Speech SI-SNR: {test_results['Speech_SI-SNR_Eval']:.2f} dB\n"
                f"Music SI-SNR: {test_results['Music_SI-SNR_Eval']:.2f} dB\n"
            )
            if pesq_key in test_results:
                test_summary += f"{pesq_key}: {test_results[pesq_key]:.2f}\n"
            if "STOI" in test_results:
                test_summary += f"STOI: {test_results['STOI']:.3f}\n"

            writer.add_text('Test/Summary', test_summary, 0)

            # --- Save 20 Random Test Samples and Spectrograms ---
            num_test_samples = 20
            test_audio_dir = os.path.join(args.save_dir, 'test_audio_samples')
            test_spec_dir = os.path.join(args.save_dir, 'test_spectrograms')
            os.makedirs(test_audio_dir, exist_ok=True)
            os.makedirs(test_spec_dir, exist_ok=True)

            print(f"\nSelecting {num_test_samples} random test samples for evaluation...")

            # Get random indices that are well-distributed
            total_samples = len(test_dataset)
            if total_samples <= num_test_samples:
                # If we have fewer than 20 samples, use all of them
                selected_indices = list(range(total_samples))
                print(f"Using all {total_samples} test samples (fewer than requested {num_test_samples})")
            else:
                # Create evenly spaced indices across the dataset, then add some randomness
                step = total_samples // num_test_samples
                base_indices = list(range(0, total_samples, step))[:num_test_samples]

                # Add some randomness to avoid perfectly even spacing
                selected_indices = [min(max(0, idx + np.random.randint(-step//4, step//4)), total_samples-1) 
                                    for idx in base_indices]

                # Ensure we have the right number and no duplicates
                while len(set(selected_indices)) < num_test_samples and len(set(selected_indices)) < total_samples:
                    missing = num_test_samples - len(set(selected_indices))
                    extra_indices = np.random.choice(
                        [i for i in range(total_samples) if i not in selected_indices],
                        size=missing,
                        replace=False
                    ).tolist()
                    selected_indices.extend(extra_indices)

                selected_indices = sorted(list(set(selected_indices)))[:num_test_samples]
                print(f"Selected {len(selected_indices)} test samples at indices: {selected_indices}")

            # Process each selected sample
            test_model.eval()
            with torch.no_grad():
                for i, sample_idx in enumerate(selected_indices):
                    # Get the sample
                    mixture, target_speech, target_music = test_dataset[sample_idx]

                    # Make sure tensors are properly shaped (B,C,T)
                    mixture_input = mixture.clone()
                    if mixture_input.ndim == 2:  # (C,T)
                        mixture_input = mixture_input.unsqueeze(0)  # Add batch dim -> (1,C,T)
                    mixture_input = mixture_input.to(device)

                    # Process through model
                    s_estimates, x_mix_recon = test_model(mixture_input)

                    # Create folder for this sample
                    sample_dir = os.path.join(test_audio_dir, f'sample_{i+1:02d}')
                    sample_spec_dir = os.path.join(test_spec_dir, f'sample_{i+1:02d}')
                    os.makedirs(sample_dir, exist_ok=True)
                    os.makedirs(sample_spec_dir, exist_ok=True)

                    # Save ground truth audio
                    save_audio(mixture, os.path.join(sample_dir, 'gt_mixture.wav'), args.sr)
                    save_audio(target_speech, os.path.join(sample_dir, 'gt_speech.wav'), args.sr)
                    save_audio(target_music, os.path.join(sample_dir, 'gt_music.wav'), args.sr)

                    # Save estimated audio
                    save_audio(s_estimates[:, 0], os.path.join(sample_dir, 'est_speech.wav'), args.sr)
                    save_audio(s_estimates[:, 1], os.path.join(sample_dir, 'est_music.wav'), args.sr)
                    save_audio(x_mix_recon, os.path.join(sample_dir, 'est_mixture.wav'), args.sr)

                    # # Calculate model outputs for speech/music separately
                    # if hasattr(test_model, 'get_speech'):
                    #     speech_direct = test_model.get_speech(mixture_input)
                    #     save_audio(speech_direct, os.path.join(sample_dir, 'est_speech_direct.wav'), args.sr)

                    # if hasattr(test_model, 'get_music'):
                    #     music_direct = test_model.get_music(mixture_input)
                    #     save_audio(music_direct, os.path.join(sample_dir, 'est_music_direct.wav'), args.sr)

                    # Save spectrograms
                    # Ground truth spectrograms
                    save_spectrogram(mixture, 
                                    os.path.join(sample_spec_dir, 'gt_mixture_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Ground Truth Mixture (Sample {i+1})")
                    save_spectrogram(target_speech, 
                                    os.path.join(sample_spec_dir, 'gt_speech_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Ground Truth Speech (Sample {i+1})")
                    save_spectrogram(target_music, 
                                    os.path.join(sample_spec_dir, 'gt_music_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Ground Truth Music (Sample {i+1})")

                    # Estimated spectrograms
                    save_spectrogram(s_estimates[:, 0], 
                                    os.path.join(sample_spec_dir, 'est_speech_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Speech (Sample {i+1})")
                    save_spectrogram(s_estimates[:, 1], 
                                    os.path.join(sample_spec_dir, 'est_music_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Music (Sample {i+1})")
                    save_spectrogram(x_mix_recon, 
                                    os.path.join(sample_spec_dir, 'est_mixture_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Mixture (Sample {i+1})")

                    # # Direct output spectrograms if available
                    # if hasattr(test_model, 'get_speech'):
                    #     save_spectrogram(speech_direct,
                    #                     os.path.join(sample_spec_dir, 'est_speech_direct_spec.png'),
                    #                     args.sr, args.n_fft, args.hop_length,
                    #                     title=f"Direct Speech Output (Sample {i+1})")

                    # if hasattr(test_model, 'get_music'):
                    #     save_spectrogram(music_direct,
                    #                     os.path.join(sample_spec_dir, 'est_music_direct_spec.png'),
                    #                     args.sr, args.n_fft, args.hop_length,
                    #                     title=f"Direct Music Output (Sample {i+1})")

                    # Calculate SI-SNR for this specific sample
                    speech_est = s_estimates[:, 0].squeeze()
                    music_est = s_estimates[:, 1].squeeze()
                    speech_gt = target_speech.to(device)
                    music_gt = target_music.to(device)

                    # Ensure they're 1D for si_snr_loss_manual (which handles shape internally)
                    if speech_est.ndim > 1: speech_est = speech_est.mean(0)
                    if music_est.ndim > 1: music_est = music_est.mean(0)
                    if speech_gt.ndim > 1: speech_gt = speech_gt.mean(0)
                    if music_gt.ndim > 1: music_gt = music_gt.mean(0)

                    # Calculate SI-SNR values
                    speech_si_snr = -si_snr_loss_manual(speech_est, speech_gt).item()
                    music_si_snr = -si_snr_loss_manual(music_est, music_gt).item()

                    # Save metrics to a text file for this sample
                    with open(os.path.join(sample_dir, 'metrics.txt'), 'w') as f:
                        f.write(f"Sample {i+1} (Dataset index: {sample_idx})\n")
                        f.write(f"Speech SI-SNR: {speech_si_snr:.2f} dB\n")
                        f.write(f"Music SI-SNR: {music_si_snr:.2f} dB\n")
                        f.write(f"Average SI-SNR: {(speech_si_snr + music_si_snr)/2:.2f} dB\n")

                    print(f"Saved sample {i+1}/{len(selected_indices)} (idx {sample_idx}): " + 
                          f"Speech SI-SNR: {speech_si_snr:.2f} dB, Music SI-SNR: {music_si_snr:.2f} dB")

                print(f"\nSaved {len(selected_indices)} test samples with audio and spectrograms to:")
                print(f"  Audio: {test_audio_dir}")
                print(f"  Spectrograms: {test_spec_dir}")

        except (ValueError, FileNotFoundError) as e:
             print(f"Could not create/run test dataset/loader: {e}")
             print("Skipping final testing.")
    else:
        print("No best model checkpoint found. Skipping final testing.")

    # Close TensorBoard writer
    writer.close()


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WFAE-Inspired MSHybridNet without PIT")

    # --- Data Arguments ---
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--segment', type=float, default=3.0, help='Duration of audio segments in seconds')
    parser.add_argument('--sr', type=int, default=8000, help='Sample rate (PESQ requires 8k or 16k)')

    # --- Training Arguments ---
    parser.add_argument('--save_dir', type=str, default='checkpoints_wfae_direct', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size') # Keep small for memory
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N batches')
    parser.add_argument('--max_batches', type=int, default=0, help='Maximum number of batches per epoch (0 for all)')

    # --- Loss Arguments ---
    parser.add_argument('--mix_recon_weight', type=float, default=0.1, help='Weight for Mixture Reconstruction L1 loss component')
    parser.add_argument('--mel_weight', type=float, default=0.05, help='Weight for Mel Spectrogram loss component for speech')
    parser.add_argument('--disable_mel_loss', action='store_true', help='Disable Mel Spectrogram loss (overrides mel_weight)')
    parser.add_argument('--mel_weight_decay', action='store_true', help='Enable linear decay of mel weight over epochs')
    parser.add_argument('--mel_weight_initial', type=float, default=5.0, help='Initial mel weight when using decay')
    parser.add_argument('--mel_weight_decay_epochs', type=int, default=5, help='Epoch by which mel weight reaches 0')
    parser.add_argument('--include_music_loss', action='store_true', help='Include SI-SNR loss for music in backpropagation')
    parser.add_argument('--n_fft', type=int, default=512, help='FFT size for Mel Spectrogram Loss')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length for Mel Spectrogram Loss')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of Mel bins for Mel Spectrogram Loss')

    # --- Model Hyperparameters ---
    # Match the MSHybridNet.__init__ arguments
    parser.add_argument('--enc_kernel_size', type=int, default=16)
    parser.add_argument('--enc_stride', type=int, default=8)
    parser.add_argument('--features', type=int, default=128, help='Encoder feature dimension (must match conformer_dim)')
    parser.add_argument('--num_blocks', type=int, default=4, help='Number of hybrid blocks')
    parser.add_argument('--tcn_hidden_channels', type=int, default=256, help='TCN hidden channels')
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN kernel size')
    parser.add_argument('--tcn_layers_per_block', type=int, default=8, help='TCN layers per block')
    parser.add_argument('--tcn_dilation_base', type=int, default=2, help='TCN dilation base')
    parser.add_argument('--conformer_dim', type=int, default=128, help='Conformer dimension (must match features)')
    parser.add_argument('--conformer_heads', type=int, default=4, help='Conformer heads')
    parser.add_argument('--conformer_kernel_size', type=int, default=31, help='Conformer kernel size')
    parser.add_argument('--conformer_ffn_expansion', type=int, default=4, help='Conformer FFN expansion factor')
    parser.add_argument('--conformer_dropout', type=float, default=0.1, help='Conformer dropout rate')

    # --- Eval/Debug Arguments ---
    parser.add_argument('--sample_index', type=int, default=-1, help='Index of validation sample to save audio for (-1 for random)')
    parser.add_argument('--test_samples', type=int, default=5, help='Number of test samples to save during final evaluation (0 to disable)')
    parser.add_argument('--test_samples_random', action='store_true', help='Use random samples for test audio instead of sequential')

    args = parser.parse_args()

    # --- Basic Validation ---
    if args.sr not in [8000, 16000]:
        raise ValueError("PESQ metric requires sr=8000 or sr=16000")
    if args.features != args.conformer_dim:
        raise ValueError(f"Features ({args.features}) must equal conformer_dim ({args.conformer_dim}) for this model configuration.")
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    train(args)
