# train.py (Modified with Checkpointing, Max Batches, Validation Sample Saving)

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
from Conv_TasNet import ConvTasNet
from audio_datasets import FolderTripletDataset
# Using torchmetrics for SI-SNR, PESQ, STOI
import torchmetrics

# --- Loss Functions ---

# Mel loss removed as it's not needed for ConvTasNet


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


# Simple SI-SNR Loss for ConvTasNet
class ConvTasNetLoss(nn.Module):
    """
    Loss for ConvTasNet using SI-SNR for both speech and music.
    Assumes fixed order: Source 0 = Speech, Source 1 = Music.
    """
    def __init__(self, speech_target_index=0, music_target_index=1, sample_rate=8000):
        super().__init__()
        self.speech_target_index = speech_target_index
        self.music_target_index = music_target_index
        self.sample_rate = sample_rate
        self.include_music_loss = True  # Always include music loss

    def to(self, device):
        super().to(device)
        return self

    def forward(self, s_estimates, targets, mix_estimate=None, mix_target=None):
        # s_estimates from ConvTasNet: list of tensors [speech, music] each of shape (batch_size, time)
        # targets: (batch_size, num_sources, channels, time)
        device = s_estimates[0].device
        
        # Ensure targets are on the same device
        targets = targets.to(device)

        # Extract speech and music targets and ensure proper dimensions
        if targets.dim() == 4:  # (B, N_src, C, T)
            if targets.shape[2] == 1:  # Single channel
                speech_tgt = targets[:, self.speech_target_index, 0]  # (B, T)
                music_tgt = targets[:, self.music_target_index, 0]    # (B, T)
            else:  # Multiple channels, average
                speech_tgt = targets[:, self.speech_target_index].mean(dim=1)  # (B, T)
                music_tgt = targets[:, self.music_target_index].mean(dim=1)    # (B, T)
        else:  # Something is wrong with the input
            raise ValueError(f"Unexpected target shape: {targets.shape}")

        # ConvTasNet outputs are already in the right format (B, T)
        speech_est = s_estimates[0]  # (B, T)
        music_est = s_estimates[1]   # (B, T)

        # Calculate SI-SNR losses
        batch_loss_sisnr_speech = si_snr_loss_manual(speech_est, speech_tgt)  # (B,)
        avg_loss_sisnr_speech = torch.mean(batch_loss_sisnr_speech)  # Scalar

        batch_loss_sisnr_music = si_snr_loss_manual(music_est, music_tgt)  # (B,)
        avg_loss_sisnr_music = torch.mean(batch_loss_sisnr_music)  # Scalar

        # Total loss is the sum of both SI-SNR losses
        total_loss = avg_loss_sisnr_speech + avg_loss_sisnr_music
        
        return total_loss, avg_loss_sisnr_speech, avg_loss_sisnr_music


# --- Evaluation Function (Modified for Direct Loss & Max Batches) ---
def evaluate(model, dataloader, loss_fn_eval, metrics_dict, device, desc="Evaluating", max_batches=0):
    """ Evaluate model on a dataloader (Adapted for ConvTasNet) """
    model.eval()
    total_loss_sisnr_speech = 0.0
    total_loss_sisnr_music = 0.0
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

            # Prepare input for ConvTasNet (expects B, T or T)
            if mixture.dim() == 3:  # (B, C, T)
                if mixture.shape[1] == 1:
                    mixture_input = mixture.squeeze(1).to(device)  # (B, T)
                else:
                    mixture_input = mixture.mean(dim=1).to(device)  # (B, T) - average channels
            else:
                mixture_input = mixture.to(device)  # Already (B, T)

            # Targets: Stack and ensure correct shape (B, N_src, C, T)
            if target_speech.ndim == 3:  # Already (B, C, T)
                targets = torch.stack([target_speech, target_music], dim=1).to(device)
            else:  # Assume (B, T), add channel dim
                targets = torch.stack([target_speech.unsqueeze(1), target_music.unsqueeze(1)], dim=1).to(device)

            batch_size = mixture_input.shape[0]
            num_samples += batch_size

            # Model forward pass -> returns list [speech_est, music_est] each (B, T)
            s_estimates = model(mixture_input)

            # Calculate loss components using the loss function
            _, batch_avg_loss_sisnr_speech, batch_avg_loss_sisnr_music = loss_fn_eval(
                s_estimates, targets, None, None
            )
            
            total_loss_sisnr_speech += batch_avg_loss_sisnr_speech.item() * batch_size
            total_loss_sisnr_music += batch_avg_loss_sisnr_music.item() * batch_size
            accumulated_music_sisnr_metric += (-batch_avg_loss_sisnr_music.item()) * batch_size  # SI-SNR metric is neg of loss
            accumulated_speech_sisnr_metric += (-batch_avg_loss_sisnr_speech.item()) * batch_size  # SI-SNR metric is neg of loss

            # Update progress bar with current metrics
            if num_samples > 0:
                eval_pbar.set_postfix(
                    speech=f"{accumulated_speech_sisnr_metric / num_samples:.2f}dB",
                    music=f"{accumulated_music_sisnr_metric / num_samples:.2f}dB"
                )

            if "SI-SNR" in metrics_dict:
                try:
                    # ConvTasNet outputs a list of source estimates
                    # Convert to the format expected by torchmetrics SI-SNR [B, S, T]
                    # Where S is the number of sources
                    s_est_stacked = torch.stack([
                        s_estimates[0],  # speech estimates
                        s_estimates[1]   # music estimates
                    ], dim=1).to(device)
                    
                    # Convert targets to the right format for SI-SNR metric
                    if targets.dim() == 4:  # (B, S, C, T)
                        if targets.shape[2] == 1:
                            targets_for_metric = targets.squeeze(2).to(device)  # (B, S, T)
                        else:
                            targets_for_metric = targets.mean(dim=2).to(device)  # (B, S, T)
                    else:
                        targets_for_metric = targets.to(device)
                    
                    # Update metric with prepared tensors
                    metrics_dict["SI-SNR"].update(s_est_stacked, targets_for_metric)
                except Exception as e:
                    print(f"Warning: SI-SNR calculation failed: {e}")
                    
            # --- Get estimated sources for metric calculation ---
            speech_est = s_estimates[loss_fn_eval.speech_target_index]  # (B, T)
            
            # --- Get target sources ---
            if targets.dim() == 4:  # (B, N_src, C, T)
                if targets.shape[2] == 1:  # Single channel
                    speech_tgt = targets[:, loss_fn_eval.speech_target_index, 0]  # (B, T)
                else:  # Multiple channels, average
                    speech_tgt = targets[:, loss_fn_eval.speech_target_index].mean(dim=1)  # (B, T)
            else:
                raise ValueError(f"Unexpected target shape: {targets.shape}")
            
            # Update PESQ/STOI for speech only
            # ConvTasNet outputs are already (B, T)
            min_len = min(speech_est.shape[-1], speech_tgt.shape[-1])
            speech_est_trim = speech_est[..., :min_len]
            speech_tgt_trim = speech_tgt[..., :min_len]

            # REMOVED: Validation check that was causing IndexError
            # Use all samples directly for metrics calculation
            sample_rate = getattr(loss_fn_eval, 'sample_rate', 8000)
            pesq_key = f"PESQ-{'WB' if sample_rate == 16000 else 'NB'}" # Access sr from loss
            
            if pesq_key in metrics_dict:
                try: metrics_dict[pesq_key].update(speech_est_trim, speech_tgt_trim)
                except Exception as e: print(f"Warning: PESQ calculation failed: {e}")
            
            if "STOI" in metrics_dict:
                try: metrics_dict["STOI"].update(speech_est_trim, speech_tgt_trim)
                except Exception as e: print(f"Warning: STOI calculation failed: {e}")

    # Compute final metric values
    if num_samples > 0:
        results = {name: metric.compute().item() for name, metric in metrics_dict.items()}
        avg_loss_sisnr_speech = total_loss_sisnr_speech / num_samples
        avg_loss_sisnr_music = total_loss_sisnr_music / num_samples
        results["Loss_Speech(SI-SNR)"] = avg_loss_sisnr_speech
        results["Loss_Music(SI-SNR)"] = avg_loss_sisnr_music
        
        # Calculate total loss (sum of speech and music SI-SNR losses)
        results["Loss_Total"] = avg_loss_sisnr_speech + avg_loss_sisnr_music

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
        results["Loss_Total"] = 0.0
        results["Speech_SI-SNR_Eval"] = 0.0
        results["Music_SI-SNR_Eval"] = 0.0
        if pesq_key in metrics_dict:
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

def plot_losses(train_losses, val_losses, save_path, train_losses_components=None, val_losses_components=None, include_music_loss=True):
    """
    Plot training and validation losses and save the figure.
    Also plots individual loss components if provided.
    
    Args:
        train_losses: List of total training losses
        val_losses: List of total validation losses
        save_path: Path to save the figure
        train_losses_components: Dict of component losses for training (speech_sisnr, music_sisnr)
        val_losses_components: Dict of component losses for validation
        include_music_loss: Whether music loss is included (always True for ConvTasNet)
    """
    if not train_losses or not val_losses:
        print("Warning: Loss history is empty, skipping plotting.")
        return None
    
    # Always 2 component plots for ConvTasNet (speech SI-SNR and music SI-SNR)
    component_plots = 2
    
    fig = plt.figure(figsize=(12, 12))
    
    # Total loss plot (always included)
    plt.subplot(3, 1, 1)
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
    
    # Component loss plots (speech and music SI-SNR)
    if train_losses_components and val_losses_components:
        # Ensure all lists have the proper length
        epochs_len = len(epochs)
        
        # Get loss components, ensuring proper length
        train_speech_sisnr = train_losses_components.get('speech_sisnr', [0] * epochs_len)[:epochs_len]
        val_speech_sisnr = val_losses_components.get('speech_sisnr', [0] * epochs_len)[:epochs_len]
        train_music_sisnr = train_losses_components.get('music_sisnr', [0] * epochs_len)[:epochs_len]
        val_music_sisnr = val_losses_components.get('music_sisnr', [0] * epochs_len)[:epochs_len]
        
        # Speech SI-SNR Loss
        plt.subplot(3, 1, 2)
        plt.plot(epochs, train_speech_sisnr, 'b-', label='Train Speech SI-SNR Loss')
        plt.plot(epochs, val_speech_sisnr, 'r-', label='Val Speech SI-SNR Loss')
        plt.title('Speech SI-SNR Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Music SI-SNR Loss
        plt.subplot(3, 1, 3)
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
    model = ConvTasNet(
        N=args.N,               # Number of filters in autoencoder
        L=args.L,               # Length of the filters (in samples)
        B=args.B,               # Number of channels in bottleneck and residual paths
        H=args.H,               # Number of channels in convolutional blocks
        P=args.P,               # Kernel size in convolutional blocks
        X=args.X,               # Number of convolutional blocks in each repeat
        R=args.R,               # Number of repeats
        norm=args.norm,         # Normalization type
        num_spks=2,             # Number of speakers (fixed at 2: speech and music)
        activate=args.activate, # Activation function
        causal=args.causal      # Causal or non-causal model
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Loss Function (Simple SI-SNR loss for ConvTasNet) ---
    loss_fn = ConvTasNetLoss(
        speech_target_index=0,  # Speech is first source
        music_target_index=1,   # Music is second source
        sample_rate=args.sr     # Keep sample rate for compatibility
    ).to(device)

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
            # Skip loading mel and recon losses for ConvTasNet
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
            # ConvTasNet doesn't use these losses
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
        
        # --- No mel loss decay for ConvTasNet (SI-SNR only) ---
        # But we'll keep the structure for compatibility
        
        model.train()
        # --- Reset epoch loss accumulators ---
        epoch_total_loss = 0.0
        epoch_loss_sisnr_speech = 0.0
        epoch_loss_sisnr_music = 0.0
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
            # Prepare input for ConvTasNet (expects B, T or T)
            if mixture_input.dim() == 3:  # (B, C, T)
                if mixture_input.shape[1] == 1:
                    mixture_for_model = mixture_input.squeeze(1)  # (B, T)
                else:
                    mixture_for_model = mixture_input.mean(dim=1)  # (B, T) - average channels
            else:
                mixture_for_model = mixture_input  # Already (B, T)
                
            # Model forward pass
            s_estimates = model(mixture_for_model) # Returns list [speech_est, music_est] each (B, T)
            
            # Calculate batch losses using loss function instance
            batch_total_loss, batch_avg_loss_sisnr_speech, batch_avg_loss_sisnr_music = loss_fn(
                s_estimates, targets, None, None
            )

            # Update torchmetrics SI-SNR (handles multiple sources internally with PIT)
            # Convert to the format expected by torchmetrics SI-SNR [B, S, T]
            s_est_stacked = torch.stack([
                s_estimates[0],  # speech estimates
                s_estimates[1]   # music estimates
            ], dim=1).to(device)
            
            # Convert targets to the right format for SI-SNR metric
            if targets.dim() == 4:  # (B, S, C, T)
                if targets.shape[2] == 1:
                    targets_for_metric = targets.squeeze(2).to(device)  # (B, S, T)
                else:
                    targets_for_metric = targets.mean(dim=2).to(device)  # (B, S, T)
            else:
                targets_for_metric = targets.to(device)
                
            train_sisnr_metric.update(s_est_stacked, targets_for_metric)

            batch_total_loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            # Accumulate per-sample losses/metrics
            epoch_total_loss += batch_total_loss.item() * batch_size
            epoch_loss_sisnr_speech += batch_avg_loss_sisnr_speech.item() * batch_size # This is neg SI-SNR loss
            epoch_loss_sisnr_music += batch_avg_loss_sisnr_music.item() * batch_size # This is neg SI-SNR loss
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
        train_losses_total.append(avg_epoch_total_loss)
        train_sisnr_s_loss.append(avg_epoch_loss_sisnr_speech) # History list name kept for consistency with checkpoint loading
        train_sisnr_m_loss.append(avg_epoch_loss_sisnr_music) # History list name kept for consistency with checkpoint loading
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
        val_sisnr_m_loss.append(val_results['Loss_Music(SI-SNR)'])
        val_speech_si_snr.append(val_results['Speech_SI-SNR_Eval'])
        val_music_si_snr.append(val_results['Music_SI-SNR_Eval'])
        val_overall_si_snr.append(val_results['SI-SNR']) 
        if pesq_key and pesq_key in val_results: val_pesq_hist.append(val_results[pesq_key])
        if "STOI" in val_results: val_stoi_hist.append(val_results['STOI'])
        val_sisnr_m_loss.append(val_results['Loss_Music(SI-SNR)'])

        epoch_time = time.time() - start_time

        # --- Logging ---
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch+1}/{args.epochs} SUMMARY | Time: {epoch_time:.2f}s")
        print(f"{'-' * 60}")

        # Train metrics
        print(f"TRAIN | Loss: {avg_epoch_total_loss:.4f}")
        print(f"       Speech SI-SNR: {-avg_epoch_loss_sisnr_speech:.2f} dB | Music SI-SNR: {-avg_epoch_loss_sisnr_music:.2f} dB")
        print(f"       Overall SI-SNR: {overall_sisnr:.2f} dB")

        # Validation metrics
        print(f"{'-' * 60}")
        print(f"VAL   | Loss: {avg_val_total_loss:.4f}" + (" (NEW BEST) ✓" if avg_val_total_loss < best_val_loss else ""))
        print(f"       Speech SI-SNR: {val_results['Speech_SI-SNR_Eval']:.2f} dB | Music SI-SNR: {val_results['Music_SI-SNR_Eval']:.2f} dB")
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
        
        # Log training metrics
        writer.add_scalar('Loss/train_total', avg_epoch_total_loss, epoch+1)
        writer.add_scalar('Loss/train_speech_sisnr', avg_epoch_loss_sisnr_speech, epoch+1)
        writer.add_scalar('Loss/train_music_sisnr', avg_epoch_loss_sisnr_music, epoch+1)

        # Log performance metrics (positive values for SI-SNR)
        writer.add_scalar('Metrics/train_speech_sisnr', -avg_epoch_loss_sisnr_speech, epoch+1)
        writer.add_scalar('Metrics/train_music_sisnr', -avg_epoch_loss_sisnr_music, epoch+1)
        writer.add_scalar('Metrics/train_overall_sisnr', overall_sisnr, epoch+1)

        # Log validation metrics
        writer.add_scalar('Loss/val_total', avg_val_total_loss, epoch+1)
        writer.add_scalar('Loss/val_speech_sisnr', val_results['Loss_Speech(SI-SNR)'], epoch+1)
        writer.add_scalar('Loss/val_music_sisnr', val_results['Loss_Music(SI-SNR)'], epoch+1)

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
            'music_sisnr': train_sisnr_m_loss
        }, {
            'speech_sisnr': val_sisnr_s_loss, 
            'music_sisnr': val_sisnr_m_loss
        }, True) # Always include music loss for ConvTasNet
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
                # ConvTasNet expects (B, T) or (T), dataset gives (C, T) -> format correctly
                if mixture_tensor.ndim == 2:  # (C, T)
                    if mixture_tensor.shape[0] == 1:
                        mixture_tensor = mixture_tensor.squeeze(0)  # (T)
                    else:
                        mixture_tensor = mixture_tensor.mean(0)  # Average channels -> (T)
                
                # Add batch dimension if needed
                if mixture_tensor.ndim == 1:
                    mixture_tensor = mixture_tensor.unsqueeze(0)  # (1, T)
                
                print(f"Saving sample audio and spectrograms for epoch {epoch+1}, input shape: {mixture_tensor.shape}")

                # Get separated sources from ConvTasNet
                s_est_sample = model(mixture_tensor)  # List [speech_est, music_est] each (B, T)
                
                # Create reconstructed mixture by adding the separated sources
                x_mix_recon_sample = s_est_sample[0] + s_est_sample[1]  # (B, T)

                # Save audio files - ConvTasNet outputs are already in the right format
                save_audio(s_est_sample[0], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_speech_est.wav'), args.sr)
                save_audio(s_est_sample[1], os.path.join(samples_dir, f'epoch_{epoch+1:03d}_music_est.wav'), args.sr)
                save_audio(x_mix_recon_sample, os.path.join(samples_dir, f'epoch_{epoch+1:03d}_mixture_recon.wav'), args.sr)

                # Save spectrograms
                save_spectrogram(
                    s_est_sample[0], 
                    os.path.join(spectrograms_dir, f'epoch_{epoch+1:03d}_speech_est_spec.png'),
                    args.sr, args.n_fft, args.hop_length,
                    title=f"Estimated Speech (Epoch {epoch+1})"
                )
                save_spectrogram(
                    s_est_sample[1], 
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
                    speech_est_cpu = s_est_sample[0].cpu()
                    music_est_cpu = s_est_sample[1].cpu()
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
            },
            {
                "speech_sisnr": val_sisnr_s_loss,
                "music_sisnr": val_sisnr_m_loss if val_sisnr_m_loss else [],
            },
            True,  # Always include music loss for ConvTasNet
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

        test_model = ConvTasNet(
            N=final_model_args["N"],
            L=final_model_args["L"],
            B=final_model_args["B"],
            H=final_model_args["H"],
            P=final_model_args["P"],
            X=final_model_args["X"],
            R=final_model_args["R"],
            norm=final_model_args["norm"],
            num_spks=2,  # Fixed at 2 for speech and music
            activate=final_model_args["activate"],
            causal=final_model_args["causal"]
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
            test_loss_fn = ConvTasNetLoss(
                speech_target_index=0,  # Speech is first source
                music_target_index=1,   # Music is second source
                sample_rate=args.sr     # Keep sample rate for compatibility
            ).to(device)

            # Run evaluation on test set using the test dataloader
            test_results = evaluate(test_model, test_loader, test_loss_fn, test_metrics, device, 
                                   desc="Testing (best model)", max_batches=0) # Pass test_loader

            print("\n" + "=" * 70)
            print(" FINAL TEST RESULTS ".center(70, "="))
            print("=" * 70)
            print(f"\nModel: ConvTasNet @ Epoch {checkpoint['epoch']}")
            print(f"\nLOSS METRICS:")
            print(f"  Total Loss:       {test_results['Loss_Total']:.4f}")
            print(f"  Speech SI-SNR:    {test_results['Loss_Speech(SI-SNR)']:.4f}")
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
            writer.add_scalar('Test/Loss_Music_SISNR', test_results['Loss_Music(SI-SNR)'], 0)

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

                    # Prepare input for ConvTasNet (expects B, T or T)
                    if mixture_input.dim() == 3:  # (B, C, T)
                        if mixture_input.shape[1] == 1:
                            mixture_for_model = mixture_input.squeeze(1)  # (B, T)
                        else:
                            mixture_for_model = mixture_input.mean(dim=1)  # (B, T) - average channels
                    else:
                        mixture_for_model = mixture_input  # Already (B, T)

                    # Process through model - returns list of [speech, music]
                    s_estimates = test_model(mixture_for_model)
                    
                    # Create reconstructed mixture by adding the sources
                    x_mix_recon = s_estimates[0] + s_estimates[1]

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
                    save_audio(s_estimates[0], os.path.join(sample_dir, 'est_speech.wav'), args.sr)
                    save_audio(s_estimates[1], os.path.join(sample_dir, 'est_music.wav'), args.sr)
                    save_audio(x_mix_recon, os.path.join(sample_dir, 'est_mixture.wav'), args.sr)

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
                    save_spectrogram(s_estimates[0], 
                                    os.path.join(sample_spec_dir, 'est_speech_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Speech (Sample {i+1})")
                    save_spectrogram(s_estimates[1], 
                                    os.path.join(sample_spec_dir, 'est_music_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Music (Sample {i+1})")
                    save_spectrogram(x_mix_recon, 
                                    os.path.join(sample_spec_dir, 'est_mixture_spec.png'), 
                                    args.sr, args.n_fft, args.hop_length,
                                    title=f"Estimated Mixture (Sample {i+1})")

                    # Calculate SI-SNR for this specific sample
                    speech_est = s_estimates[0].squeeze()
                    music_est = s_estimates[1].squeeze()
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
    parser.add_argument('--segment', type=float, default=1.0, help='Duration of audio segments in seconds')
    parser.add_argument('--sr', type=int, default=8000, help='Sample rate (PESQ requires 8k or 16k)')

    # --- Training Arguments ---
    parser.add_argument('--save_dir', type=str, default='checkpoints_ConvTasNet', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size') # Keep small for memory
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm value (0 to disable)')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training loss every N batches')
    parser.add_argument('--max_batches', type=int, default=0, help='Maximum number of batches per epoch (0 for all)')

    # --- Model Hyperparameters ---
    # Match the ConvTasNet.__init__ arguments
    parser.add_argument('--N', type=int, default=512, help='Number of filters in autoencoder')
    parser.add_argument('--L', type=int, default=16, help='Length of the filters in samples')
    parser.add_argument('--B', type=int, default=128, help='Number of channels in bottleneck and residual paths')
    parser.add_argument('--H', type=int, default=512, help='Number of channels in convolutional blocks')
    parser.add_argument('--P', type=int, default=3, help='Kernel size in convolutional blocks')
    parser.add_argument('--X', type=int, default=8, help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', type=int, default=3, help='Number of repeats')
    parser.add_argument('--norm', type=str, default='gln', choices=['gln', 'cln', 'bn'], help='Normalization type')
    parser.add_argument('--activate', type=str, default='relu', choices=['relu', 'sigmoid', 'softmax'], help='Activation function')
    parser.add_argument('--causal', action='store_true', help='Use causal model')
    parser.add_argument('--n_fft', type=int, default=512, help='FFT size for spectrograms')
    parser.add_argument('--hop_length', type=int, default=128, help='Hop length for spectrograms')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel bands for mel spectrograms')

    # --- Eval/Debug Arguments ---
    parser.add_argument('--sample_index', type=int, default=-1, help='Index of validation sample to save audio for (-1 for random)')
    parser.add_argument('--test_samples', type=int, default=5, help='Number of test samples to save during final evaluation (0 to disable)')
    parser.add_argument('--test_samples_random', action='store_true', help='Use random samples for test audio instead of sequential')

    args = parser.parse_args()

    # --- Basic Validation ---
    if args.sr not in [8000, 16000]:
        raise ValueError("PESQ metric requires sr=8000 or sr=16000")
    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    train(args)
