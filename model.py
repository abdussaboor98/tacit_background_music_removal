# model.py (Modified for WFAE-Inspired Approach)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm # Using standard LayerNorm

# --- Helper Modules ---
# Assuming DepthwiseSeparableConv, TCNBlock, FeedForwardModule,
# ConformerConvModule, ConformerBlock, HybridBlock are defined exactly
# as in your uploaded model.py file (with corrected LayerNorm usage)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block inspired by Conv-TasNet"""
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        # Block 1: Pointwise Conv -> PReLU -> Norm
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        # Apply LayerNorm across the hidden_channels dimension
        self.norm1 = LayerNorm(hidden_channels, eps=1e-8) # Expects (B, T, F)

        # Block 2: Depthwise Separable Conv -> PReLU -> Norm
        self.ds_conv = DepthwiseSeparableConv(
            hidden_channels, in_channels, kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        self.prelu2 = nn.PReLU()
        # Apply LayerNorm across the in_channels dimension
        self.norm2 = LayerNorm(in_channels, eps=1e-8) # Expects (B, T, F)

    def forward(self, x):
        # Input x shape: (Batch, Features=in_channels, Time)
        residual = x

        # Block 1
        x = self.conv1(x)
        x = self.prelu1(x)
        # Transpose -> LayerNorm -> Transpose Back
        x = x.transpose(1, 2) # (B, T, F=hidden_channels)
        x = self.norm1(x)
        x = x.transpose(1, 2) # (B, F=hidden_channels, T)

        # Block 2
        x = self.ds_conv(x)
        x = self.prelu2(x)
        # Transpose -> LayerNorm -> Transpose Back
        x = x.transpose(1, 2) # (B, T, F=in_channels)
        x = self.norm2(x)
        x = x.transpose(1, 2) # (B, F=in_channels, T)

        # Add residual
        return x + residual

class FeedForwardModule(nn.Module):
    """Simple MLP FeedForward module for Conformer"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(), # Swish activation often used in Conformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # Input shape: (Batch, Time/Seq, Dim)
        return self.net(x)

class ConformerConvModule(nn.Module):
    """Convolution Module for Conformer block"""
    def __init__(self, dim, kernel_size=31, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(dim, dim * 2, kernel_size=1) # Expand dim
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                        padding=padding, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim) # BatchNorm often used here
        self.activation = nn.SiLU() # Swish
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (Batch, T, D) from conformer block norm
        x_res = x
        x = x.transpose(1, 2) # (B, D, T) for convs
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1) # Gated Linear Unit using pointwise convs -> (B, D, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        # Output shape: (B, T, D) with residual
        return x.transpose(1, 2) + x_res


class ConformerBlock(nn.Module):
    """Conformer Block inspired by SE-Conformer/TD-Conformer"""
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=4,
                 conv_kernel_size=31, dropout=0.1):
        super().__init__()
        hidden_dim = dim * ffn_expansion_factor
        self.ffn1 = FeedForwardModule(dim, hidden_dim, dropout)
        self.norm_ffn1_out = LayerNorm(dim, eps=1e-8) # Norm after FFN1 before residual
        self.dropout_ffn1 = nn.Dropout(dropout)

        self.norm_mhsa = LayerNorm(dim, eps=1e-8)
        # NOTE: Requires positional encoding added to input 'x' before this layer
        self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout)
        self.norm_mhsa_out = LayerNorm(dim, eps=1e-8) # Norm after MHSA before residual

        self.norm_conv = LayerNorm(dim, eps=1e-8)
        self.conv_module = ConformerConvModule(dim, conv_kernel_size, dropout)
        self.norm_conv_out = LayerNorm(dim, eps=1e-8) # Norm after Conv before residual

        self.norm_ffn2 = LayerNorm(dim, eps=1e-8)
        self.ffn2 = FeedForwardModule(dim, hidden_dim, dropout)
        self.dropout_ffn2 = nn.Dropout(dropout)
        self.norm_final = LayerNorm(dim, eps=1e-8) # Final norm for the block output

    def forward(self, x):
        # Input shape: (Batch, Time/Seq, Dim)
        # Macaron-style: FFN -> MHSA -> Conv -> FFN
        # Apply FFN1
        residual = x
        x = self.ffn1(x)
        x = self.norm_ffn1_out(residual + self.dropout_ffn1(x) * 0.5)

        # Apply MHSA
        residual = x
        x = self.norm_mhsa(x)
        attn_output, _ = self.mhsa(x, x, x)
        x = self.norm_mhsa_out(residual + self.dropout_mhsa(attn_output))

        # Apply Conv Module
        residual = x
        x = self.norm_conv(x)
        x = self.conv_module(x) # Conv module has internal residual
        x = self.norm_conv_out(x) 

        # Apply FFN2
        residual = x
        x = self.norm_ffn2(x)
        x = self.ffn2(x)
        x = self.norm_final(residual + self.dropout_ffn2(x) * 0.5) # Final norm replaces norm_ffn2_out

        return x


class HybridBlock(nn.Module):
    """Hybrid Block alternating TCN and Conformer"""
    def __init__(self, features, tcn_hidden_channels, tcn_kernel_size, tcn_dilation_base,
                 tcn_layers_per_block, conformer_dim, conformer_heads, conformer_kernel_size,
                 conformer_ffn_expansion, conformer_dropout):
        super().__init__()
        self.norm_tcn_in = LayerNorm(features, eps=1e-8)
        self.tcn_stack = nn.ModuleList([
            TCNBlock(features, tcn_hidden_channels, tcn_kernel_size, tcn_dilation_base**i)
            for i in range(tcn_layers_per_block)
        ])

        self.norm_conf_in = LayerNorm(features, eps=1e-8)
        if conformer_dim != features:
            raise ValueError(f"Conformer dim ({conformer_dim}) must equal TCN features ({features})")
        self.conformer = ConformerBlock(
            dim=conformer_dim,
            num_heads=conformer_heads,
            ffn_expansion_factor=conformer_ffn_expansion,
            conv_kernel_size=conformer_kernel_size,
            dropout=conformer_dropout
        )
        
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # fusion parameter for TCN  
        self.logit_beta = nn.Parameter(torch.tensor(0.0))   # fusion parameter for Conformer

    def forward(self, tcn_in, conf_in):
        """
        Process TCN input and pass output to Conformer along with skip from previous Conformer.
        
        Args:
            tcn_in: Input for TCN stream (Batch, Features, Time)
            conf_in: Input for Conformer stream (Batch, Features, Time)
            
        Returns:
            Tuple of (tcn_out, conf_out):
                tcn_out: Output from TCN stream (Batch, Features, Time)
                conf_out: Output from Conformer stream (Batch, Features, Time)
        """
        
        alpha = torch.sigmoid(self.logit_alpha) # scale factor for TCN
        beta = torch.sigmoid(self.logit_beta)   # scale factor for Conformer

        # --- TCN Stream ---
        # Add residual connection within the TCN stream
        tcn_residual = tcn_in
        
        # Combine TCN input with Conformer output
        combined_tcn_in = tcn_in + alpha * conf_in
        # Normalize input
        tcn_norm = self.norm_tcn_in(combined_tcn_in.transpose(1, 2)).transpose(1, 2) # (B, F, T)
        tcn_out = tcn_norm
        # Process through TCN stack
        for layer in self.tcn_stack:
            tcn_out = layer(tcn_out) # TCNBlock has internal residual
        

        # --- Conformer Stream ---
        # Add residual connection from the previous Conformer output
        conf_residual = conf_in
        
        # We're combining the TCN output with the previous Conformer output
        combined_conf_in = conf_in + beta * tcn_out
        
        # Conformer expects (Batch, Time, Features)
        conf_time_first = combined_conf_in.transpose(1, 2) # (B, T, F)
        # Normalize input
        conf_norm = self.norm_conf_in(conf_time_first)
        # Process through Conformer
        conf_time_out = self.conformer(conf_norm) # Conformer block has internal residual
        # Convert back to (Batch, Features, Time)
        conf_out = conf_time_out.transpose(1, 2)
        
        # Apply residual connection to TCN stream
        tcn_out = tcn_out + tcn_residual
        # Apply residual connection to Conformer stream
        conf_out = conf_out + conf_residual

        # Return both streams separately without mixing
        return tcn_out, conf_out


# --- Main Network (MODIFIED for WFAE-Inspired Approach) ---

class MSHybridNet(nn.Module):
    """
    Music-Speech Hybrid Network with specialized dual stream architecture.
    
    The model uses a dual-stream approach:
    - TCN stream primarily for music separation
    - Conformer stream primarily for speech separation
    
    Each stream has its own mask estimation head but they feed information
    to each other during processing through the HybridBlocks.
    """
    def __init__(self,
                 channels=1, # Added channels argument
                 # Encoder/Decoder Params
                 enc_kernel_size=16,
                 enc_stride=8,
                 enc_features=512,
                 # Separator Params
                 num_blocks=6,
                 # TCN Params within Hybrid Block
                 tcn_hidden_channels=1024,
                 tcn_kernel_size=3,
                 tcn_layers_per_block=8,
                 tcn_dilation_base=2,
                 # Conformer Params within Hybrid Block
                 conformer_dim=512, # Should match enc_features
                 conformer_heads=8,
                 conformer_kernel_size=31,
                 conformer_ffn_expansion=4,
                 conformer_dropout=0.1
                 ):
        super().__init__()
        self.enc_features = enc_features
        self.channels = channels # Store channels
        
        # Define source indices
        self.speech_idx = 0  # Index for speech source in output
        self.music_idx = 1   # Index for music source in output

        # Encoder
        self.encoder = nn.Conv1d(self.channels, enc_features, kernel_size=enc_kernel_size, stride=enc_stride, bias=False)
        self.encoder_activation = nn.ReLU()

        # Separator - Stacked Hybrid Blocks
        self.separator = nn.ModuleList([
            HybridBlock(
                features=enc_features,
                tcn_hidden_channels=tcn_hidden_channels,
                tcn_kernel_size=tcn_kernel_size,
                tcn_layers_per_block=tcn_layers_per_block,
                tcn_dilation_base=tcn_dilation_base,
                conformer_dim=conformer_dim,
                conformer_heads=conformer_heads,
                conformer_kernel_size=conformer_kernel_size,
                conformer_ffn_expansion=conformer_ffn_expansion,
                conformer_dropout=conformer_dropout
            ) for _ in range(num_blocks)
        ])

        # Separate Mask Heads
        # TCN stream mask head (for music)
        self.mask_head_tcn = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(enc_features, enc_features, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Conformer stream mask head (for speech)
        self.mask_head_conf = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(enc_features, enc_features, kernel_size=1),
            nn.Sigmoid()
        )

        # Source-specific decoders
        self.decoder_music = nn.ConvTranspose1d(enc_features, self.channels, 
                                              kernel_size=enc_kernel_size, 
                                              stride=enc_stride, bias=False)
        self.decoder_speech = nn.ConvTranspose1d(enc_features, self.channels, 
                                               kernel_size=enc_kernel_size, 
                                               stride=enc_stride, bias=False)
    def forward(self, mixture_waveform):
        # mixture_waveform: (Batch, Channels, Time) - Keep channel dim

        # --- Encoding ---
        encoded_features = self.encoder(mixture_waveform) # W
        encoded_features_act = self.encoder_activation(encoded_features)
        # Output W: (Batch, Features, Frames)
        
        # --- Dual-Stream Separation ---
        # Initialize both streams with the same encoded features
        tcn_stream = encoded_features_act
        conf_stream = encoded_features_act
        
        # Pass through separator blocks maintaining dual streams
        # Each block passes TCN output to Conformer along with previous Conformer output
        for block in self.separator:
            tcn_stream, conf_stream = block(tcn_stream, conf_stream)
        
        # --- Mask Estimation (Using Separate Mask Heads) ---
        # Music mask from TCN stream
        music_mask = self.mask_head_tcn(tcn_stream)  # (B, F, T)
        # Speech mask from Conformer stream
        speech_mask = self.mask_head_conf(conf_stream)  # (B, F, T)
        
        # --- Source Reconstruction ---
        # Apply masks to encoded features
        music_features = encoded_features_act * music_mask
        speech_features = encoded_features_act * speech_mask
        
        # Decode to waveforms using separate decoders
        music_waveform = self.decoder_music(music_features)  # (B, C, T)
        speech_waveform = self.decoder_speech(speech_features)  # (B, C, T)
        
        # Stack sources with speech first, music second
        s_estimates = torch.stack([speech_waveform, music_waveform], dim=1)  # (B, 2, C, T)
        
        # --- Mixture Reconstruction ---
        x_mix_recon = speech_waveform + music_waveform
        
        return s_estimates, x_mix_recon
    
    def get_music(self, mixture_waveform):
        """Get the separated music source from a mixture"""
        s_estimates, _ = self.forward(mixture_waveform)
        return s_estimates[:, self.music_idx]  # Return music source
    
    def get_speech(self, mixture_waveform):
        """Get the separated speech source from a mixture"""
        s_estimates, _ = self.forward(mixture_waveform)
        return s_estimates[:, self.speech_idx]  # Return speech source

# --- Example Usage (if needed) ---
# Keep __main__ block as provided in uploaded model.py file
if __name__ == '__main__':
    model = MSHybridNet(
        channels=1, # Specify mono channel
        enc_kernel_size=16, enc_stride=8, enc_features=128, num_blocks=4,
        tcn_hidden_channels=256, tcn_kernel_size=3, tcn_layers_per_block=8, tcn_dilation_base=2,
        conformer_dim=128, conformer_heads=4, conformer_kernel_size=31,
        conformer_ffn_expansion=4, conformer_dropout=0.1
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    dummy_input = torch.randn(2, 1, 24000) # (B, C=1, T)
    s_estimates_out, x_mix_recon_out = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Separated Sources Output shape: {s_estimates_out.shape}") # (B, 2, C, T) with music at idx 0, speech at idx 1
    print(f"Reconstructed Mixture Output shape: {x_mix_recon_out.shape}") # Should be (B, C, T)
    
    # Test convenience methods
    music_out = model.get_music(dummy_input)
    speech_out = model.get_speech(dummy_input)
    print(f"Music output shape: {music_out.shape}") # (B, C, T)
    print(f"Speech output shape: {speech_out.shape}") # (B, C, T)