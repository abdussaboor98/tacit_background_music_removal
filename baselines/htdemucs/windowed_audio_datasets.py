import math # For ceil/floor if precise frame calculation needs it, or direct int casting.
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class FolderTripletDataset(Dataset):
    """
    Dataset for loading windowed segments from audio files.
    Each sample directory is expected to contain three long WAV files:
        <root>/<split>/sample_0001/
            mixture.wav – mixture (speech + music)
            speech.wav  – target speech stem
            music.wav   – target music stem

    The dataset creates (potentially overlapping) windows from these files.
    Segments that would be shorter than `segment_length_sec` at the very 
    end of the files are ignored.
    """

    def __init__(self, root: Path, split: str = "train", 
                 segment_length_sec: float = 3.0, 
                 hop_length_sec: float = 1.5, # Default hop, e.g., 50% overlap if segment is 3s
                 sr: int = 8000):
        
        if not isinstance(root, Path):
            root = Path(root) # Ensure root is a Path object

        split_dir = root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist. Looked for {split_dir.resolve()}")
        
        if segment_length_sec <= 0:
            raise ValueError("segment_length_sec must be positive.")
        if hop_length_sec <= 0:
            raise ValueError("hop_length_sec must be positive.")

        self.segment_frames = int(segment_length_sec * sr)
        self.hop_frames = int(hop_length_sec * sr)
        self.sr = sr

        self.windows = []
        potential_sample_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

        if not potential_sample_dirs:
            print(f"Warning: No subdirectories found in {split_dir}.")


        for sample_dir in potential_sample_dirs:
            mix_path = sample_dir / "mixture.wav"
            if not mix_path.exists():
                mix_path = sample_dir / "mix.wav" # Alternative name
            
            speech_path = sample_dir / "speech.wav"
            music_path = sample_dir / "music.wav"

            # Check existence of all three required files
            required_files_exist = True
            if not mix_path.exists():
                # print(f"Warning: Mixture file (mixture.wav or mix.wav) not found in {sample_dir}. Skipping directory.")
                required_files_exist = False
            if not speech_path.exists():
                # print(f"Warning: Speech file (speech.wav) not found in {sample_dir}. Skipping directory.")
                required_files_exist = False
            if not music_path.exists():
                # print(f"Warning: Music file (music.wav) not found in {sample_dir}. Skipping directory.")
                required_files_exist = False
            
            if not required_files_exist:
                continue

            try:
                paths_map = {"mix": mix_path, "speech": speech_path, "music": music_path}
                infos = {}
                effective_frames_at_target_sr = {}

                for name, path_obj in paths_map.items():
                    # torchaudio.info can take Path object directly
                    info = torchaudio.info(path_obj) 
                    infos[name] = info
                    # Calculate effective number of frames if resampled to target sr
                    effective_frames_at_target_sr[name] = math.floor(
                        info.num_frames * (self.sr / info.sample_rate)
                    )
                
                min_effective_frames = min(effective_frames_at_target_sr.values())

                if min_effective_frames < self.segment_frames:
                    # print(f"Info: Skipping {sample_dir} as its shortest track's effective length ({min_effective_frames} frames at {self.sr}Hz) is less than segment length ({self.segment_frames} frames).")
                    continue
                
                num_possible_windows = (min_effective_frames - self.segment_frames) // self.hop_frames + 1
                
                for i in range(num_possible_windows):
                    start_frame_at_target_sr = i * self.hop_frames
                    self.windows.append({
                        "sample_dir_path": sample_dir,
                        "start_frame_target_sr": start_frame_at_target_sr,
                        "original_infos": infos 
                    })
            except Exception as e:
                print(f"Error processing directory {sample_dir} during init: {str(e)}. Skipping.")
                continue
        
        if not self.windows:
            print(f"Warning: No valid windows found for split '{split}' in '{root}' with segment {segment_length_sec}s, hop {hop_length_sec}s at {self.sr}Hz. Check audio file lengths and dataset structure.")


    def _load_audio_segment(self, file_path: Path, original_info: torchaudio.backend.common.AudioMetaData, 
                            start_frame_target_sr: int, num_frames_target_sr: int) -> torch.Tensor:
        """
        Loads a specific segment from an audio file, handling resampling and ensuring exact length.
        """
        try:
            original_sr = original_info.sample_rate
            
            load_offset_orig_sr = math.floor(start_frame_target_sr * (original_sr / self.sr))
            load_num_frames_orig_sr = math.ceil(num_frames_target_sr * (original_sr / self.sr))

            # Ensure load_offset is not negative (shouldn't happen with valid inputs)
            load_offset_orig_sr = max(0, load_offset_orig_sr)

            # Ensure we don't try to load beyond the file's actual number of frames from the offset
            if load_offset_orig_sr >= original_info.num_frames:
                 # Trying to start reading past the end of the file.
                # print(f"Warning: Load offset ({load_offset_orig_sr}) is beyond file frames ({original_info.num_frames}) for {file_path}. Returning silent segment.")
                return torch.zeros(1, num_frames_target_sr)

            # Adjust num_frames if it exceeds available frames from offset
            available_frames_from_offset = original_info.num_frames - load_offset_orig_sr
            load_num_frames_orig_sr = min(load_num_frames_orig_sr, available_frames_from_offset)
            
            if load_num_frames_orig_sr <= 0:
                # This implies not enough frames at original SR to form the segment,
                # or we are at the very end with insufficient remainder.
                # print(f"Warning: Calculated load_num_frames_orig_sr <= 0 for {file_path} (offset {load_offset_orig_sr}, original frames {original_info.num_frames}). Returning silent segment.")
                return torch.zeros(1, num_frames_target_sr)

            wav, sr_loaded = torchaudio.load(
                file_path, # Path object is fine
                frame_offset=load_offset_orig_sr, 
                num_frames=load_num_frames_orig_sr,
                channels_first=True, 
                normalize=True
            )

            if sr_loaded != self.sr:
                wav = torchaudio.functional.resample(wav, sr_loaded, self.sr)

            if wav.size(0) > 1: # Convert to mono
                wav = wav.mean(0, keepdim=True)
            elif wav.ndim == 1: # Ensure (1, T) shape
                wav = wav.unsqueeze(0)
            
            current_len_at_target_sr = wav.shape[1]
            if current_len_at_target_sr > num_frames_target_sr:
                wav = wav[:, :num_frames_target_sr] # Trim
            elif current_len_at_target_sr < num_frames_target_sr:
                padding_needed = num_frames_target_sr - current_len_at_target_sr
                wav = F.pad(wav, (0, padding_needed)) # Pad
            
            return wav

        except Exception as e:
            print(f"Error loading audio segment from {file_path} (target offset {start_frame_target_sr}): {str(e)}. Returning silent segment.")
            return torch.zeros(1, num_frames_target_sr)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.windows)):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.windows)} samples.")

        window_data = self.windows[idx]
        sample_dir = window_data["sample_dir_path"]
        start_frame = window_data["start_frame_target_sr"]
        original_infos = window_data["original_infos"]

        mix_fname = "mixture.wav" if (sample_dir / "mixture.wav").exists() else "mix.wav"
        
        paths = {
            "mix": sample_dir / mix_fname,
            "speech": sample_dir / "speech.wav",
            "music": sample_dir / "music.wav"
        }

        segments = {}
        for name in ["mix", "speech", "music"]:
            # These files should exist due to checks in __init__
            # If one was deleted post-init, _load_audio_segment will handle it.
            segments[name] = self._load_audio_segment(
                paths[name], 
                original_infos[name], 
                start_frame, 
                self.segment_frames
            )
        
        mix_s = segments["mix"]
        speech_s = segments["speech"]
        music_s = segments["music"]

        # Fallback for speech: if speech_s is all zeros (error in loading or truly silent), use mix_s
        if torch.all(speech_s == 0) and not torch.all(mix_s == 0): # Avoid fallback if mix is also zero
            # print(f"Info: Speech segment for {paths['speech']} (idx {idx}) was silent/failed, using mixture.")
            speech_s = mix_s.clone()

        # Fallback for music: if music_s is all zeros, use (mix_s - speech_s)
        if torch.all(music_s == 0) and not (torch.all(mix_s == 0) and torch.all(speech_s == 0)): # Avoid if both mix/speech are zero
            # print(f"Info: Music segment for {paths['music']} (idx {idx}) was silent/failed, using (mixture - speech).")
            music_s = mix_s - speech_s
            music_s = torch.clamp(music_s, min=-1.0, max=1.0) # Clamp assuming normalized audio

        return mix_s, speech_s, music_s

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Create dummy dataset structure
    # root_dir = Path("./dummy_dataset")
    # (root_dir / "train" / "sample1").mkdir(parents=True, exist_ok=True)
    # (root_dir / "val" / "sample1").mkdir(parents=True, exist_ok=True)

    # # Create dummy audio files (e.g., 10 seconds long at 16000 Hz)
    # sr_orig = 16000
    # duration_sec = 10
    # dummy_audio_orig_sr = torch.randn(1, sr_orig * duration_sec)
    
    # # Resample to 8000 Hz for one file to test resampling
    # dummy_audio_8k_sr = torchaudio.functional.resample(dummy_audio_orig_sr, sr_orig, 8000)


    # torchaudio.save(root_dir / "train" / "sample1" / "mixture.wav", dummy_audio_orig_sr, sr_orig)
    # torchaudio.save(root_dir / "train" / "sample1" / "speech.wav", dummy_audio_8k_sr, 8000) # Save this one at 8k
    # torchaudio.save(root_dir / "train" / "sample1" / "music.wav", dummy_audio_orig_sr, sr_orig)

    # # Test dataset
    # print("Testing training dataset:")
    # train_dataset = FolderTripletDataset(root=root_dir, split="train", segment_length_sec=2.0, hop_length_sec=1.0, sr=8000)
    # print(f"Number of training windows: {len(train_dataset)}")
    # if len(train_dataset) > 0:
    #     mix, speech, music = train_dataset[0]
    #     print(f"Mix shape: {mix.shape}, Speech shape: {speech.shape}, Music shape: {music.shape}")
    #     print(f"Mix SR should be 8000, segment length {2.0*8000} = {mix.shape[1]}")

    #     # Check a few more samples
    #     if len(train_dataset) > 1:
    #          mix2, _, _ = train_dataset[1]
    #          print(f"Mix2 shape: {mix2.shape}")


    # # Clean up dummy files
    # # import shutil
    # # shutil.rmtree(root_dir)
    pass # Keep the pass if __main__ is not intended for submission 