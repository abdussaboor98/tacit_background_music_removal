import random
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class FolderTripletDataset(Dataset):
    """Each sample lives in its own directory with three WAV files:

        <root>/<split>/sample_0001/
            mixture.wav    – mixture (speech + music)
            speech.wav – target speech stem (speech + ambient noise)
            music.wav  – target music stem

    All clips are 3 seconds long in your dataset, so the default `segment`
    length is set to **3.0 s**. If you ever use longer clips you can override
    it with the `--segment` CLI flag.
    """

    def __init__(self, root: Path, split: str = "train", segment: float = 3.0, sr: int = 8000):
        split_dir = root / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        self.sample_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        self.segment = segment
        self.sr = sr
        self.split = split

    # ---------------------------------------------------------------------
    # Utility: load a random crop (or pad) to `segment` seconds
    # ---------------------------------------------------------------------
    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        # Convert to mono if stereo
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # Convert to mono by averaging channels
        else:
            wav = wav.squeeze(0)  # Remove channel dimension if already mono
            
        seg_len = int(self.segment * self.sr)
        if wav.shape[0] > seg_len:
            # For validation/test, use center crop instead of random crop
            if self.split != "train":
                start = (wav.shape[0] - seg_len) // 2
            else:
                start = random.randint(0, wav.shape[0] - seg_len)
            wav = wav[start : start + seg_len]
        else:
            wav = F.pad(wav, (0, seg_len - wav.shape[0]))
        return wav.unsqueeze(0)  # (1, T)

    def _load_wav(self, path: Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(path, channels_first=True, normalize=True)
            # Convert to mono if stereo
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            return wav
        except Exception as e:
            print(f"Error loading audio file {path}: {str(e)}")
            # Return a silent audio segment of the correct length as fallback
            return torch.zeros(1, int(self.segment * self.sr))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        try:
            d = self.sample_dirs[idx]
            mix_path = d / "mixture.wav"
            if mix_path.exists():
                mix = self._crop(self._load_wav(mix_path))
            else:
                mix = self._crop(self._load_wav(d / "mix.wav"))
            
            speech_path = d / "speech.wav"
            if speech_path.exists():
                speech = self._crop(self._load_wav(speech_path))
            else:
                speech = mix  # fallback: use mixture
                
            music_path = d / "music.wav"
            if music_path.exists():
                music = self._crop(self._load_wav(music_path))
            else:
                music = mix - speech  # fallback: use mixture - speech
            return mix, speech, music
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return zero tensors as fallback
            dummy = torch.zeros(1, int(self.segment * self.sr))
            return dummy, dummy, dummy 