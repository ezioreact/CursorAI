from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .augment import WaveformAugmenter


torchaudio.set_audio_backend("sox_io")


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def load_waveform(
    path: str,
    target_sample_rate: int,
) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    waveform = _to_mono(waveform)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def pad_or_trim(
    waveform: torch.Tensor,
    target_num_samples: int,
    random_pad: bool = True,
) -> torch.Tensor:
    current = waveform.size(-1)
    if current == target_num_samples:
        return waveform
    if current > target_num_samples:
        if random_pad:
            start = random.randint(0, current - target_num_samples)
        else:
            start = 0
        return waveform[..., start : start + target_num_samples]
    # pad
    pad_total = target_num_samples - current
    if random_pad:
        pad_left = random.randint(0, pad_total)
    else:
        pad_left = 0
    pad_right = pad_total - pad_left
    return torch.nn.functional.pad(waveform, (pad_left, pad_right))


@dataclass
class DataConfig:
    dataset_dir: str
    sample_rate: int = 16000
    duration_sec: float = 4.0
    batch_size: int = 32
    num_workers: int = 2
    augment: bool = True

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


class HornDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        sample_rate: int,
        num_samples: int,
        apply_augment: bool = False,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        pass_dir = os.path.join(root_dir, "pass")
        fail_dir = os.path.join(root_dir, "fail")

        pass_files = [os.path.join(pass_dir, f) for f in os.listdir(pass_dir) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        fail_files = [os.path.join(fail_dir, f) for f in os.listdir(fail_dir) if f.lower().endswith((".wav", ".flac", ".mp3"))]

        all_items = [(f, 1) for f in pass_files] + [(f, 0) for f in fail_files]
        all_items.sort()

        # Stratified split by filename hash for reproducibility
        def split_bucket(path: str) -> str:
            base = os.path.basename(path)
            h = abs(hash(base)) % 100
            # 70/15/15 split
            if h < 70:
                return "train"
            elif h < 85:
                return "val"
            return "test"

        items = [(p, y) for (p, y) in all_items if split_bucket(p) == split]
        self.items: List[Tuple[str, int]] = items
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.apply_augment = apply_augment
        self.augmenter = WaveformAugmenter(sample_rate) if apply_augment else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        waveform, _ = load_waveform(path, self.sample_rate)
        waveform = pad_or_trim(waveform, self.num_samples, random_pad=self.apply_augment)
        if self.augmenter is not None:
            waveform = self.augmenter.apply(waveform)
        # Normalize to -1..1 range with small clamp
        waveform = waveform.clamp(-1.0, 1.0)
        return waveform, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_ds = HornDataset(
        config.dataset_dir,
        split="train",
        sample_rate=config.sample_rate,
        num_samples=config.num_samples,
        apply_augment=config.augment,
    )
    val_ds = HornDataset(
        config.dataset_dir,
        split="val",
        sample_rate=config.sample_rate,
        num_samples=config.num_samples,
        apply_augment=False,
    )
    test_ds = HornDataset(
        config.dataset_dir,
        split="test",
        sample_rate=config.sample_rate,
        num_samples=config.num_samples,
        apply_augment=False,
    )

    # Compute class weights for imbalance handling
    labels = [y for _, y in train_ds.items]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32)

    # Weighted sampling to balance mini-batches
    sample_weights = [1.0 / (num_pos if y == 1 else num_neg) for y in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, pos_weight