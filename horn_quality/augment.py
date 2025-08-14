from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import random
import torch
import torchaudio.functional as AF


@dataclass
class WaveformAugmenter:
	sample_rate: int
	max_time_shift_ms: int = 100  # +/- 100 ms
	add_noise_prob: float = 0.5
	noise_snr_db_range: tuple[float, float] = (10.0, 25.0)
	random_gain_db_range: tuple[float, float] = (-3.0, 3.0)
	bandpass_prob: float = 0.3
	band_min_hz: int = 200
	band_max_hz: int = 6000

	def apply(self, waveform: torch.Tensor) -> torch.Tensor:
		x = waveform
		if self.max_time_shift_ms > 0:
			x = self.random_time_shift(x)
		if random.random() < self.bandpass_prob:
			x = self.random_bandpass(x)
		x = self.random_gain(x)
		if random.random() < self.add_noise_prob:
			x = self.add_gaussian_noise_for_snr(x)
		return x

	def random_time_shift(self, x: torch.Tensor) -> torch.Tensor:
		max_shift = int(self.sample_rate * self.max_time_shift_ms / 1000)
		if max_shift <= 0:
			return x
		shift = random.randint(-max_shift, max_shift)
		if shift == 0:
			return x
		return torch.roll(x, shifts=shift, dims=-1)

	def random_gain(self, x: torch.Tensor) -> torch.Tensor:
		min_db, max_db = self.random_gain_db_range
		gain_db = random.uniform(min_db, max_db)
		gain = math.pow(10.0, gain_db / 20.0)
		return x * gain

	def add_gaussian_noise_for_snr(self, x: torch.Tensor) -> torch.Tensor:
		min_db, max_db = self.noise_snr_db_range
		snr_db = random.uniform(min_db, max_db)
		signal_power = x.pow(2).mean().clamp(min=1e-8)
		snr = math.pow(10.0, snr_db / 10.0)
		noise_power = signal_power / snr
		noise = torch.randn_like(x) * torch.sqrt(noise_power)
		return x + noise

	def random_bandpass(self, x: torch.Tensor) -> torch.Tensor:
		f1 = random.uniform(self.band_min_hz, self.band_max_hz * 0.7)
		f2 = random.uniform(max(f1 + 100, self.band_min_hz + 50), self.band_max_hz)
		y = AF.highpass_biquad(x, self.sample_rate, cutoff_freq=f1)
		y = AF.lowpass_biquad(y, self.sample_rate, cutoff_freq=f2)
		return y