import os
import argparse
import torch
import torchaudio
from horn_quality.model import build_horn_quality_model
from horn_quality.data import load_waveform, pad_or_trim


def load_model(checkpoint_path: str, base_channels: int = 64, dropout: float = 0.1, se_reduction: int = 8, device: str = None):
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	model = build_horn_quality_model(base_channels=base_channels, dropout=dropout, se_reduction=se_reduction)
	ckpt = torch.load(checkpoint_path, map_location=device)
	if "model_state" in ckpt:
		model.load_state_dict(ckpt["model_state"])
	else:
		model.load_state_dict(ckpt)
	model.eval().to(device)
	return model, device


def predict(audio_path: str, checkpoint_path: str, sample_rate: int = 16000, duration_sec: float = 4.0, threshold: float = 0.5):
	model, device = load_model(checkpoint_path)
	wave, _ = load_waveform(audio_path, sample_rate)
	wave = pad_or_trim(wave, int(sample_rate * duration_sec), random_pad=False)
	wave = wave.to(device)
	with torch.no_grad():
		logit = model(wave.unsqueeze(0))
		prob = torch.sigmoid(logit).item()
	label = "pass" if prob >= threshold else "fail"
	return {"prob_pass": prob, "pred": label}


def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--audio", required=True)
	parser.add_argument("--ckpt", required=True)
	parser.add_argument("--sample_rate", type=int, default=16000)
	parser.add_argument("--duration", type=float, default=4.0)
	parser.add_argument("--threshold", type=float, default=0.5)
	cli = parser.parse_args(args)
	res = predict(cli.audio, cli.ckpt, cli.sample_rate, cli.duration, cli.threshold)
	print(res)


if __name__ == "__main__":
	main()