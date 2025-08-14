import os
import argparse
from dataclasses import dataclass
from typing import Optional
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from horn_quality.data import DataConfig, create_dataloaders
from horn_quality.model import build_horn_quality_model, build_horn_quality_model_lite
 from horn_quality.utils import set_global_seed, evaluate_classification, save_checkpoint


@dataclass
 class TrainConfig:
 	dataset_dir: str
 	output_dir: str = "./outputs"
 	seed: int = 42
 	sample_rate: int = 16000
 	duration_sec: float = 4.0
 	batch_size: int = 32
 	lr: float = 3e-4
 	weight_decay: float = 1e-4
 	epochs: int = 50
 	patience: int = 8
 	base_channels: int = 64
 	dropout: float = 0.1
 	se_reduction: int = 8
 	augment: bool = True
 	model_variant: str = "lite"  # "lite" or "advanced"
 	device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, criterion, optimizer, device):
	model.train()
	total_loss = 0.0
	for wave, labels in tqdm(loader, desc="train", leave=False):
		wave = wave.to(device)
		labels = labels.to(device)
		optimizer.zero_grad(set_to_none=True)
		logits = model(wave)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * wave.size(0)
	return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
	model.eval()
	all_scores = []
	all_labels = []
	for wave, labels in tqdm(loader, desc="eval", leave=False):
		wave = wave.to(device)
		logits = model(wave)
		probs = torch.sigmoid(logits).detach().cpu()
		all_scores.extend(probs.tolist())
		all_labels.extend(labels.tolist())
	metrics = evaluate_classification(all_labels, all_scores)
	return metrics


def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_dir", type=str, required=True)
	parser.add_argument("--output_dir", type=str, default="./outputs")
		parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
 	parser.add_argument("--model_variant", type=str, default="lite", choices=["lite", "advanced"]) 
 	cli = parser.parse_args(args)

	# Load default config and override from YAML if provided
	cfg = TrainConfig(dataset_dir=cli.dataset_dir, output_dir=cli.output_dir)
	if cli.config is not None and os.path.exists(cli.config):
		with open(cli.config, "r") as f:
			user_cfg = yaml.safe_load(f)
			for k, v in user_cfg.items():
				if hasattr(cfg, k):
					setattr(cfg, k, v)

	set_global_seed(cfg.seed)
	os.makedirs(cfg.output_dir, exist_ok=True)

	data_cfg = DataConfig(
		dataset_dir=cfg.dataset_dir,
		sample_rate=cfg.sample_rate,
		duration_sec=cfg.duration_sec,
		batch_size=cfg.batch_size,
		augment=cfg.augment,
	)
	train_loader, val_loader, test_loader, pos_weight = create_dataloaders(data_cfg)

	model = build_horn_quality_model(
		num_input_channels=1,
		base_channels=cfg.base_channels,
		dropout=cfg.dropout,
		se_reduction=cfg.se_reduction,
	).to(cfg.device)

	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(cfg.device))
	optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

	best_val_auc = -1.0
	best_path = os.path.join(cfg.output_dir, "best.pt")
	last_path = os.path.join(cfg.output_dir, "last.pt")
	no_improve_epochs = 0

	for epoch in range(1, cfg.epochs + 1):
		print(f"Epoch {epoch}/{cfg.epochs}")
		loss = train_epoch(model, train_loader, criterion, optimizer, cfg.device)
		val_metrics = eval_epoch(model, val_loader, cfg.device)
		val_auc = val_metrics["auc"]
		scheduler.step(val_auc if not (val_auc != val_auc) else 0.0)  # handle NaN
		print({"train_loss": loss, "val_auc": val_auc})
		# Save last
		save_checkpoint(last_path, model, optimizer, epoch, best_val_auc)
		# Save best
		if val_auc > best_val_auc:
			best_val_auc = val_auc
			no_improve_epochs = 0
			save_checkpoint(best_path, model, optimizer, epoch, best_val_auc)
		else:
			no_improve_epochs += 1
			if no_improve_epochs >= cfg.patience:
				print("Early stopping")
				break

	# Final test with best model
	ckpt = torch.load(best_path, map_location="cpu") if os.path.exists(best_path) else None
	if ckpt is not None:
		model.load_state_dict(ckpt["model_state"])
		epoch = ckpt.get("epoch", -1)
		print(f"Loaded best checkpoint from epoch {epoch}")
	test_metrics = eval_epoch(model, test_loader, cfg.device)
	print("Test:", test_metrics)

	# Export inference checkpoint
	jit_path = os.path.join(cfg.output_dir, "horn_quality_scripted.pt")
	model_cpu = build_horn_quality_model(
		num_input_channels=1,
		base_channels=cfg.base_channels,
		dropout=cfg.dropout,
		se_reduction=cfg.se_reduction,
	)
	model_cpu.load_state_dict(model.state_dict())
	model_cpu.eval()
	example = torch.randn(1, 1, int(cfg.sample_rate * cfg.duration_sec))
	scripted = torch.jit.trace(model_cpu, example)
	torch.jit.save(scripted, jit_path)
	print(f"Saved TorchScript model to {jit_path}")


if __name__ == "__main__":
	main()