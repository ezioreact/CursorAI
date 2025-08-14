from __future__ import annotations
import os
import random
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def set_global_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: list[int]) -> torch.Tensor:
	num_pos = sum(labels)
	num_neg = len(labels) - num_pos
	return torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32)


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
	probs = torch.sigmoid(logits)
	return (probs >= threshold).to(torch.int64)


def evaluate_classification(y_true: list[int], y_scores: list[float], threshold: float = 0.5) -> dict:
	probs = np.asarray(y_scores)
	y_true_arr = np.asarray(y_true)
	y_pred = (probs >= threshold).astype(int)
	try:
		auc = roc_auc_score(y_true_arr, probs)
	except Exception:
		auc = float("nan")
	report = classification_report(y_true_arr, y_pred, target_names=["fail", "pass"], digits=4, output_dict=True)
	cm = confusion_matrix(y_true_arr, y_pred)
	return {
		"auc": auc,
		"report": report,
		"confusion_matrix": cm.tolist(),
	}


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_auc: float, **extra) -> None:
	state = {
		"model_state": model.state_dict(),
		"optimizer_state": optimizer.state_dict(),
		"epoch": epoch,
		"best_val_auc": best_val_auc,
	}
	state.update(extra)
	torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
	ckpt = torch.load(path, map_location="cpu")
	model.load_state_dict(ckpt["model_state"])
	if optimizer is not None and "optimizer_state" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer_state"])
	return ckpt