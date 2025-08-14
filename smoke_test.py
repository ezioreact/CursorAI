import torch
from horn_quality.model import build_horn_quality_model


def main():
	model = build_horn_quality_model(base_channels=32)
	model.eval()
	batch = torch.randn(2, 1, 16000 * 4)
	with torch.no_grad():
		out = model(batch)
	print("OK", out.shape)


if __name__ == "__main__":
	main()