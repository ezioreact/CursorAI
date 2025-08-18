## Horn Quality Classification (Raw 1D CNN)

Advanced 1D CNN to classify 4-second horn audio as pass/fail directly from raw waveforms. Model uses residual blocks, depthwise-separable convolutions, squeeze-and-excitation, batch norm, and dropout.

### Dataset layout
Place your audio files under a root directory with this structure:

```
/path/to/dataset/
  pass/
    horn_001.wav
    ...
  fail/
    horn_101.wav
    ...
```

Notes:
- WAV recommended. Other formats supported by torchaudio may work (mp3/flac) if codecs are available.
- Any sample rate is accepted; audio is resampled to 16 kHz by default.
- Each clip should be at least 4s. Shorter clips are zero-padded; longer ones are randomly cropped during training.

### Quick start with Docker (recommended)

Build (CPU-only):

```bash
docker build -t horn-quality:cpu .
```

Train:

```bash
docker run --rm -it \
  -v /path/to/dataset:/data \
  -v $(pwd)/outputs:/workspace/outputs \
  horn-quality:cpu \
  python train.py --dataset_dir /data --output_dir /workspace/outputs
```

Infer on a single file:

```bash
docker run --rm -it \
  -v /path/to/audio.wav:/in.wav \
  -v $(pwd)/outputs:/workspace/outputs \
  horn-quality:cpu \
  python infer.py --audio /in.wav --ckpt /workspace/outputs/best.pt
```

### Local setup (if you already have a working Python + PyTorch)

```bash
python -m venv .venv
. .venv/bin/activate
# Install torch/torchaudio matching your system (example below is CPU only):
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchaudio==2.3.1
pip install -r requirements.txt
```

Train:

```bash
python train.py --dataset_dir /path/to/dataset --output_dir ./outputs
```

Infer:

```bash
python infer.py --audio /path/to/sample.wav --ckpt ./outputs/best.pt
```

### Configuration
Override defaults via YAML:

```bash
python train.py --dataset_dir /data --output_dir ./outputs --config config.example.yaml
```

See `config.example.yaml` for fields.

### Model and training details
- Input: mono waveform, 4.0 s at 16 kHz (resampled automatically)
- Loss: BCE with logits + `pos_weight` for class imbalance
- Sampling: `WeightedRandomSampler` to balance batches
- Scheduler: ReduceLROnPlateau on validation AUC
- Early stopping: patience 8 epochs
- Export: best checkpoint (`outputs/best.pt`) and TorchScript (`outputs/horn_quality_scripted.pt`)

### Tips for small dataset (100 pass / 50 fail)
- Keep `augment: true` (default) to improve generalization
- Consider more epochs with stronger regularization: increase `dropout` to 0.2, use smaller `base_channels` (32)
- Use mixed validation: run several times with different splits (filename-hash split is used by default)
- If possible, add more fail samples or synthetically create near-fail variants to balance the dataset.

### Inference output
`infer.py` prints a dict like:

```python
{"prob_pass": 0.83, "pred": "pass"}
```

### Repository layout
- `horn_quality/`: model, data, augmentation, utilities
- `train.py`: training/eval/early-stopping/checkpointing
- `infer.py`: single-file inference
- `config.example.yaml`: example training configuration
- `README.md`: this file
- `Dockerfile`: CPU-only docker image with pinned PyTorch/torchaudio