# Inference

Run every command from the repository root with the [installation steps](../README.md#quick-start) completed. The 8K config and checkpoint must be used together, as must the 64K pair.

## Check the checkpoint

```bash
python scripts/download_weights.py --source hf --variant 64k
python scripts/smoke_test.py --variant 64k
```

The helper accepts `--source modelscope` and `--variant 8k` or `all`. It verifies the download. The smoke test loads the model, encodes one second of synthetic audio, decodes it, and checks finite output.

## Reconstruct an audio file

```bash
python infer.py reconstruct \
  --config configs/relmcodec_64k.yaml \
  --checkpoint models/relmcodec_64k.pt \
  --input input.wav \
  --output reconstructed.wav
```

Input can be a format supported by SoundFile, including WAV or FLAC. It is downmixed to mono and resampled to 16 kHz. The output is a 16 kHz WAV. For long recordings, split the input into shorter utterances to keep peak memory manageable. The SSL frontend may produce slightly fewer than 50 frames for very short clips; output length is determined by the emitted token count.

## Save and decode tokens

```bash
python infer.py encode \
  --config configs/relmcodec_64k.yaml \
  --checkpoint models/relmcodec_64k.pt \
  --input input.wav \
  --output tokens.pt

python infer.py decode \
  --config configs/relmcodec_64k.yaml \
  --checkpoint models/relmcodec_64k.pt \
  --input tokens.pt \
  --output decoded.wav
```

`tokens.pt` is a PyTorch dictionary with `indices`, `sample_rate`, `hop_length`, and `source_samples`. `indices` has one row of integer IDs (50 frames per second nominally). Keep the variant and its checkpoint with the token file; 8K and 64K token IDs are not interchangeable. `torch.load(..., weights_only=True)` is used for token input.

For direct Python use, load the YAML with `relmcodec.config.load_config`, create the model with `ReLMCodec.from_checkpoint`, call `model.eval()`, and use `model.encode(audio)` or `model.decode(indices)` under `torch.inference_mode()`. Tensors are batch-first. See [`infer.py`](../infer.py) for the complete path, including audio conversion and output writing.

## Evaluation

[Evaluation](EVALUATION.md) provides the LibriSpeech manifest, metric definitions, Whisper/WavLM-SV/UTMOS setup, and commands for the paper's reconstruction table.
