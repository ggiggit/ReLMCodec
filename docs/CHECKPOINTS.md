# Checkpoint provenance and selection

The two downloadable checkpoint files are **inference-only** `torch.save` dictionaries. Each contains a `model` state dict with **1,317 tensor names** plus a few plain provenance fields. The model state includes the frozen W2v-BERT 2.0 encoder, trainable XCodec2 acoustic encoder, PAPA adapter, quantizer, Vocos decoder, and ISTFT head. The files contain no optimizer state, scheduler, AMP scaler, discriminator, Lightning loop/callback state, or RNG snapshot.

| Published file | Source experiment | Source step | Size |
| :--- | :--- | ---: | ---: |
| `models/relmcodec_8k.pt` | 8K WavLM-SV perceptual stage, `step200000.ckpt` inference export | 200,000 | 3,131,979,506 bytes |
| `models/relmcodec_64k.pt` | 64K WavLM-SV perceptual **fine-tune**, `last.ckpt` | 50,000 fine-tune steps | 3,135,963,435 bytes |

SHA-256 of the hosted weight files:

```text
relmcodec_8k.pt   b55ee1aa176a7a9b3e00b952a07f4932c1770103e6e6e4e4bd91e0d6345d3335
relmcodec_64k.pt  951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a
```

The 64K source was a full Lightning training checkpoint. Its model tensors were remapped and strictly matched to the standalone inference model; **252 non-model entries** in its `state_dict` (discriminators and Mel-loss buffers) and all optimizer/scheduler/loop metadata were excluded. The 8K source was already a Lightning-format inference export; its model tensors were remapped the same way.

Two extra feature-normalizer buffers in the source checkpoints have exactly `mean=0` and `std=1` in all 2,048 dimensions. The standalone model applies no feature normalization. Conversion checked those values before excluding the buffers; a non-identity normalizer would have stopped conversion.

## Why this 64K checkpoint?

The previously exported 64K inference checkpoint came from the 200K perceptual stage. The closest retained full-test record to the paper's 64K row used a **30K additional fine-tune** checkpoint, but that exact weight file is no longer available locally. A 50K fine-tune `last.ckpt` was available and was compared with the 200K checkpoint on 50 held-out LibriSpeech dev utterances, sampled evenly from sorted dev-clean and dev-other file lists (25 from each). The exact IDs are in [`dev_selection_ids.txt`](dev_selection_ids.txt).

| Candidate | Seven-resolution Log-Mel ↓ | PESQ-wb ↑ | STOI ↑ |
| :--- | ---: | ---: | ---: |
| 64K perceptual 200K | 1.32935 | 2.22928 | 0.90791 |
| **64K fine-tune 50K** | **1.27314** | **2.34414** | **0.91641** |

All 50 files were reconstructed with the same standalone code and `l2_normalize: false`. Log-Mel was computed in memory with PyTorch/torchaudio 2.1.1; PESQ/STOI were computed on the generated float WAVs with the corresponding Python packages in the 4090 environment. This independent dev comparison selected the 50K file before any new test-set run. These dev values are not directly comparable with the paper's test-set values.

## Quantizer rule in the available weights

The source checkpoint hyperparameter is `quantizer_type: vq_v2`. The original `vq_v2` implementation uses Euclidean nearest-code lookup in the eight-dimensional projected space; its separate `vq_v2_l2` class performs L2-normalized lookup. The manuscript describes the latter, but these available checkpoints use the former. Therefore both bundled YAML files explicitly set `quantizer.l2_normalize: false`. This choice follows the checkpoint's actual token assignments and is necessary for a faithful reconstruction comparison. Switching it to `true` changes the inferred codes and metrics without changing a single tensor.

## Inspect locally

```bash
python - <<'PY'
import torch
for variant in ('8k', '64k'):
    ckpt = torch.load(f'models/relmcodec_{variant}.pt', map_location='cpu', weights_only=True, mmap=True)
    print(variant, list(ckpt), len(ckpt['model']))
    assert 'model' in ckpt and len(ckpt['model']) == 1317
    assert all(isinstance(value, torch.Tensor) for value in ckpt['model'].values())
    assert not {'optimizer_states', 'lr_schedulers', 'rng_state', 'state_dict'} & ckpt.keys()
PY

python scripts/smoke_test.py --variant 8k
python scripts/smoke_test.py --variant 64k
```

Use the matching config and weight file together. Model loading is strict; a missing, unexpected, or shape-incompatible tensor is an error.
