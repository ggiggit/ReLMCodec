# ReLMCodec · Inference checkpoints

**Single-stream, 50 Hz speech coding at 650 / 800 bps.**

[Paper](https://arxiv.org/abs/2608.08286) · [Project page](https://github.com/ggiggit/ReLMCodec) · [Reproduction results](docs/RESULTS.md)

![Figure 3 from the ReLMCodec paper: preserve, control, and refine](assets/paper-figure-3-architecture.png)

This model repository contains the **complete ReLMCodec inference release**: both codec checkpoints, their matching YAML configs, the frozen W2v-BERT 2.0 parameters embedded in each checkpoint, and the code needed to reconstruct audio. The weights have no optimizer, scheduler, discriminator, or RNG state.

| Variant | Codebook | Rate | Checkpoint | Size | SHA-256 |
| :--- | ---: | ---: | :--- | ---: | :--- |
| 8K | 8,192 | 650 bps | `models/relmcodec_8k.pt` | 3.13 GB | `b55ee1aa176a7a9b3e00b952a07f4932c1770103e6e6e4e4bd91e0d6345d3335` |
| 64K | 65,536 | 800 bps | `models/relmcodec_64k.pt` | 3.14 GB | `951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a` |

The 8K checkpoint is the 200K perceptual-stage export. The 64K checkpoint is the available 50K additional fine-tune, selected by Log-Mel/PESQ/STOI on a separate 50-file dev set. The closest retained record to the paper's 64K table used an unavailable 30K fine-tune weight. [Checkpoint provenance](docs/CHECKPOINTS.md) records this distinction and the actual Euclidean VQ assignment rule used by the released weights.

## Quick inference

Use Python 3.10 or 3.11 on Linux. From a local snapshot of this model repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
export USE_TF=0

python scripts/smoke_test.py --variant 64k
python infer.py reconstruct \
  --config configs/relmcodec_64k.yaml \
  --checkpoint models/relmcodec_64k.pt \
  --input input.wav --output reconstructed.wav
```

The model file and its config must match. Input audio is downmixed and resampled to 16 kHz. `infer.py encode` saves token IDs; `infer.py decode` reconstructs them. See [Inference](docs/INFERENCE.md) for commands and token format.

## From the paper

Across 24 representations under a matched quantizer and language model, pre-quantization phoneme separability correlates with next-token accuracy (Spearman ρ = 0.911). Figure 3 shows how ReLMCodec preserves frozen SSL phoneme structure, controls acoustic adaptation with PAPA, and refines the latent space with a training-only WavLM teacher.

![Table 2 from the ReLMCodec paper: end-to-end reconstruction comparison](assets/paper-table-2-reconstruction.png)

These images are cropped from the [paper PDF](https://arxiv.org/pdf/2608.08286). Table 2 gives the **paper's** LibriSpeech reconstruction results. The released checkpoints were rerun on all 5,559 test-clean + test-other utterances; [verified release results](docs/RESULTS.md) include exact metrics, per-file JSON, and checkpoint hashes. [Evaluation](docs/EVALUATION.md) gives the reproduction protocol and external WER, SIM, and UTMOS evaluators. Those three external metrics were not freshly rerun for the packaged weights.

## Scope and license

These weights are for speech reconstruction and token extraction, with mono 16 kHz input. The paper's predictability probes and downstream TTS models are not included. Code is [MIT licensed](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) cover embedded W2v-BERT parameters and vendored components.

```bibtex
@misc{wan2026relmcodec,
  title         = {ReLMCodec: Designing Predictable Speech Tokens from Pre-Quantization Phoneme Structure},
  author        = {Zixiang Wan and Xusheng Yang and Zheng Wang and Peiji Yang},
  year          = {2026},
  eprint        = {2608.08286},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2608.08286}
}
```
