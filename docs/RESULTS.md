# Full-test validation on SSH host 4090

The release checkpoints were evaluated on **all 5,559 LibriSpeech test utterances** (2,620 test-clean and 2,939 test-other) using this repository's `evaluate.py`, matching YAML files, and the seven-resolution reconstruction loss described in [EVALUATION.md](EVALUATION.md). Each reported mean is the arithmetic mean over utterances; every core metric has 5,559 valid values.

## Fresh reconstruction results

| Model | Weight SHA-256 prefix | Log-Mel ↓ | PESQ-wb ↑ | STOI ↑ |
| :--- | :--- | ---: | ---: | ---: |
| Paper 8K target | — | 1.370 | 2.17 | 0.900 |
| **Bundled 8K, retested** | `b55ee1aa176a` | **1.370** | **2.171** | **0.900** |
| Paper 64K target | — | 1.270 | 2.40 | 0.917 |
| **Bundled 64K, retested** | `951abb8e1b8a` | **1.287** | **2.372** | **0.916** |

The unrounded 8K means are Log-Mel **1.3703533049**, PESQ **2.1708382840**, and STOI **0.9000300027**. They agree with the paper row to its displayed precision. The unrounded 64K means are Log-Mel **1.2871223025**, PESQ **2.3715504134**, and STOI **0.9158549122**. The 64K Log-Mel is 0.017 above the paper row; PESQ is 0.028 below it; STOI is 0.0011 below it. The 64K release weight is the available 50K fine-tune, while the closest retained full-test record to the paper row used a 30K fine-tune whose weights were unavailable. The comparison therefore tests the *published package*, not bitwise reproduction of the original table checkpoint.

Per-file values, split means, counts, and checkpoint hashes are in the [8K JSON](results/reproduced_8k.json) and [64K JSON](results/reproduced_64k.json). The JSON uses LibriSpeech utterance IDs rather than machine-specific absolute paths. All 5,559 utterance IDs are unique in each file, and neither run had a PESQ or STOI error.

## How this run was made

Environment: Python 3.10.18, PyTorch/torchaudio 2.9.0+cu128, Transformers 4.57.1, NumPy 1.26.4, SoundFile 0.13.1, `pesq` 0.0.4, `pystoi` 0.4.1. Inference used one NVIDIA RTX 4090 (`cuda:7`) on host 4090. `USE_TF=0 TRANSFORMERS_NO_TF=1` was set to avoid unrelated TensorFlow imports. The evaluation manifest was generated with `scripts/prepare_librispeech_eval.py` from `/data/wzx_data/LibriSpeech`.

```bash
python scripts/prepare_librispeech_eval.py \
  --root /data/wzx_data/LibriSpeech --output-dir data/filelists

USE_TF=0 TRANSFORMERS_NO_TF=1 python evaluate.py \
  --checkpoint models/relmcodec_64k.pt \
  --config configs/relmcodec_64k.yaml \
  --filelist data/filelists/test-all.txt \
  --output-dir outputs/64k/reconstructions \
  --result-json outputs/64k/metrics.json \
  --device cuda:7
```

The 8K run used `models/relmcodec_8k.pt`, `configs/relmcodec_8k.yaml`, and a separate output directory with the same command. Both runs used byte-for-byte copies of the bundled weights, verified after transfer: 8K SHA-256 `b55ee1aa176a7a9b3e00b952a07f4932c1770103e6e6e4e4bd91e0d6345d3335`; 64K SHA-256 `951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a`.

## Other paper metrics

WER (Whisper-Large-v3), SIM (WavLM-Large-SV), and UTMOS require three additional public evaluator models; they were **not freshly rerun** in this validation. The release provides their commands in [EVALUATION.md](EVALUATION.md). The compact [historical 8K](results/historical_8k_200k.json) and [historical 64K 30K fine-tune](results/historical_64k_30k.json) full-test result records include these metrics for reference. Those records are labeled historical because the latter uses a different 64K weight, and neither came from this package's new run.
