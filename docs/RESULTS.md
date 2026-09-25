# Full-test validation on SSH host 4090

The release checkpoints were evaluated on **all 5,559 LibriSpeech test utterances** (2,620 test-clean and 2,939 test-other). `evaluate.py` computed Log-Mel, PESQ, and STOI from the in-memory reconstruction; the saved 16 kHz WAVs were scored with Whisper-Large-v3, WavLM-Large-SV, and UTMOS using the commands in [EVALUATION.md](EVALUATION.md). Every metric below has 5,559 valid values.

## Fresh reconstruction results

| Model | Weight SHA-256 prefix | Log-Mel ↓ | PESQ-wb ↑ | STOI ↑ |
| :--- | :--- | ---: | ---: | ---: |
| Paper 8K target | — | 1.370 | 2.17 | 0.900 |
| **Bundled 8K, retested** | `b55ee1aa176a` | **1.370** | **2.171** | **0.900** |
| Paper 64K target | — | 1.270 | 2.40 | 0.917 |
| **Bundled 64K, retested** | `951abb8e1b8a` | **1.287** | **2.372** | **0.916** |

The unrounded 8K means are Log-Mel **1.3703533049**, PESQ **2.1708382840**, and STOI **0.9000300027**. They agree with the paper row to its displayed precision. The unrounded 64K means are Log-Mel **1.2871223025**, PESQ **2.3715504134**, and STOI **0.9158549122**. The 64K Log-Mel is 0.017 above the paper row; PESQ is 0.028 below it; STOI is 0.0011 below it. The 64K release weight is the available 50K fine-tune, while the closest retained full-test record to the paper row used a 30K fine-tune whose weights were unavailable. The comparison therefore tests the *published package*, not bitwise reproduction of the original table checkpoint.

### README command recheck · 2026-09-25

We reran the README's 64K evaluation path from a fresh source export of GitHub commit `04f78b4` on host 4090. In an existing Python 3.10.18 evaluation environment with PyTorch/torchaudio 2.9.0+cu128, `python -m pip install -e '.[eval]'` succeeded. With `LIBRISPEECH_ROOT=/data/wzx_data/LibriSpeech`, the manifest command produced 2,620 test-clean and 2,939 test-other paths (5,559 unique files). A four-file `--max-files 4` smoke run passed, followed by the full `python evaluate.py` command without `--max-files` on one RTX 4090 (`CUDA_VISIBLE_DEVICES=7`).

The full command exited **0** and wrote **5,559 reconstruction WAVs** plus a result JSON with **5,559 per-file entries**. Log-Mel, PESQ-wb, and STOI each have **5,559 valid values**, with no PESQ/STOI errors. The rerun means were Log-Mel **1.287122302538715**, PESQ-wb **2.3715504134252443**, and STOI **0.9158549122489322**. All **5,559 per-file metric triples** remained unchanged when the external metrics were added to the published [64K JSON](results/reproduced_64k.json). The tested weight was the published inference file with SHA-256 `951abb8e1b8af372c963b9ca360c3246ee9ed255a6c0f251018fe3223846231a`; it was linked from the existing local copy to avoid a second 3.1 GB download. WER, speaker similarity, and UTMOS were run separately on the saved WAVs as described below.

Per-file values, split means, counts, and checkpoint hashes for all six metrics are in the [8K JSON](results/reproduced_8k.json) and [64K JSON](results/reproduced_64k.json). The JSON uses LibriSpeech utterance IDs rather than machine-specific absolute paths. All 5,559 utterance IDs are unique in each file, and neither run had a PESQ or STOI error.

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

## WER, speaker similarity, and UTMOS · fresh full-test runs

| Model | WER ↓ | SIM ↑ | UTMOS ↑ |
| :--- | ---: | ---: | ---: |
| Paper 8K | 4.16% | 0.749 | 4.03 |
| **Bundled 8K, retested** | **3.950%** | **0.751** | **4.027** |
| Paper 64K | 3.96% | 0.804 | 4.07 |
| **Bundled 64K, retested** | **3.824%** | **0.807** | **4.029** |

These runs used the official OpenAI Whisper-Large-v3, Microsoft UniSpeech WavLM-Large-SV, and UTMOS-demo model files, with SHA-256 values and dependency versions in [EVALUATION.md](EVALUATION.md). WER is the mean per-utterance error after Whisper's English text normalization; SIM is the mean cosine similarity of reference and reconstruction speaker embeddings; UTMOS is the mean model score of the reconstructed WAVs. The full 8K and 64K results contain **5,559 finite values and 5,559 unique utterance IDs for each metric**. The 64K release is the available 50K fine-tune; the closest retained record to the paper row used an unavailable 30K fine-tune.

Whisper's complete-utterance path was checked on the longest saved sample (**34.94 s**), then with `evaluate.py --whisper` on four freshly reconstructed samples, and finally on all 5,559 saved reconstructions for each variant. The four WER shards partitioned the manifest without overlap. The 8K test-clean/test-other means were **2.261% / 5.456%**; the 64K means were **2.313% / 5.170%**. SIM was also tested through `evaluate.py --wavlm-sv` on four freshly reconstructed samples before the full saved-WAV runs. UTMOS was tested on four samples before the full runs. The compact [historical 8K](results/historical_8k_200k.json) and [historical 64K 30K fine-tune](results/historical_64k_30k.json) records remain as separate reference evidence.
