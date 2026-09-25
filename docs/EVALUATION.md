# Reconstruction evaluation

This guide covers the **reconstruction** metrics in the ReLMCodec paper using LibriSpeech test-clean and test-other. Run commands from the repository root after installing `python -m pip install -e '.[eval]'` and downloading the matching checkpoint with `scripts/download_weights.py`.

## 1. Data and manifest

Obtain LibriSpeech from the [official OpenSLR release](https://www.openslr.org/12). The root passed below must contain `test-clean/` and `test-other/`, each with FLAC audio and `*.trans.txt` transcripts.

```bash
python scripts/prepare_librispeech_eval.py \
  --root /datasets/LibriSpeech --output-dir data/filelists
```

Expected counts are **2,620 test-clean + 2,939 test-other = 5,559 utterances**. The generated `test-all.txt` contains absolute audio paths, sorted within each split; `transcripts.txt` contains matching utterance IDs and reference text. `--limit-per-split 2` makes a four-file smoke manifest, but must not be used for paper numbers.

## 2. Core reconstruction metrics

```bash
python evaluate.py \
  --checkpoint models/relmcodec_64k.pt \
  --config configs/relmcodec_64k.yaml \
  --filelist data/filelists/test-all.txt \
  --output-dir outputs/64k/reconstructions \
  --result-json outputs/64k/metrics_test-all.json

python evaluate.py \
  --checkpoint models/relmcodec_8k.pt \
  --config configs/relmcodec_8k.yaml \
  --filelist data/filelists/test-all.txt \
  --output-dir outputs/8k/reconstructions \
  --result-json outputs/8k/metrics_test-all.json
```

Each run computes:

| Metric | Protocol | JSON key |
| :--- | :--- | :--- |
| Log-Mel | Sum of L1 losses on `log10` Mel spectra at FFT sizes 32, 64, 128, 256, 512, 1024, 2048 with 5, 10, 20, 40, 80, 160, 320 Mel bands; hop = FFT/4, power = 1, Slaney scaling, clamp = `1e-5` | `mel` |
| PESQ | `pesq(..., mode="wb")` on 16 kHz reference and reconstruction | `pesq` |
| STOI | `pystoi.stoi(..., extended=False)` on 16 kHz audio | `stoi` |

Inputs are mono and resampled to 16 kHz. Reference and reconstruction are trimmed to their common length before metric calculation. One run writes three files: `metrics_test-clean.json` (2,620 utterances), `metrics_test-other.json` (2,939), and `metrics_test-all.json` (5,559). Each contains its own metrics, count, and per-file values. Compare the **test-all** mean with the paper. PESQ/STOI errors are recorded per file; check each metric count before comparing.

## 3. WER with Whisper-Large-v3

Install the [official OpenAI Whisper implementation](https://github.com/openai/whisper) in the codec environment:

```bash
python -m pip install openai-whisper
```

Add `--whisper large-v3 --transcripts data/filelists/transcripts.txt` to either `evaluate.py` command. The script transcribes the **complete utterance** in English, applies OpenAI Whisper's English text normalizer, and reports the **mean utterance WER**. The JSON value is a fraction; multiply by 100 for the paper's percentage. The first call downloads Whisper-Large-v3 unless it is already cached. A local Whisper `.pt` path is also accepted.

To score WAVs already reconstructed by section 2, without another codec forward pass:

```bash
python -m scripts.score_wer_sim \
  --filelist data/filelists/test-all.txt \
  --audio-dir outputs/64k/reconstructions \
  --transcripts data/filelists/transcripts.txt \
  --whisper large-v3 \
  --result-json outputs/64k/wer_test-all.json
```

## 4. Speaker similarity with WavLM-Large-SV

The paper uses the fixed-pretrain **WavLM large** speaker-verification release, with the official [Microsoft UniSpeech speaker-verification wrapper](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification). It is separate from the ordinary `microsoft/wavlm-large` self-supervised checkpoint.

1. Clone `https://github.com/microsoft/UniSpeech` and copy `downstreams/speaker_verification/` to a model directory outside this repository.
2. Download the [official **WavLM large, Fix pre-train = Yes** checkpoint](https://1drv.ms/u/s!AqeByhGUtINrgcp_7CsbcBjYW2Tr-w?e=VeCMic) from the UniSpeech model table into that directory as `wavlm_large_finetune.pth`.
3. Install the wrapper's dependencies, including the [officially specified s3prl revision](https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/README.md):

   ```bash
   python -m pip install 's3prl @ git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3'
   ```

The directory must contain `models/` and `wavlm_large_finetune.pth`. Add `--wavlm-sv /path/to/wavlm-large-sv-eval` to `evaluate.py`, or use the saved-WAV scorer:

```bash
python -m scripts.score_wer_sim \
  --filelist data/filelists/test-all.txt \
  --audio-dir outputs/64k/reconstructions \
  --wavlm-sv /path/to/wavlm-large-sv-eval \
  --result-json outputs/64k/sim_test-all.json
```

Both paths use the official UniSpeech ECAPA-TDNN architecture and the same WavLM-Large-SV checkpoint. The loader takes the upstream WavLM parameters from that checkpoint, so it does not need s3prl's expired WavLM download URL. It requires the UniSpeech-pinned s3prl revision above. SIM is the mean per-utterance cosine similarity between reference and reconstruction embeddings. The saved-WAV path reads 16-bit PCM output; very small differences from the in-memory `evaluate.py` score are expected.

## 5. UTMOS in its own environment

The original evaluation used the [`UTMOS-demo` `Score` API](https://huggingface.co/spaces/sarulab-speech/UTMOS-demo), with `epoch=3-step=7459.ckpt`, on saved reconstructions. Keep its Fairseq dependencies separate from the codec environment if needed.

```bash
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo /path/to/UTMOS-demo
cd /path/to/UTMOS-demo && git lfs pull && cd -

python -m pip install 'Cython<3' bitarray sacrebleu \
  'hydra-core==1.3.2' 'omegaconf==2.3.0' \
  'pytorch-lightning==1.5.10' 'torchmetrics==0.7.2' \
  'pyDeprecate==0.3.1' tensorboard
python -m pip install --no-build-isolation \
  'fairseq @ git+https://github.com/pytorch/fairseq.git@d03f4e771484a433f025f47744017c2eb6e9c6bc'

python scripts/score_utmos.py \
  --audio-dir outputs/64k/reconstructions \
  --utmos-dir /path/to/UTMOS-demo \
  --result-json outputs/64k/utmos_test-all.json
```

The WER, SIM, and UTMOS scorers also write matching `test-clean`, `test-other`, and `test-all` JSON files. The UTMOS scorer expects `score.py`, `epoch=3-step=7459.ckpt`, and `wav2vec_small.pt` in that directory. Install the dependencies listed by that upstream project in the UTMOS environment. The upstream package pins Fairseq at `d03f4e771484a433f025f47744017c2eb6e9c6bc` and PyTorch Lightning 1.5.10. `score_utmos.py` checks 16 kHz input and writes per-file scores, mean, standard deviation, and count. It supports the original checkpoint format with current PyTorch, but load checkpoint files only from a trusted source.

## 6. Compare with the paper

| Model | WER ↓ | SIM ↑ | Log-Mel ↓ | PESQ ↑ | STOI ↑ | UTMOS ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReLMCodec@8K | 4.16% | 0.749 | 1.370 | 2.17 | 0.900 | 4.03 |
| ReLMCodec@64K | 3.96% | 0.804 | 1.270 | 2.40 | 0.917 | 4.07 |

These are **paper results**, not measured values for the downloadable files. The bundled 64K weight is a different fine-tune from the paper's 64K row; see [checkpoint notes](CHECKPOINTS.md). For a reproducible comparison, record evaluator package versions and use the complete 5,559-file manifest. The standalone WER/SIM and UTMOS scorers read saved PCM WAV files.
