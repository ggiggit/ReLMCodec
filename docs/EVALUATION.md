# Reconstruction evaluation

This guide reproduces the **reconstruction** part of the ReLMCodec paper. It uses the complete LibriSpeech test-clean and test-other splits, not the 50-file dev set used to select the bundled 64K checkpoint. Run all commands from the repository root after installing `python -m pip install -e '.[eval]'` and downloading the matching checkpoint with `scripts/download_weights.py`.

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
  --result-json outputs/64k/metrics.json

python evaluate.py \
  --checkpoint models/relmcodec_8k.pt \
  --config configs/relmcodec_8k.yaml \
  --filelist data/filelists/test-all.txt \
  --output-dir outputs/8k/reconstructions \
  --result-json outputs/8k/metrics.json
```

Each run computes:

| Metric | Protocol | JSON key |
| :--- | :--- | :--- |
| Log-Mel | Sum of L1 losses on `log10` Mel spectra at FFT sizes 32, 64, 128, 256, 512, 1024, 2048 with 5, 10, 20, 40, 80, 160, 320 Mel bands; hop = FFT/4, power = 1, Slaney scaling, clamp = `1e-5` | `mel` |
| PESQ | `pesq(..., mode="wb")` on 16 kHz reference and reconstruction | `pesq` |
| STOI | `pystoi.stoi(..., extended=False)` on 16 kHz audio | `stoi` |

Inputs are mono and resampled to 16 kHz. Reference and reconstruction are trimmed to their common length before metric calculation. The JSON records a combined mean, per-split means, sample counts, and per-file values. Report the **combined mean over 5,559 files**. PESQ/STOI errors are recorded per file; check that each metric's count equals `num_files` before comparing with the paper.

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
  --result-json outputs/64k/wer.json
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
  --result-json outputs/64k/sim.json
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
  --result-json outputs/64k/utmos.json
```

The UTMOS scorer expects `score.py`, `epoch=3-step=7459.ckpt`, and `wav2vec_small.pt` in that directory. Install the dependencies listed by that upstream project in the UTMOS environment. The upstream package pins Fairseq at `d03f4e771484a433f025f47744017c2eb6e9c6bc` and PyTorch Lightning 1.5.10. `score_utmos.py` checks 16 kHz input and writes per-file scores, mean, standard deviation, and count. It supports the original checkpoint format with current PyTorch, but load checkpoint files only from a trusted source.

The full-test verification used Python 3.10.18, PyTorch/torchaudio 2.9.0+cu128, OpenAI Whisper `20250625`, s3prl `0.3.1` at the pinned revision above, Fairseq `1.0.0a0+d03f4e7`, PyTorch Lightning 1.5.10, Hydra Core 1.3.2, and OmegaConf 2.3.0. Evaluator model SHA-256 values were:

| File | SHA-256 |
| :--- | :--- |
| `large-v3.pt` | `e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb` |
| `wavlm_large_finetune.pth` | `cb44af16a20ce497c133cbbaff6cad88728d57bf204048b3fa902cd39feb517b` |
| `epoch=3-step=7459.ckpt` | `44c57e3e4135a243b43d2c82b6a693fcd56f15f9ad0e1eb2a8b31fdecd3a49b8` |
| `wav2vec_small.pt` | `c66c39eaed1b79a61ea8573f71e08f6641ff156b6a8f458cfaab53877dfa4a26` |

## 6. Compare with the paper

| Model | WER ↓ | SIM ↑ | Log-Mel ↓ | PESQ ↑ | STOI ↑ | UTMOS ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReLMCodec@8K | 4.16% | 0.749 | 1.370 | 2.17 | 0.900 | 4.03 |
| ReLMCodec@64K | 3.96% | 0.804 | 1.270 | 2.40 | 0.917 | 4.07 |

The 8K file is the same perceptual 200K checkpoint family used for its paper row. The bundled 64K file is the **available 50K additional fine-tune**. The closest retained full-test record to the paper's 64K row used a 30K fine-tune checkpoint; its weight file was unavailable for this package. Exact 64K equality is therefore not expected even with a matched protocol. Historical full-test result JSON for the original 8K and 30K evaluation is retained under [`results/`](results/); those files are reference evidence, not output from this repository's new commands. Fresh full-test results from this repository are in [`RESULTS.md`](RESULTS.md).

For a reproducible run, record `torch`, `torchaudio`, `transformers`, `pesq`, `pystoi`, Whisper, WavLM-SV, and UTMOS versions along with the checkpoint SHA-256. The default evaluator writes saved audio as PCM WAV; Log-Mel/PESQ/STOI/WER/SIM are computed from the in-memory float reconstruction before file writing. UTMOS reads the saved WAV files.
