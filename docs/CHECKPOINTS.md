# Checkpoints

The primary 8K and 64K checkpoints include the frozen W2v-BERT 2.0 frontend and contain only inference weights. Use each file with its matching config.

The 8K release comes from the 200K perceptual stage. The primary 64K release is the best retained perceptual fine-tune, selected on a separate development set. The paper's 64K table used a different fine-tune checkpoint, so an exact numerical match is not expected.

Both released weights use **Euclidean nearest-code assignment**. Keep `quantizer.l2_normalize: false` in the bundled configs: changing it alters token IDs and reconstruction quality.

For comparisons, the model hubs also provide the Stage 1 checkpoints `relmcodec_8k_stage1.pt` and `relmcodec_64k_stage1.pt`. Stage 1 does not use the WavLM perceptual loss. These supplementary exports use the same architecture and matching configs; they are not the paper's primary results.

Download them with `python scripts/download_weights.py --variant 8k-stage1` or `--variant 64k-stage1`. Pass the supplementary path to `infer.py --checkpoint` with its matching config. To check one, run `python scripts/smoke_test.py --variant 8k --checkpoint models/relmcodec_8k_stage1.pt` (or `64k`).

Before running a full evaluation, check the downloaded model with `python scripts/smoke_test.py --variant 64k` (or `8k`).
