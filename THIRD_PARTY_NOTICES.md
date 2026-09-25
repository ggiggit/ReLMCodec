# Third-party notices

- The trainable acoustic encoder is adapted from the public
  [X-Codec-2.0 implementation](https://github.com/zhenye234/X-Codec-2.0)
  (MIT, Copyright 2025 YE Zhen). Its license is retained at
  `relmcodec/vendor/xcodec2/LICENSE`.
- The Vocos backbone and ISTFT head are adapted from Vocos/WavTokenizer-style public
  implementations. Their source notices are retained in the corresponding files.
- The alias-free activation implementation under
  `relmcodec/vendor/xcodec2/vq/alias_free_torch/` is adapted from `alias-free-torch` under
  Apache-2.0, as noted in the source headers.
- Snake/SnakeBeta is adapted from the public Snake implementation; the source header
  is retained in `relmcodec/vendor/xcodec2/vq/activations.py`.
- Both bundled ReLMCodec checkpoints include the frozen
  [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)
  parameters (upstream model card: MIT license). WavLM-Large-SV, Whisper, and UTMOS
  evaluator weights are **not** included and retain their upstream licenses.
