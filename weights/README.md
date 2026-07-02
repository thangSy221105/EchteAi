# Local checkpoints

This directory stores local model checkpoints and is intentionally excluded from Git by `**/*.pt`.

- `fp32_best.pt`: FP32 initialization checkpoint.
- `qat_last.pt`: prepared-QAT resume checkpoint after epoch 1, named for direct Kaggle loading.
- `qat_last_epoch1.pt`: the same checkpoint with an explicit local archive name.
- `selective_int8_epoch1.pt`: converted CPU INT8 checkpoint for inference only; it cannot resume QAT.

Upload the first two files to a private Kaggle Dataset. Do not push them to GitHub because each file exceeds GitHub's file-size limit.
