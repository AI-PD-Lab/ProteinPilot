# Checkpoints

Hosted in the Hugging Face model repo [`AI-PD/Harmonia`](https://huggingface.co/AI-PD/Harmonia). Download with
`python checkpoints/download.py --model rfd3|rfd4|all`.

| File | Model | What it is |
|---|---|---|
| `Harmonia-RFD3.ckpt` | Harmonia-RFD3 | RFdiffusion3 + SS/ADJ + ss_ratio conditioning, epoch 698 of the final fine-tuning stage (phase 10). 2.7 GB. |
| `Harmonia-RFD4/config.json`, `Harmonia-RFD4/model.safetensors` | Harmonia-RFD4 | RFD4-Proteina + SS/ADJ conditioning, release R3 (8,000 steps, EMA). RFD4 bundle format. 1.7 GB. |

Loading: see [`rfd3/README.md`](../rfd3/README.md) and [`rfd4/README.md`](../rfd4/README.md). Both checkpoints are released under the BSD
3-Clause licence of their base models.
