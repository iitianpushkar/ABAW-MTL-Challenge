# ABAW 2026 Research Project

This repository starts from the proven Colab prototype: a pretrained ViT encodes facial appearance, a Transformer encodes 68 facial landmarks, bidirectional cross-attention fuses both streams, and a regression head predicts valence and arousal.

The current AFEW-VA subset contains 50 videos and 2,381 annotated frames. Its labels are in `[-10, 10]`; the loader scales them to `[-1, 1]` to match Aff-Wild2 and the model's bounded output.

## Setup

Use Python 3.11 or 3.12. The existing `myenv` uses Python 3.14 and should not be used for the PyTorch environment.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first run downloads the pretrained ViT weights from Hugging Face.

## Sanity Check

```bash
abaw-smoke --config configs/baseline.yaml
```

Expected tensor shapes are image `(3, 224, 224)`, landmarks `(68, 2)`, and prediction `(1, 2)`. The prediction is random before training.

## Train

```bash
abaw-train --config configs/baseline.yaml
```

The split is made by video rather than frame, preventing neighboring frames from the same clip leaking into validation. The best checkpoint is written to `outputs/baseline/best.pt`.

## Architecture Direction

The strongest direction is a progressive, reliability-aware temporal MTL system:

1. Pretrain separate task specialists for VA, expression, and action units.
2. Extract frame tokens from a face-affect backbone and geometry tokens from landmarks; add audio and sparse VLM behavior embeddings only after strong visual baselines exist.
3. Replace unconditional feature addition with directed cross-modal experts and a quality gate. Face detection confidence, landmark stability, blur, occlusion, and speech activity should influence each modality's weight.
4. Run a lightweight temporal model over 16-32 frame windows. Predict the center frame and average overlapping-window predictions.
5. Optimize a hybrid CCC and pointwise loss, then use task-aware gradient balancing when EXPR and AU heads are added.

The novel core can be a **Quality-Gated Progressive Cross-Task Temporal Transformer**: task-specialist tokens act as experts, directed cross-attention models asymmetric transfer, and quality-conditioned routing suppresses harmful experts.

## Experiment Order

Run controlled ablations in this order:

1. Image-only baseline.
2. Landmark-only baseline.
3. Current frame-level fusion.
4. Temporal fusion with video-level splits.
5. VA plus auxiliary EXPR/AU supervision.
6. Audio features.
7. Sparse VLM behavior features.
8. Reliability-aware expert routing and ensembles.

Do not judge the architecture from AFEW-VA numbers alone. Use it to validate data, optimization, and temporal code; final model selection must use the official ABAW split and metric.
