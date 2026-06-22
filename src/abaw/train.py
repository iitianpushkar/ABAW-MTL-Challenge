from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from abaw.config import load_config
from abaw.data import AFEWVADataset, split_video_ids
from abaw.losses import HybridVALoss, concordance_correlation_coefficient
from abaw.model import AffectFusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AFEW-VA fusion baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: AffectFusionModel,
    loader: DataLoader,
    criterion: HybridVALoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch in tqdm(loader, leave=False):
        pixel_values = batch["pixel_values"].to(device)
        landmarks = batch["landmarks"].to(device)
        target = batch["target"].to(device)
        with torch.set_grad_enabled(training):
            prediction = model(pixel_values=pixel_values, landmarks=landmarks)
            loss = criterion(prediction, target)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
    epoch_prediction = torch.cat(predictions)
    epoch_target = torch.cat(targets)
    ccc = concordance_correlation_coefficient(epoch_prediction, epoch_target)
    return float(np.mean(losses)), ccc


def main() -> None:
    config = load_config(parse_args().config)
    seed_everything(config["seed"])
    device = select_device()
    train_ids, val_ids = split_video_ids(
        config["data"]["root"],
        config["data"]["val_fraction"],
        config["seed"],
    )
    train_dataset = AFEWVADataset(
        config["data"]["root"],
        train_ids,
        config["data"]["image_size"],
        augment=True,
    )
    val_dataset = AFEWVADataset(
        config["data"]["root"],
        val_ids,
        config["data"]["image_size"],
    )
    loader_args = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"]["num_workers"],
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = AffectFusionModel(**config["model"]).to(device)
    criterion = HybridVALoss(
        config["training"]["ccc_weight"], config["training"]["mse_weight"]
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    print(
        f"device={device} train_frames={len(train_dataset)} "
        f"val_frames={len(val_dataset)}"
    )
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_ccc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.inference_mode():
            val_loss, val_ccc = run_epoch(model, val_loader, criterion, device)
        val_score = val_ccc.mean().item()
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_ccc={train_ccc.mean().item():.4f} val_loss={val_loss:.4f} "
            f"valence_ccc={val_ccc[0].item():.4f} "
            f"arousal_ccc={val_ccc[1].item():.4f}"
        )
        if val_score > best_score:
            best_score = val_score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_ccc": val_ccc,
                    "train_video_ids": train_ids,
                    "val_video_ids": val_ids,
                },
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

