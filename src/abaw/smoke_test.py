from __future__ import annotations

import argparse

import torch

from abaw.config import load_config
from abaw.data import AFEWVADataset
from abaw.model import AffectFusionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AFEW-VA model forward pass")
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    dataset = AFEWVADataset(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
    )
    sample = dataset[0]
    model = AffectFusionModel(**config["model"])
    model.eval()
    with torch.inference_mode():
        output = model(
            pixel_values=sample["pixel_values"].unsqueeze(0),
            landmarks=sample["landmarks"].unsqueeze(0),
            return_features=True,
        )
    print(f"dataset size: {len(dataset)}")
    print(f"image shape: {tuple(sample['pixel_values'].shape)}")
    print(f"landmarks shape: {tuple(sample['landmarks'].shape)}")
    print(f"target [-1, 1]: {sample['target'].tolist()}")
    print(f"prediction shape: {tuple(output['prediction'].shape)}")
    print(f"image gate: {output['image_gate'].item():.4f}")


if __name__ == "__main__":
    main()

