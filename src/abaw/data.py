from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class FrameRecord:
    image_path: Path
    landmarks: list[list[float]]
    target: tuple[float, float]
    video_id: str
    frame_id: str


def normalize_landmarks(landmarks: torch.Tensor) -> torch.Tensor:
    centered = landmarks - landmarks.mean(dim=0, keepdim=True)
    scale = centered.square().sum(dim=-1).sqrt().amax().clamp_min(1e-6)
    return centered / scale


def split_video_ids(
    root: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    root = Path(root)
    video_ids = sorted(path.parent.name for path in root.glob("*/*.json"))
    if len(video_ids) < 2:
        raise ValueError(f"At least two videos are required under {root}")
    generator = random.Random(seed)
    generator.shuffle(video_ids)
    val_count = max(1, round(len(video_ids) * val_fraction))
    return sorted(video_ids[val_count:]), sorted(video_ids[:val_count])


class AFEWVADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        video_ids: list[str] | None = None,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.records = self._load_records(video_ids)
        if not self.records:
            raise ValueError(f"No annotated frames found under {self.root}")

        if augment:
            image_ops = [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.02,
                ),
            ]
        else:
            image_ops = [transforms.Resize((image_size, image_size))]
        self.image_transform = transforms.Compose(
            [
                *image_ops,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _load_records(self, video_ids: list[str] | None) -> list[FrameRecord]:
        selected = set(video_ids) if video_ids is not None else None
        records: list[FrameRecord] = []
        for annotation_path in sorted(self.root.glob("*/*.json")):
            video_id = annotation_path.parent.name
            if selected is not None and video_id not in selected:
                continue
            with annotation_path.open(encoding="utf-8") as handle:
                annotation = json.load(handle)
            for frame_id, frame in sorted(annotation["frames"].items()):
                image_path = annotation_path.parent / f"{frame_id}.png"
                if not image_path.exists() or len(frame.get("landmarks", [])) != 68:
                    continue
                records.append(
                    FrameRecord(
                        image_path=image_path,
                        landmarks=frame["landmarks"],
                        target=(frame["valence"] / 10.0, frame["arousal"] / 10.0),
                        video_id=video_id,
                        frame_id=frame_id,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            pixel_values = self.image_transform(image.convert("RGB"))
        landmarks = normalize_landmarks(torch.tensor(record.landmarks, dtype=torch.float32))
        target = torch.tensor(record.target, dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "landmarks": landmarks,
            "target": target,
            "video_id": record.video_id,
            "frame_id": record.frame_id,
        }
