from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class SegmentCellsCellpose(BaseModel):
    active: bool = True
    nuclei: bool = True
    cito: bool = False
    channel_nuclei: int
    channel_cito: int
    diameter: int = 150
    cellpose_models: list[Path] = ['/to/models']
    resample: bool = True
    remove_touching_border: bool = True


class SegmentCellsThreshold(BaseModel):
    active: bool = False
    nuclei: bool = True
    cito: bool = False
    channel_nuclei: int
    channel_cito: int
    method: Literal["otsu", "triangle", "li"] = "otsu"
    remove_touching_border: bool = True
    gaussian_sigma: int = 3
    upper_clip: float = 0.95
    min_distance: int = 50


class SegmentOtherSegmentationModels(BaseModel):
    active: bool = False
    channels: list[int] = [0]
    models: list[Path] = ['/path/to/model']
    backbones: list[str] = ['inception']


class SegmentOtherThreshold(BaseModel):
    active: bool = False
    channels: list[int] = [0]
    method: Literal["otsu", "li", "multi-otsu"] = "multi-otsu"

class Colocalize(BaseModel):
    active: bool = True
    channels: list[tuple[int, int]]