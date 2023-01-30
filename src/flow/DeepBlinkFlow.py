from os.path import join
from pathlib import Path
from typing import Union, Literal

from cpr.Serializer import cpr_serializer
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from prefect import flow

from src.flow.FixedCellFlow import load_images, preprocess_3D_to_2D
from src.flow.ParquetResource import ParquetTarget


@flow(
    name="deepblink",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def deepblink_spot_detection_flow(
        preprocessed: list[ImageSource],
        detection_channels: list[int],
        deepblink_models: list[Path],
        output_dir: Path,
) -> list[dict[int, ParquetTarget]]:
    # Run deepblink somehow
    pass


@flow(
    name="DeepBlink Spot Detection",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def run_deepblink(
        input_dir: Union[Path, str] = "/path/to/acquisition/dir",
        output_dir: Union[Path, str] = "/path/to/output/dir",
        file_ext: str = ".nd",
        detection_channels: list[int] = [0],
        deepblink_models: list[Path] = ['/path/to/model'],
):

    raw_images = load_images(input_dir, ext=file_ext)

    deepblink_spot_detection_flow(raw_images, detection_channels,
                                          deepblink_models,
                                          output_dir)