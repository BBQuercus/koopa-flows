from pathlib import Path
from typing import Union, Literal, List
import os

from cpr.Serializer import cpr_serializer
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from prefect import flow
from prefect import task
import koopa.preprocess

from .deepblink_flow import deepblink_spot_detection_flow
from .flow_parameters import Colocalize
from .flow_parameters import SegmentCellsCellpose
from .flow_parameters import SegmentCellsThreshold
from .flow_parameters import SegmentOtherSegmentationModels
from .flow_parameters import SegmentOtherThreshold
from .cpr_parquet import ParquetSource
from .cpr_parquet import ParquetTarget


@task(cache_key_fn=task_input_hash)
def preprocess_3D_to_2D(
    img: ImageSource, projection_operator: str, output_path: Path
) -> ImageTarget:
    data = koopa.preprocess.register_3d_image(img.get_data(), projection_operator)

    output = ImageTarget.from_path(
        path=os.path.join(output_path, img.get_name() + ".tif"),
        metadata=img.get_metadata(),
        resolution=img.get_resolution(),
    )
    output.set_data(data)
    return output


@task(cache_key_fn=task_input_hash)
def colocalize_spots(
    spot: dict[int, ParquetSource], colocalize: Colocalize, output_path: Path
) -> ParquetTarget:
    # Run all possible coloc-variations
    pass


def threshold_segmentation_task(
    preprocessed: ImageSource,
    cell_segmentation_threshold: SegmentCellsThreshold,
    output_path: Path,
) -> ImageTarget:
    pass


def merge(
    spots: List[ParquetSource],
    segmentations: List[ImageSource],
    other_segmentations: List[ImageSource],
    output_path: Path,
) -> tuple[CSVTarget, CSVTarget]:
    pass


@flow(
    name="Fixed Cell Analysis",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def fixed_cell_flow(
    input_path: Union[Path, str] = "/path/to/acquisition/dir",
    output_path: Union[Path, str] = "/path/to/output/dir",
    file_ext: str = ".nd",
    projection_operator: Literal["maximum", "mean", "sharpest"] = "maximum",
    detection_channels: List[int] = [0],
    deepblink_models: List[Union[Path, str]] = ["/path/to/deepblink/model"],
    colocalize: Colocalize = Colocalize(),
    cell_segmentation_cellpose: SegmentCellsCellpose = SegmentCellsCellpose(),
    cell_segmentation_threshold: SegmentCellsThreshold = SegmentCellsThreshold(),
    other_segmentation_dl: SegmentOtherSegmentationModels = SegmentOtherSegmentationModels(),
    other_segmentation_threshold: SegmentOtherThreshold = SegmentOtherThreshold(),
):
    if cell_segmentation_cellpose.active:
        assert not cell_segmentation_threshold.active, (
            "Select one of the " "cell segmentation " "options."
        )
    if cell_segmentation_threshold.active:
        assert not cell_segmentation_cellpose.active, (
            "Select one of the " "cell segmentation " "options."
        )
        if cell_segmentation_threshold.cito:
            assert cell_segmentation_threshold.nuclei, "Cito only is not " "possible."

    if not cell_segmentation_threshold.active and not cell_segmentation_cellpose.active:
        raise RuntimeError("Activate a cell segmentation method.")

    # TODO: Assert only one other segmentation method per channel

    raw_images = load_images(input_path, ext=file_ext)

    preprocessed = []
    preprocessed_out = os.path.join(output_path, "preprocessed")
    for img in raw_images:
        preprocessed.append(preprocess_3D_to_2D(img, projection_operator, output_path))

    spots = deepblink_spot_detection_flow(
        preprocessed, detection_channels, deepblink_models, output_path
    )

    if colocalize.active:
        spots_ = []
        for spot in spots:
            spots_.append(colocalize_spots(spot, colocalize, output_path))

        spots = spots_

    if cell_segmentation_cellpose.active:
        segmentations = cellpose_flow(preprocessed, cell_segmentation_cellpose)
    elif cell_segmentation_threshold.active:
        segmentations = []
        for prep in preprocessed:
            segmentations.append(
                threshold_segmentation_task(prep, cell_segmentation_threshold)
            )
    else:
        raise RuntimeError("No cell segmentation method selected.")

    other_segmentations = []
    if other_segmentation_dl.active or other_segmentation_threshold.active:
        other_segmentations = segment_other_flow(
            preprocessed, other_segmentation_dl, other_segmentation_threshold
        )

    merge(spots, segmentations, other_segmentations)
