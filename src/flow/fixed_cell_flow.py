from glob import glob
from pathlib import Path
from typing import Union, Literal, List
import os

from cpr.Serializer import cpr_serializer
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from prefect import flow, get_client
from prefect import task
import koopa.preprocess
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.futures import PrefectFuture
from pydantic import BaseModel

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
    # TODO: Run all possible coloc-variations
    pass

@task(cache_key_fn=task_input_hash)
def threshold_segmentation_task(
    preprocessed: ImageSource,
    cell_segmentation_threshold: SegmentCellsThreshold,
    output_path: Path,
) -> ImageTarget:
    # TODO
    pass

@task(cache_key_fn=task_input_hash)
def merge(
    spots: List[ParquetSource],
    segmentations: List[ImageSource],
    other_segmentations: List[ImageSource],
    output_path: Path,
) -> tuple[CSVTarget, CSVTarget]:
    # TODO
    pass

@task(
    cache_result_in_memory=False,
    persist_result=True,
    cache_key_fn=task_input_hash,
)
def run_deepblink(
    image_dicts: list[dict],
    output_path: str,
    detection_channels: list[int],
    deepblink_models: list[Path]
):
    parameters = {
        "serialized_preprocessed": image_dicts,
        "out_dir": output_path,
        "detection_channels": detection_channels,
        "deepblink_models": deepblink_models,
    }

    run: FlowRun = run_deployment(
        name="deepblink/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


def load_images(input_path, ext):
    files = glob(os.paht.join(input_path, "*" + ext))
    # TODO: pixel-size and metadata?
    return [ImageSource.from_path(f) for f in files]


class Cellpose(BaseModel):
    model: str = "nuclei"
    seg_channel: int = 0
    nuclei_channel: int = 0
    diameter: float = 40.0
    flow_threshold: float = 0.4
    cell_probability_threshold: float = 0.0
    resample: bool = True
    remove_touching_border: bool = False,

class OutputFormat(BaseModel):
    output_dir: str = "/home/tibuch/Gitrepos/prefect-cellpose/test-output/"
    imagej_compatible: bool = True

@task(
    cache_result_in_memory=False,
    persist_result=True,
    cache_key_fn=task_input_hash,
)
def run_cellpose(
    image_dicts: list[dict],
    cellpose_parameter: Cellpose,
    output_format: OutputFormat
):
    parameters = {
        "image_dicts": image_dicts,
        "cellpose_parameter": cellpose_parameter,
        "output_format": output_format,
    }

    run: FlowRun = run_deployment(
        name="Run cellpose inference 2D/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


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
            "Select only one of the cell segmentation options."
        )
    if cell_segmentation_threshold.active:
        assert not cell_segmentation_cellpose.active, (
            "Select only one of the cell segmentation options."
        )
        if cell_segmentation_threshold.cito:
            assert cell_segmentation_threshold.nuclei, "Cito only is not " "possible."

    if not cell_segmentation_threshold.active and not cell_segmentation_cellpose.active:
        raise RuntimeError("Activate a cell segmentation method.")

    # TODO: Assert only one other segmentation method per channel

    raw_images = load_images(input_path, ext=file_ext)

    preprocessed = []
    for img in raw_images:
        preprocessed.append(preprocess_3D_to_2D(img, projection_operator,
                                                os.path.join(output_path, "preprocessed")))

    if cell_segmentation_cellpose.active:
        # Cellpose segmentation runs in stand-alone flow
        cp_params = Cellpose()
        # TODO: What are the segmentation options? Nuclei & Cito, Nuclei
        #  only, Cito only?
        if cell_segmentation_cellpose.nuclei:
            cp_params.seg_channel = cell_segmentation_cellpose.channel_nuclei
            cp_params.nuclei_channel = cell_segmentation_cellpose.channel_nuclei
        else:
            cp_params.seg_channel = cell_segmentation_cellpose.channel_cito
            # TODO: is the nuclei channel used?
            cp_params.nuclei_channel = cell_segmentation_cellpose.channel_nuclei

        cp_params.diameter = cell_segmentation_cellpose.diameter
        cp_params.resample = cell_segmentation_cellpose.resample
        cp_params.remove_touching_border = cell_segmentation_cellpose.remove_touching_border

        # TODO: There are multiple models. Is cellpose run once with each?
        cp_params.model = cell_segmentation_cellpose.cellpose_models[0]

        output_format = OutputFormat()
        output_format.output_dir = os.path.join(output_path, "cellpose")

        segmentations = run_cellpose.submit(
            image_dicts=[p.serialize() for p in preprocessed],
            cellpose_parameter=cp_params,
            output_format=output_format,
        )
    elif cell_segmentation_threshold.active:
        # This runs inside this infrastructure
        segmentations = []
        for prep in preprocessed:
            segmentations.append(
                threshold_segmentation_task.submit(prep,
                                              cell_segmentation_threshold)
            )
    else:
        raise RuntimeError("No cell segmentation method selected.")

    other_segmentations = []
    if other_segmentation_dl.active or other_segmentation_threshold.active:
        # Stand-alone flow
        other_segmentations = segment_other_flow.submit(
            preprocessed, other_segmentation_dl, other_segmentation_threshold
        )

    # Deepblink runs in a stand-alone flow
    spots = run_deepblink.submit(
        image_dicts=[p.serialize() for p in preprocessed],
        output_path=os.path.join(output_path, "spots"),
        detection_channels=detection_channels,
        deepblink_models=deepblink_models,
    )

    if colocalize.active:
        spots_ = []
        # Wait for spot-detection to finish.
        for spot in spots.result():
            spots_.append(colocalize_spots.submit(spot, colocalize,
                                                 output_path))

        spots = spots_


    # Wait for segmentations to finish.
    if isinstance(segmentations, PrefectFuture):
        segmentations = segmentations.result()

    merge(spots, segmentations, other_segmentations)
