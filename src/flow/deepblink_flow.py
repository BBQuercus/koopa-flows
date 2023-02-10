from pathlib import Path
from typing import Union, Literal, List, Dict
import os

from cpr.Serializer import cpr_serializer
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
import deepblink as pink
import koopa

# import numpy as np
import prefect

from .cpr_parquet import ParquetTarget


@prefect.task(
    name="deepblink-single-channel",
    cache_result_in_memory=False,
    persist_result=True,
    cache_key_fn=task_input_hash,
)
def deepblink_spot_detection_task(
    image: ImageSource, detection_channel: int, deepblink_model: Path
) -> ParquetTarget:
    logger = prefect.get_run_logger()
    logger.info("running task")

    model = pink.io.load_model(deepblink_model)
    df = koopa.detect.detect_image(
        image.get_data(), detection_channel, model, refinement_radius=3
    )
    df.insert(loc=0, column="FileID", value=image.get_name())
    logger.info("detected")

    path_base = os.path.abspath(os.path.join(image.get_path(), os.pardir, os.pardir))
    fname_out = os.path.join(
        path_base, f"test2d_detection_c{detection_channel}", f"{image.get_name()}.parq"
    )
    output = ParquetTarget.from_path(fname_out)
    output.set_data(df)

    logger.info(f"image name - {image.get_name()}")
    logger.info(f"output name - {fname_out}")
    logger.info(f"output path - {output.get_path()}")
    logger.info("finished task")
    return output


@prefect.flow(
    name="deepblink",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    validate_parameters=False,
)
def deepblink_spot_detection_flow(
    preprocessed: ImageSource,
    detection_channels: List[int],
    deepblink_models: List[Path],
) -> List[Dict[int, ParquetTarget]]:
    logger = prefect.get_run_logger()
    logger.info("running sub flow")

    for channel, model in zip(detection_channels, deepblink_models):
        output = deepblink_spot_detection_task.submit(
            preprocessed, channel, model
        ).wait()

    logger.info(f"output from task - {output}")
    logger.info("finished sub flow")


@prefect.flow(
    name="DeepBlink Spot Detection",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    validate_parameters=False,
)
def run_deepblink(
    input_path: Path = "/path/to/acquisition/dir",
    output_path: Path = "/path/to/output/dir",
    file_ext: str = "nd",
    detection_channels: List[int] = [0],
    deepblink_models: List[Path] = ["/path/to/model.h5"],
):
    logger = prefect.get_run_logger()
    logger.info("running main flow")

    # TODO add nd file parsing from input_path
    # TODO add preprocessing step saving in output_path
    fname = "/tungstenfs/scratch/gchao/eichbast/koopa-data/input_2d/20221115_EGFP_R3_3h_16.tif"
    deepblink_models = ["/tungstenfs/scratch/gchao/deepblink/model_fish.h5"]

    image = koopa.io.load_image(fname)
    output = ImageTarget.from_path(fname)
    output.set_data(image)
    deepblink_spot_detection_flow(output, detection_channels, deepblink_models)

    logger.info("finished main flow")
