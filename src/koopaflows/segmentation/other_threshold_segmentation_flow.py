from glob import glob
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Literal

import koopa.segment_other_threshold as koct
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopaflows.cpr_parquet import koopa_serializer
from prefect import task, flow, get_client, get_run_logger
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel


class SegmentOther(BaseModel):
    active: bool = True
    method: Literal["otsu", "li", "multiotsu"] = "multiotsu"
    channel: int = 0

@task(cache_key_fn=task_input_hash)
def segment_other_task(
        img: ImageTarget,
        output_dir: str,
        segment_other
):
    result = ImageTarget.from_path(
        join(output_dir, img.get_name() + ".tif")
    )
    get_run_logger().info("Start thresholding.")
    mask = koct.segment(
            image=img.get_data()[segment_other.channel],
            method=segment_other.method,
        )
    get_run_logger().info("Done thresholding.")

    result.set_data(
        mask
    )
    get_run_logger().info("Return result.")
    return result


@flow(
    name="other-seg-threshold-2d",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
)
def other_threshold_segmentation_flow(
        serialized_images: list[dict],
        output_dir: str,
        segment_other: SegmentOther = SegmentOther(),
):
    images = [ImageSource(**d) for d in serialized_images]
    segmentation_result: list[dict[str, ImageTarget]] = []

    other_seg_output = join(output_dir,
                            f"segmentation_c{segment_other.channel}")
    makedirs(other_seg_output, exist_ok=True)

    buffer = []
    for img in images:
        buffer.append(
            segment_other_task.submit(
                img=img,
                output_dir=other_seg_output,
                segment_other=segment_other,
            )
        )

        while len(buffer) >= 1:
            segmentation_result.append(
                {
                    f"other_c{segment_other.channel}": buffer.pop(0).result()
                }
            )

    while len(buffer) > 0:
        segmentation_result.append(
            {
                f"other_c{segment_other.channel}": buffer.pop(0).result()
            }
        )

    return segmentation_result

@flow(
    name="Other-Segmentation 2D",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("koopa"),
)
def run_other_threshold_segmentation(
    input_path: Path = "/path/to/input_dir/",
    output_path: Path = "/path/to/output_dir",
    run_name: str = "run-1",
    pattern: str = "*.tif",
    segment_other: SegmentOther = SegmentOther(),
):
    images = [ImageSource.from_path(p) for p in glob(join(input_path, pattern))]

    parameters = {
        "serialized_images": [img.serialize() for img in images],
        "output_dir": join(output_path, run_name),
        "segment_other": segment_other.dict(),
    }

    run: FlowRun = run_deployment(
        name="other-seg-threshold-2d/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()
