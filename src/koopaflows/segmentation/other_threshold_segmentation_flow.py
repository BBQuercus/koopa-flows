from os import makedirs
from os.path import join
from typing import Literal

import koopa.segment_other_threshold as koct
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from prefect import task, flow
from pydantic import BaseModel


class SegmentOther(BaseModel):
    active: bool = True
    method: Literal["otsu", "li", "multiotsu"] = "multiotsu"
    channel: int = 0


@task(cache_key_fn=task_input_hash)
def segment_other_task(
        img: ImageTarget,
        output_dir: str,
        segment_other: SegmentOther
):
    result = ImageTarget.from_path(
        join(output_dir, img.get_name() + ".tif")
    )
    result.set_data(
        koct.segment(
            image=img.get_data(),
            channel=segment_other.channel,
            method=segment_other.method,
        )
    )
    return result


@flow(
    name="Koopa - Segmentation Other [2D]",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def other_threshold_segmentation_flow(
        images: list[ImageTarget],
        output_dir: str,
        segment_other: SegmentOther = SegmentOther(),
):
    segmentation_result: list[dict[str, ImageTarget]] = []

    other_seg_output = join(output_dir,
                            f"segmentation_c{segment_other.channel}")
    makedirs(other_seg_output, exist_ok=True)

    for img in images:
        segmentation_result.append(
            {
                f"other_c{segment_other.channel}": segment_other.submit(
                    img=img,
                    output_dir=other_seg_output,
                    segment_other=segment_other,
                )
            }
        )

    return segmentation_result
