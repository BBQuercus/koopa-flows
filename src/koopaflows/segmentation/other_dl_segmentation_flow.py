import threading
from os import makedirs
from os.path import join
from threading import Semaphore
from typing import Any, Optional

import koopa.segment_other_deep as kocp
import numpy as np
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopaflows.cpr_parquet import koopa_serializer
from koopaflows.storage_key import RESULT_STORAGE_KEY
from koopaflows.utils import wait_for_task_runs
from prefect import task, flow
from prefect.context import TaskRunContext
from pydantic import BaseModel


class SegmentOtherDL(BaseModel):
    active: bool = True
    fname_model: str = ""
    backbone: str = ""
    channel: int = 0


def exlude_semaphore_and_model_task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if (not isinstance(item, threading.Semaphore)) and (
            not isinstance(item, kocp.DeepSegmentation)
        ):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@task(cache_key_fn=exlude_semaphore_and_model_task_input_hash,
      result_storage_key=RESULT_STORAGE_KEY)
def segment_other_dl_task(
        img: ImageTarget,
        output_dir: str,
        segment_other: SegmentOtherDL,
        segmenter: kocp.DeepSegmentation,
        gpu_semaphore: Semaphore,
):
    input_img = img.get_data()[segment_other.channel:segment_other.channel+1]

    try:
        masks = [segmenter.segment(frame) for frame in input_img]
    except Exception as e:
        raise e
    finally:
        gpu_semaphore.release()

    masks = np.array(masks).squeeze()

    result = ImageTarget.from_path(
        join(output_dir, img.get_name() + ".tif"),
        imagej=False,
    )
    result.set_data(
        masks.astype(np.uint16)
    )

    return result


@flow(
    name="other-segmentation-DL",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
)
def other_segmentation_DL_flow(
        image_dicts: list[dict],
        output_dir: str,
        segment_other: SegmentOtherDL = SegmentOtherDL(),
):
    if 'data_hash' in image_dicts[0].keys():
        images = [ImageTarget(**d) for d in image_dicts]
    else:
        images = [ImageSource(**d) for d in image_dicts]

    segmentation_result: list[dict[str, ImageTarget]] = []

    other_seg_output = join(output_dir,
                            f"segmentation_c{segment_other.channel}")
    makedirs(other_seg_output, exist_ok=True)

    segmenter = kocp.DeepSegmentation(
        fname_model=segment_other.fname_model,
        backbone=segment_other.backbone,
    )

    gpu_semaphore = Semaphore(1)
    buffer = []
    for img in images:
        buffer.append(
            segment_other_dl_task.submit(
                img=img,
                output_dir=other_seg_output,
                segment_other=segment_other,
                segmenter=segmenter,
                gpu_semaphore=gpu_semaphore,
            )
        )

        wait_for_task_runs(
            results=segmentation_result,
            buffer=buffer,
            max_buffer_length=2,
            result_insert_fn=lambda r: {f"other_c{segment_other.channel}": r}
        )

    wait_for_task_runs(
        results=segmentation_result,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=lambda r: {f"other_c{segment_other.channel}": r}
    )

    return segmentation_result

