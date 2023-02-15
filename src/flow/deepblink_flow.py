import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import deepblink as pink
import prefect
import tensorflow as tf
from cpr.Serializer import cpr_serializer
from cpr.image.ImageSource import ImageSource
from cpr.utilities.utilities import task_input_hash
from koopa.detect import detect_image

from .cpr_parquet import ParquetTarget


def exclude_sem_and_model_input_hash(
        context: "TaskRunContext", arguments: Dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if ((not isinstance(item, threading.Semaphore)) and
                (not isinstance(item, tf.keras.models.Model))):
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@prefect.task(
    name="deepblink-single-channel",
    cache_result_in_memory=False,
    persist_result=True,
    cache_key_fn=exclude_sem_and_model_input_hash,
)
def deepblink_spot_detection_task(
        image: ImageSource,
        detection_channel: int,
        out_dir: Path,
        model: tf.keras.Model,
        model_name: str,
        gpu_sem: threading.Semaphore
):
    logger = prefect.get_run_logger()
    logger.info(f"Detect spots in {image.get_path()} with model {model_name}.")

    data = image.get_data()

    try:
        gpu_sem.acquire()
        df = detect_image(
            data, detection_channel, model, refinement_radius=3
        )
    except RuntimeError as e:
        raise e
    finally:
        gpu_sem.release()

    df.insert(loc=0, column="FileID", value=image.get_name())

    fname_out = os.path.join(
        out_dir, f"test2d_detection_c{detection_channel}",
        f"{image.get_name()}.parq"
    )
    output = ParquetTarget.from_path(fname_out)
    output.set_data(df)
    return output


@prefect.flow(
    name="deepblink",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
    validate_parameters=False,
)
def deepblink_spot_detection_flow(
        serialized_preprocessed: List[dict],
        out_dir: Path,
        detection_channels: List[int],
        deepblink_models: List[Path],
) -> Dict[int, List[ParquetTarget]]:
    preprocessed = [ImageSource(**d) for d in serialized_preprocessed]

    gpu_sem = threading.Semaphore(1)

    output = {}
    for channel, model_path in zip(detection_channels, deepblink_models):
        model = pink.io.load_model(model_path)
        detections = []
        buffer = []
        for img in preprocessed:
            task = deepblink_spot_detection_task.submit(
                image=img,
                detection_channel=channel,
                out_dir=out_dir,
                model=model,
                model_name=model_path,
                gpu_sem=gpu_sem,
            )
            buffer.append(task)
            while len(buffer) >= 1:
                # With len(buffer) >= 2 deep_blink_spot_detection_task
                # crashes "sometimes" in `koopa.detect.detect_image`.
                # Don't know really why. Might be related to numba.
                detections.append(buffer.pop(0).wait())

        while len(buffer) > 0:
            detections.append(buffer.pop(0).wait())

        output[channel] = detections

    return output
