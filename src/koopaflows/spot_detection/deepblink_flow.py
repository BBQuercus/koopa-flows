import os
import threading
from os.path import join
from pathlib import Path
from typing import List, Dict, Any, Optional

import deepblink as pink
import prefect
import tensorflow as tf
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopa.detect import detect_image
from koopaflows.cpr_parquet import ParquetTarget, koopa_serializer
from prefect.filesystems import LocalFileSystem


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
    refresh_cache=True,
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
            data, detection_channel, model, refinement_radius=3,
            engine="numba",
        )
    except RuntimeError as e:
        raise e
    finally:
        gpu_sem.release()

    df.insert(loc=0, column="FileID", value=image.get_name())

    fname_out = os.path.join(
        out_dir, f"detection_raw_c{detection_channel}",
        f"{image.get_name()}.parq"
    )
    output = ParquetTarget.from_path(fname_out)
    output.set_data(df)
    return output


@prefect.flow(
    name="deepblink",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("deepblink"),
)
def deepblink_spot_detection_flow(
        serialized_preprocessed: List[dict],
        output_path: str,
        run_name: str,
        detection_channels: List[int],
        deepblink_models: List[Path],
):
    run_dir = join(output_path, run_name)

    preprocess_output = run_dir
    os.makedirs(preprocess_output, exist_ok=True)

    if 'data_hash' in serialized_preprocessed[0].keys():
        preprocessed = [ImageTarget(**d) for d in serialized_preprocessed]
    else:
        preprocessed = [ImageSource(**d) for d in serialized_preprocessed]

    gpu_sem = threading.Semaphore(1)

    output_channels = []
    for channel, model_path in zip(detection_channels, deepblink_models):
        model = pink.io.load_model(model_path)
        detections = []
        buffer = []
        for img in preprocessed:
            buffer.append(
                deepblink_spot_detection_task.submit(
                    image=img,
                    detection_channel=channel,
                    out_dir=preprocess_output,
                    model=model,
                    model_name=model_path,
                    gpu_sem=gpu_sem,
                )
            )

            while len(buffer) >= 4:
                detections.append(buffer.pop(0).result())

        while len(buffer) > 0:
            detections.append(buffer.pop(0).result())

        output_channels.append(detections)

    output = []
    for i in range(len(output_channels[0])):
        img_results = {}
        for j, ch in enumerate(detection_channels):
            img_results[ch] = output_channels[j][i]
        output.append(img_results)

    return output
