import gc
import os
from os.path import basename, splitext
from pathlib import Path

import numpy as np
import psutil
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopa.io import load_raw_image
from koopa.preprocess import register_3d_image, crop_image, bin_image
from koopaflows.storage_key import RESULT_STORAGE_KEY
from prefect import task, get_run_logger


@task(cache_key_fn=task_input_hash)
def load_and_preprocess_3D_to_2D(
        file: str,
        ext: str,
        projection_operator: str,
        out_dir: Path
) -> ImageTarget:
    data = register_3d_image(
        load_raw_image(fname=file, file_ext=ext),
        projection_operator
    )

    name, _ = splitext(basename(file))
    output = ImageTarget.from_path(
        path=os.path.join(out_dir, name + ".tif"),
    )
    output.set_data(data)
    return output


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def load_and_preprocess_brains(
        file: str,
        ext: str,
        crop_start: int,
        crop_end: int,
        scale_factors: list[float],
        out_dir: Path
) -> ImageTarget:
    logger = get_run_logger()
    gc.collect()
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.debug(f"[Start] Process memory usage: {mem_usage} GB")
    logger.info(f"Loading file: {file}")
    data = load_raw_image(fname=file, file_ext=ext)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.debug(f"[Load Raw] Process memory usage: {mem_usage} GB")

    logger.debug(f"Cropping image with shape {data.shape}.")
    data = crop_image(
        image=data,
        crop_start=crop_start,
        crop_end=crop_end,
    )
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.debug(f"[Crop] Process memory usage: {mem_usage} GB")

    logger.debug(f"Bin cropped image with shape {data.shape}.")
    logger.debug(f"    scale_factors = {scale_factors}")
    data = bin_image(
        image=data,
        bin_axes=scale_factors,
    )
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.debug(f"[Bin] Process memory usage: {mem_usage} GB")

    logger.debug(f"Final image shape {data.shape}.")
    name, _ = splitext(basename(file))
    output = ImageTarget.from_path(
        path=os.path.join(out_dir, name + ".tif"),
        metadata={
            'axes': 'CZYX',
        },
        imagej=False
    )
    output.set_data(data)
    logger.debug(output.metadata)
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1e9
    logger.debug(f"[End] Process memory usage: {mem_usage} GB")
    return output