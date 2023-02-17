import os
import re
from os.path import join
from typing import Literal

from cpr.Serializer import cpr_serializer
from cpr.image.ImageSource import ImageSource
from cpr.utilities.utilities import task_input_hash
from prefect import flow, task
from pydantic import BaseModel

from koopaflows.preprocessing.task import preprocess_3D_to_2D

@task(cache_key_fn=task_input_hash)
def load_images(input_dir, ext):
    assert ext in ['.tif', '.stk', '.nd', '.czi'], 'File format not supported.'

    pattern_re = re.compile(f".*{ext}")

    files = []
    for entry in os.scandir(input_dir):
        if entry.is_file():
            if pattern_re.fullmatch(entry.name):
                files.append(ImageSource.from_path(entry.path))

    return files


class Preprocess3Dto2D(BaseModel):
    file_extension: str = ".nd"
    projection_operator: Literal["maximum", "mean", "sharpest"] = "maximum"


@flow(
    name="Preprocess 3D to 2D",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def preprocess_flow(
        input_path: str = "/tungstenfs/scratch/gchao/grieesth/Export_DRB/20221216_HeLa11ht-pIM40nuc-JunD-2_HS-42C-30or1h_DRB-4h_washout-30min-1h-2h_smFISH-IF_HSPH1_SC35/",
        output_path: str = "/path/to/output/dir",
        run_name: str = "run-1",
        prepocess: Preprocess3Dto2D = Preprocess3Dto2D(),
):
    run_dir = join(output_path, run_name)

    preprocess_output = join(run_dir,
                             f"preprocess_3D-2D_{prepocess.projection_operator}")
    os.makedirs(preprocess_output, exist_ok=True)

    raw_images = load_images(input_path, ext=prepocess.file_extension)

    preprocessed = []
    for img in raw_images:
        preprocessed.append(
            preprocess_3D_to_2D.submit(
                img=img,
                projection_operator=prepocess.projection_operator,
                output_path=preprocess_output,
            )
        )

    return preprocessed
