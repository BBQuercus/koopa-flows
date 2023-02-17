import os
from os.path import basename, splitext
from pathlib import Path

from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopa.io import load_raw_image
from koopa.preprocess import register_3d_image
from prefect import task


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
