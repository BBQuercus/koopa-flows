import os
from pathlib import Path

from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopa.preprocess import register_3d_image
from prefect import task


@task(cache_key_fn=task_input_hash)
def preprocess_3D_to_2D(
        img: ImageSource, projection_operator: str, out_dir: Path
) -> ImageTarget:
    data = register_3d_image(img.get_data(), projection_operator)

    output = ImageTarget.from_path(
        path=os.path.join(out_dir, img.get_name() + ".tif"),
        metadata=img.get_metadata(),
        resolution=img.get_resolution(),
    )
    output.set_data(data)
    return output
