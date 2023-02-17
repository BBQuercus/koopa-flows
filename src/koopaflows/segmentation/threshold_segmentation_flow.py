from os import makedirs
from os.path import join
from typing import Literal

import skimage
from cpr.Serializer import cpr_serializer
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from prefect import task, flow
from pydantic import BaseModel
import koopa.segment_cells_threshold as ksct

class SegmentNuclei(BaseModel):
    channel: int = 2
    gaussian: int = 3
    min_size_nuclei: int = 1000
    min_distance: int = 50


class SegmentCyto(BaseModel):
    active: bool = True
    channel: int = 1
    method: Literal["otsu", "li", "triangle"] = "triangle"
    upper_clip: float = 0.95
    gaussian: int = 3
    min_size: int = 5000

@task(cache_key_fn=task_input_hash)
def segment_nuclei_task(
        img: ImageTarget,
        output_dir: str,
        segment_nuclei: SegmentNuclei
):
    result = ImageTarget.from_path(
        join(output_dir, img.get_name() + ".tif")
    )
    result.set_data(
        ksct.segment_nuclei(
            image=img.get_data()[segment_nuclei.channel],
            gaussian=segment_nuclei.gaussian,
            min_size_nuclei=segment_nuclei.min_size_nuclei,
            min_distance=segment_nuclei.min_distance,
        )
    )
    return result

@task(cache_key_fn=task_input_hash)
def segment_cyto_task(
        img: ImageTarget,
        nuc_seg: ImageTarget,
        output_dir: str,
        segment_cyto: SegmentCyto,
):
    result = ImageTarget.from_path(
        join(output_dir, img.get_name() + ".tif")
    )
    image_cyto = img.get_data()[segment_cyto.channel]
    segmap_cyto = ksct.segment_background(
        image=image_cyto,
        method=segment_cyto.method,
        upper_clip=segment_cyto.upper_clip,
        gaussian=segment_cyto.gaussian,
        min_size=segment_cyto.min_size,
    )
    segmap_cyto = skimage.segmentation.watershed(
        image=~image_cyto,
        markers=nuc_seg.get_data(),
        mask=segmap_cyto,
        watershed_line=True,
    )
    result.set_data(segmap_cyto)
    return result

@flow(
    name="Koopa - Segmentation [2D]",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=cpr_serializer(),
)
def threshold_segmentation_flow(
        images: list[ImageTarget],
        output_dir: str,
        segment_nuclei: SegmentNuclei = SegmentNuclei(),
        segment_cyto: SegmentCyto = SegmentCyto()
):
    segmentation_result: list[dict[str, ImageTarget]] = []

    nuc_seg_output = join(output_dir, "segmentation_nuclei")
    makedirs(nuc_seg_output, exist_ok=True)

    cyto_seg_output = join(output_dir, "segmentation_cyto")
    makedirs(cyto_seg_output, exist_ok=True)

    for img in images:
        result = {}
        nuc_seg_task = segment_nuclei_task.submit(
            img=img,
            output_dir=nuc_seg_output,
            segment_nuclei=segment_nuclei
        )
        result["nuclei"] = nuc_seg_task

        if segment_cyto.active:
            cyto_seg_task = segment_cyto_task.submit(
                img=img,
                nuc_seg=nuc_seg_task,
                output_dir=cyto_seg_output,
                segment_cyto=segment_cyto
            )
            result["cyto"] = cyto_seg_task

        segmentation_result.append(result)

    return segmentation_result

