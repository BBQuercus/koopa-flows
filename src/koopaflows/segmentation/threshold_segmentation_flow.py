from glob import glob
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Literal

import koopa.segment_cells_threshold as ksct
import skimage
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopaflows.cpr_parquet import koopa_serializer
from prefect import task, flow, get_client
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from pydantic import BaseModel


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
    name="cell-seg-threshold-2d",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("koopa"),
)
def threshold_segmentation_flow(
        serialized_images: list[dict],
        output_dir: str,
        segment_nuclei: SegmentNuclei = SegmentNuclei(),
        segment_cyto: SegmentCyto = SegmentCyto()
):
    images = [ImageSource(**d) for d in serialized_images]
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

@flow(
    name="Cell-Segmentation 2D",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("koopa"),
)
def run_cell_seg_threshold_2d(
    input_path: Path = "/path/to/input_dir/",
    output_path: Path = "/path/to/output_dir",
    run_name: str = "run-1",
    pattern: str = "*.tif",
    segment_nuclei: SegmentNuclei = SegmentNuclei(),
    segment_cyto: SegmentCyto = SegmentCyto(),
):
    images = [ImageSource.from_path(p) for p in glob(join(input_path,
                                                          pattern))]

    parameters = {
        "serialized_images": [img.serialize() for img in images],
        "output_dir": join(output_path, run_name),
        "segment_nuclei": segment_nuclei.dict(),
        "segment_cyto": segment_cyto.dict(),
    }

    run: FlowRun = run_deployment(
        name="cell-seg-threshold-2d/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()
