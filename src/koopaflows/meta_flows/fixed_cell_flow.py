import os
from glob import glob
from os.path import join
from pathlib import Path
from typing import Union, List

import pandas as pd
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from koopa.postprocess import get_segmentation_data
from koopaflows.cpr_parquet import ParquetSource, koopa_serializer
from koopaflows.preprocessing.flow import Preprocess3Dto2D, load_images
from koopaflows.preprocessing.task import load_and_preprocess_3D_to_2D
from koopaflows.segmentation.other_threshold_segmentation_flow import \
    SegmentOther, segment_other_task
from koopaflows.segmentation.threshold_segmentation_flow import SegmentNuclei, \
    SegmentCyto, segment_nuclei_task, segment_cyto_task
from koopaflows.utils import wait_for_task_runs
from prefect import flow, get_client
from prefect import task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture


@task(
    cache_key_fn=task_input_hash,
)
def run_deepblink(
        image_dicts: list[dict],
        output_path: str,
        run_name: str,
        detection_channels: list[int],
        deepblink_models: list[Path]
):
    parameters = {
        "serialized_preprocessed": image_dicts,
        "output_path": output_path,
        "run_name": run_name,
        "detection_channels": detection_channels,
        "deepblink_models": deepblink_models,
    }

    run: FlowRun = run_deployment(
        name="deepblink/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


@task(cache_key_fn=task_input_hash)
def merge(
    all_spots: List[dict[int, ParquetSource]],
    segmentations: List[dict[str, ImageSource]],
    other_segmentations: List[dict[int, ImageSource]],
    output_path: str,
) -> tuple[CSVTarget, CSVTarget]:

    result_dfs, result_cell_dfs = [], []
    for all_spots_per_image, nuc_cyto_seg, other_seg in zip(all_spots,
                                                      segmentations, other_segmentations):
        dfs = []
        for spots_per_channel in all_spots_per_image.values():
            dfs.append(spots_per_channel.get_data())

        df = pd.concat(dfs)

        segs = {}
        for k, v in nuc_cyto_seg.items():
            segs[k] = v.get_data()

        for k, v in other_seg.items():
            segs[k] = v.get_data()

        df, cell_df = get_segmentation_data(
            df,
            segs,
            {
                "do_3d": False,
                "brains_enabled": False,
                "selection": "both",
            }
        )
        result_dfs.append(df)
        result_cell_dfs.append(cell_df)

    result_dfs = pd.concat(result_dfs)
    result_cell_dfs = pd.concat(result_cell_dfs)

    result = CSVTarget.from_path(os.path.join(output_path, "summary.csv"))
    result_cells = CSVTarget.from_path(os.path.join(output_path,
                                                    "summary_cells.csv"))

    result.set_data(result_dfs)
    result_cells.set_data(result_cell_dfs)
    return result, result_cells


def cell_segmentation(
    images: list[ImageTarget],
    output_dir: str,
    segment_nuclei: SegmentNuclei,
    segment_cyto: SegmentCyto
):
    nuc_seg_output = join(output_dir, "segmentation_nuclei")
    os.makedirs(nuc_seg_output, exist_ok=True)

    cyto_seg_output = join(output_dir, "segmentation_cyto")
    os.makedirs(cyto_seg_output, exist_ok=True)

    nuc_results = []
    buffer = []
    for img in images:
        buffer.append(
            segment_nuclei_task.submit(
                img=img,
                output_dir=nuc_seg_output,
                segment_nuclei=segment_nuclei
            )
        )
        wait_for_task_runs(
            results=nuc_results,
            buffer=buffer,
            max_buffer_length=48,
        )

    wait_for_task_runs(
        results=nuc_results,
        buffer=buffer,
        max_buffer_length=0
    )

    if segment_cyto.active:
        buffer = []
        cyto_results = []
        for img, nuc in zip(images, nuc_results):
            buffer.append(
                segment_cyto_task.submit(
                    img=img,
                    nuc_seg=nuc,
                    output_dir=cyto_seg_output,
                    segment_cyto=segment_cyto
                )
            )

            wait_for_task_runs(
                results=cyto_results,
                buffer=buffer,
                max_buffer_length=48,
            )

        wait_for_task_runs(
            results=cyto_results,
            buffer=buffer,
            max_buffer_length=0,
        )

        results = []
        for nuc, cyto in zip(nuc_results, cyto_results):
            results.append(
                {
                    "nuclei": nuc,
                    "cyto": cyto,
                }
            )
    else:
        results = [ {"nuclei": nuc} for nuc in nuc_results ]

    return results


def other_segmentation(
    preprocessed: list[ImageTarget],
    output_dir: str,
    segment_other: SegmentOther
):
    other_seg_output = join(output_dir,
                            f"segmentation_c{segment_other.channel}")
    os.makedirs(other_seg_output, exist_ok=True)

    other_segmentations: list[dict[str, ImageTarget]] = []
    buffer = []
    for img in preprocessed:
        buffer.append(
            segment_other_task.submit(
                img=img,
                output_dir=other_seg_output,
                segment_other=segment_other,
            )
        )

        wait_for_task_runs(
            results=other_segmentations,
            buffer=buffer,
            max_buffer_length=6,
            result_insert_fn=lambda r: {f"other_c{segment_other.channel}":
                                            r.result()}
        )

    wait_for_task_runs(
        results=other_segmentations,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=lambda r: {f"other_c{segment_other.channel}":
                                        r.result()}
    )

    return other_segmentations


def preprocessing(
    raw_files: list[ImageSource],
    output_dir: str,
    preprocess: Preprocess3Dto2D,
):
    preprocess_output = join(output_dir,
                             "preprocessd")
    os.makedirs(preprocess_output, exist_ok=True)

    preprocessed = []
    buffer = []
    for file in raw_files:
        buffer.append(
            load_and_preprocess_3D_to_2D.submit(
                file=file,
                ext=preprocess.file_extension,
                projection_operator=preprocess.projection_operator,
                out_dir=preprocess_output,
            )
        )

        wait_for_task_runs(
            results=preprocessed,
            buffer=buffer,
            max_buffer_length=20,
        )

    wait_for_task_runs(
        results=preprocessed,
        buffer=buffer,
        max_buffer_length=0
    )

    return preprocessed


@flow(
    name="Fixed Cell Analysis",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("koopa"),
)
def fixed_cell_flow(
        input_path: Union[Path, str] = "/tungstenfs/scratch/gchao/grieesth/Export_DRB/20221216_HeLa11ht-pIM40nuc-JunD-2_HS-42C-30or1h_DRB-4h_washout-30min-1h-2h_smFISH-IF_HSPH1_SC35/",
        output_path: Union[Path, str] = "/path/to/output/dir",
        run_name: str = "run-1",
        preprocess: Preprocess3Dto2D = Preprocess3Dto2D(),
        deepblink_models: List[Union[Path, str]] = [
            "/tungstenfs/scratch/gchao/deepblink/model_fish.h5"],
        detection_channels: List[int] = [1],
        segment_nuclei: SegmentNuclei = SegmentNuclei(),
        segment_cyto: SegmentCyto = SegmentCyto(),
        segment_other: SegmentOther = SegmentOther(),
):
    raw_files = load_images(input_path, preprocess.file_extension)

    preprocessed = preprocessing(
        raw_files=raw_files,
        output_dir=join(output_path, run_name),
        preprocess=preprocess
    )

    # Deepblink runs in GPU TensorFlow env
    spots = run_deepblink.submit(
        image_dicts=[p.serialize() for p in preprocessed],
        output_path=output_path,
        run_name=run_name,
        detection_channels=detection_channels,
        deepblink_models=deepblink_models,
    )

    cell_segmentations = cell_segmentation(
        preprocessed,
        os.path.join(output_path, run_name),
        segment_nuclei,
        segment_cyto,
    )

    other_segmentations = other_segmentation(
        preprocessed,
        os.path.join(output_path, run_name),
        segment_other
    )

    merge(
        all_spots=spots.result(),
        segmentations=cell_segmentations,
        other_segmentations=other_segmentations,
        output_path=join(output_path, run_name)
    )
