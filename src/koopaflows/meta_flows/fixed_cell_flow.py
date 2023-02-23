import os
from glob import glob
from os.path import join
from pathlib import Path
from typing import Union, List

import pandas as pd
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.utilities.utilities import task_input_hash
from koopa.postprocess import get_segmentation_data
from koopaflows.cpr_parquet import ParquetSource, koopa_serializer
from koopaflows.preprocessing.flow import Preprocess3Dto2D
from koopaflows.preprocessing.task import load_and_preprocess_3D_to_2D
from koopaflows.segmentation.other_threshold_segmentation_flow import \
    SegmentOther
from koopaflows.segmentation.threshold_segmentation_flow import SegmentNuclei, \
    SegmentCyto
from prefect import flow, get_client
from prefect import task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem


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

@task(
    cache_key_fn=task_input_hash,
)
def run_cell_segmentation(
        serialized_images: list[dict],
        output_dir: str,
        segment_nuclei: SegmentNuclei,
        segment_cyto: SegmentCyto,
):
    parameters = {
        "serialized_images": serialized_images,
        "output_dir": output_dir,
        "segment_nuclei": segment_nuclei.dict(),
        "segment_cyto": segment_cyto.dict(),
    }

    run: FlowRun = run_deployment(
        name="cell-seg-threshold-2d/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


@task(
    cache_key_fn=task_input_hash,
)
def run_other_segmentation(
        serialized_images: list[dict],
        output_dir: str,
        segment_other: SegmentOther,
):
    parameters = {
        "serialized_images": serialized_images,
        "output_dir": output_dir,
        "segment_other": segment_other.dict(),
    }

    run: FlowRun = run_deployment(
        name="other-seg-threshold-2d/default",
        parameters=parameters,
        client=get_client(),
    )

    return run.state.result()


@task(cache_key_fn=task_input_hash)
def load_images(input_path, ext):
    assert ext in ['tif', 'stk', 'nd', 'czi'], 'File format not supported.'
    files = glob(os.path.join(input_path, "*." + ext))
    return [ImageSource.from_path(f) for f in files]


@task(cache_key_fn=task_input_hash)
def merge(
    all_spots: List[List[ParquetSource]],
    segmentations: List[dict[str, ImageSource]],
    other_segmentations: List[ImageSource],
    output_path: str,
) -> tuple[CSVTarget, CSVTarget]:

    result_dfs, result_cell_dfs = [], []
    for all_spots_per_image, nuc_cyto_seg, other_seg in zip(all_spots,
                                                      segmentations, other_segmentations):
        dfs = []
        for spots_per_channel in all_spots_per_image:
            dfs.append(spots_per_channel.get_data())

        df = pd.concat(dfs)

        segs = {**nuc_cyto_seg, **other_seg}

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
    run_dir = join(output_path, run_name)

    preprocess_output = join(run_dir,
                             f"preprocess_3D-2D_"
                             f"{preprocess.projection_operator}")
    os.makedirs(preprocess_output, exist_ok=True)

    raw_files = load_images(input_path, preprocess.file_extension)

    preprocessed = []
    for file in raw_files:
        preprocessed.append(
            load_and_preprocess_3D_to_2D.submit(
                file=file,
                ext=preprocess.file_extension,
                projection_operator=preprocess.projection_operator,
                out_dir=preprocess_output,
            )
        )

    preprocessed = [p.result() for p in preprocessed]

    # Deepblink runs in GPU TensorFlow env
    spots = run_deepblink.submit(
        image_dicts=[p.serialize() for p in preprocessed],
        output_path=output_path,
        run_name=run_name,
        detection_channels=detection_channels,
        deepblink_models=deepblink_models,
    )

    cell_segmentations = run_cell_segmentation.submit(
        serialized_images=[p.serialize() for p in preprocessed],
        output_dir=os.path.join(output_path, run_dir),
        segment_nuclei=segment_nuclei,
        segment_cyto=segment_cyto,
    )

    other_segmentations = run_other_segmentation.submit(
        serialized_images=[p.serialize() for p in preprocessed],
        output_dir=join(output_path, run_name),
        segment_other=segment_other,
    )

    merge(spots, cell_segmentations, other_segmentations)
