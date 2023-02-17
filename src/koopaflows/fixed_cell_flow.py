import os
from glob import glob
from pathlib import Path
from typing import Union, Literal, List

import pandas as pd
from cpr.Serializer import cpr_serializer
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.utilities.utilities import task_input_hash
from koopa.postprocess import get_segmentation_data
from prefect import flow, get_client
from prefect import task
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment

from .cpr_parquet import ParquetSource
from .merge.merge_tasks import merge
from .preprocessing.preprocess import preprocess_3D_to_2D
from .segmentation.other_threshold_segmentation_flow import \
    other_threshold_segmentation_flow, SegmentOther
from .segmentation.threshold_segmentation_flow import SegmentNuclei, \
    SegmentCyto, threshold_segmentation_flow


@task(
    cache_result_in_memory=False,
    persist_result=True,
    cache_key_fn=task_input_hash,
)
def run_deepblink(
        image_dicts: list[dict],
        output_path: str,
        detection_channels: list[int],
        deepblink_models: list[Path]
):
    parameters = {
        "serialized_preprocessed": image_dicts,
        "out_dir": output_path,
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
def load_images(input_path, ext):
    assert ext in ['.tif', '.stk', '.nd', '.czi'], 'File format not supported.'
    files = glob(os.paht.join(input_path, "*" + ext))
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
    result_serializer=cpr_serializer(),
)
def fixed_cell_flow(
        input_path: Union[Path, str] = "/tungstenfs/scratch/gchao/grieesth/Export_DRB/20221216_HeLa11ht-pIM40nuc-JunD-2_HS-42C-30or1h_DRB-4h_washout-30min-1h-2h_smFISH-IF_HSPH1_SC35/",
        output_path: Union[Path, str] = "/path/to/output/dir",
        file_ext: str = ".nd",
        projection_operator: Literal[
            "maximum", "mean", "sharpest"] = "maximum",
        deepblink_models: List[Union[Path, str]] = [
            "/tungstenfs/scratch/gchao/deepblink/model_fish.h5"],
        detection_channels: List[int] = [1],
        segment_nuclei: SegmentNuclei = SegmentNuclei(),
        segment_cyto: SegmentCyto = SegmentCyto(),
        segment_other: SegmentOther = SegmentOther(),
):
    # TODO: double-check output path
    raw_images = load_images(input_path, ext=file_ext)

    preprocessed = []
    for img in raw_images:
        preprocessed.append(preprocess_3D_to_2D(img, projection_operator,
                                                os.path.join(output_path,
                                                             "preprocessed")))

    # Deepblink runs in GPU TensorFlow env
    spots = run_deepblink.submit(
        image_dicts=[p.serialize() for p in preprocessed],
        output_path=os.path.join(output_path, "spots"),
        detection_channels=detection_channels,
        deepblink_models=deepblink_models,
    )

    segmentations = threshold_segmentation_flow(
        images=preprocessed,
        output_dir=output_path,
        segment_nuclei=segment_nuclei,
        segment_cyto=segment_cyto,
    )

    other_segmentations = other_threshold_segmentation_flow(
        images=preprocessed,
        output_dir=output_path,
        segment_other=segment_other,
    )

    merge(spots, segmentations, other_segmentations)
