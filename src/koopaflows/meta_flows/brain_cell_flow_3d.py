import json
import os
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Union, Literal, Any, Optional

import koopa.segment_flies
import koopa.colocalize
import koopa.track
import koopa.postprocess
import numpy as np
import pandas as pd
import pkg_resources
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.image.ImageTarget import ImageTarget
from cpr.utilities.utilities import task_input_hash
from faim_prefect.prefect import get_prefect_context
from koopaflows.cpr_parquet import koopa_serializer, \
    ParquetTarget
from koopaflows.preprocessing.flow import load_images
from koopaflows.preprocessing.task import load_and_preprocess_brains
from koopaflows.storage_key import RESULT_STORAGE_KEY
from koopaflows.utils import wait_for_task_runs
from prefect import flow, get_client, get_run_logger
from prefect import task
from prefect.client.schemas import FlowRun
from prefect.context import get_run_context
from prefect.deployments import run_deployment
from prefect.filesystems import LocalFileSystem
from prefect.futures import PrefectFuture
from pydantic import BaseModel


class SegmentNuclei(BaseModel):
    cellpose_model: str
    brain_channel: int
    min_intensity: int
    min_area: int
    max_area: int
    dilation: int



class SpotDetection(BaseModel):
    deepblink_models: list[str]
    detection_channels: list[int]
    search_range: int
    gap_frames: int
    min_length: int


class Colocalization(BaseModel):
    active: bool = True
    coloc_channels: list[list[int]]
    z_distance = 2
    distance_cutoff = 5


@task(
    cache_key_fn=task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY
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
def process_nuclei_labels(
    image: ImageTarget,
    labeling: ImageTarget,
    nuc_channel: int,
    min_intensity: int,
    min_area: int,
    max_area: int,
    dilation: int,
    output_dir: str,
):
    seg_map = koopa.segment_flies.remove_false_objects(
        image=image.get_data()[nuc_channel],
        segmap=labeling.get_data()[0],
        min_intensity=min_intensity,
        min_area=min_area,
        max_area=max_area,
    )

    seg_map = koopa.segment_flies.dilate_segmap(
        segmap=seg_map,
        dilation=dilation,
    )

    clean_segmentation: ImageTarget = ImageTarget.from_path(
        join(
            output_dir,
            labeling.get_name() + labeling.ext
        ),
        imagej=False,
    )
    clean_segmentation.set_metadata(labeling.get_metadata())
    clean_segmentation.set_resolution(labeling.get_resolution())
    clean_segmentation.set_data(seg_map[np.newaxis])

    return clean_segmentation


@task(
    cache_key_fn=task_input_hash,
    result_storage_key=RESULT_STORAGE_KEY
)
def run_cellpose(
    image_dicts: list[dict],
    cellpose_parameter: dict,
    output_format: dict,
):
    run: FlowRun = run_deployment(
        name="Run cellpose inference/default",
        parameters={
            "image_dicts": image_dicts,
            "cellpose_parameter": cellpose_parameter,
            "output_format": output_format,
        },
        client=get_client(),
    )

    return run.state.result()



def cell_segmentation(
    images: list[ImageTarget],
    cellpose_model: str,
    brains_channel: int,
    min_intensity: int,
    min_area: int,
    max_area: int,
    dilation: int,
    output_dir: str,
):

    image_dicts = [img.serialize() for img in images]

    cellpose_parameter = {
        "model": cellpose_model,
        "seg_channel": brains_channel,
        "nuclei_channel": brains_channel,
        "diameter": 30,
        "flow_threshold": 0.4,
        "cell_probability_threshold": 0.0,
        "resample": True,
        "save_labeling": True,
        "save_flows": False,
        "do_3D": True,
    }

    os.makedirs(join(output_dir, "segmentation_nuclei"), exist_ok=True)
    output_format = {
        "output_dir": join(output_dir, "segmentation_nuclei"),
        "imagej_compatible": False,
    }

    labels = run_cellpose.submit(
        image_dicts=image_dicts,
        cellpose_parameter=cellpose_parameter,
        output_format=output_format,
        wait_for=[images]
    )

    os.makedirs(join(output_dir, "segmentation_cyto"), exist_ok=True)
    nuclei_segmentations = []
    buffer = []
    for image, labeling in zip(images, labels.result()):
        buffer.append(
            process_nuclei_labels.submit(
                image=image,
                labeling=labeling['mask'],
                nuc_channel=brains_channel,
                min_intensity=min_intensity,
                min_area=min_area,
                max_area=max_area,
                dilation=dilation,
                output_dir=join(output_dir, "segmentation_cyto"),
                wait_for=[labels],
            )
        )

        wait_for_task_runs(
            results=nuclei_segmentations,
            buffer=buffer,
            max_buffer_length=2,
        )


    wait_for_task_runs(
        results=nuclei_segmentations,
        buffer=buffer,
        max_buffer_length=0,
    )

    return nuclei_segmentations


class Preprocess3D(BaseModel):
    file_extension: Literal["tif", "stk", "nd", "czi"] = "nd"
    crop_start: int
    crop_end: int
    bin_axes: list[float]


def preprocessing(
    raw_files: list[ImageSource],
    output_dir: str,
    preprocess: Preprocess3D,
):
    preprocess_output = join(output_dir,
                             "preprocessed")
    os.makedirs(preprocess_output, exist_ok=True)

    preprocessed = []
    buffer = []
    def obtain_result(future: PrefectFuture):
        result = future.result(raise_on_failure=False)
        if future.get_state().is_completed():
            return result
        else:
            return None

    for file in raw_files:
        buffer.append(
            load_and_preprocess_brains.submit(
                file=file,
                ext=preprocess.file_extension,
                crop_start=preprocess.crop_start,
                crop_end=preprocess.crop_end,
                scale_factors=1/np.array(preprocess.bin_axes),
                out_dir=preprocess_output,
            )
        )

        wait_for_task_runs(
            results=preprocessed,
            buffer=buffer,
            max_buffer_length=1,
            result_insert_fn=obtain_result
        )

    wait_for_task_runs(
        results=preprocessed,
        buffer=buffer,
        max_buffer_length=0,
        result_insert_fn=obtain_result
    )

    return list(filter(None, preprocessed))


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def non_maxima_suppression(
    raw_spots: dict[int, ParquetTarget],
    output_dir: str,
    search_range: int,
    gap_frames: int,
    min_length: int
):
    tracks = {}
    for ch, rs in raw_spots.items():
        track = koopa.track.link_brightest_particles(
            df=rs.get_data(),
            track=koopa.track.track(
                rs.get_data(),
                search_range=search_range,
                gap_frames=gap_frames,
                min_length=min_length,
            ),
        )

        track = koopa.track.clean_particles(track)

        fname_out = join(
            output_dir, f"detection_final_c{ch}",
            f"{rs.get_name()}.parq"
        )
        output = ParquetTarget.from_path(fname_out)
        output.set_data(track)
        tracks[ch] = output

    return tracks

@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def colocalize(
        all_spots: dict[int, ParquetTarget],
          output_path,
          coloc_channels,
          z_distance,
          distance_cutoff,
    ):
    logger = get_run_logger()
    logger.debug(all_spots)
    logger.debug(coloc_channels)

    file_name = all_spots[str(coloc_channels[0])].get_name()
    name = f"{coloc_channels[0]}-{coloc_channels[1]}"
    df_reference = all_spots[str(coloc_channels[0])].get_data()
    df_transform = all_spots[str(coloc_channels[1])].get_data()
    df = koopa.colocalize.colocalize_frames(
        df_one=df_reference,
        df_two=df_transform,
        name=name,
        z_distance=z_distance,
        distance_cutoff=distance_cutoff,
    )

    coloc_result = ParquetTarget.from_path(join(output_path,
                                                f"{file_name}.parq"))

    coloc_result.set_data(df)

    return coloc_result


@task(cache_key_fn=task_input_hash, result_storage_key=RESULT_STORAGE_KEY)
def merge(segmentations, all_colocs, output_path):
    colocs_per_file = [[coloc] for coloc in all_colocs[0]]
    for ac in all_colocs[1:]:
        for i, coloc in enumerate(ac):
            colocs_per_file[i].append(coloc)

    get_run_logger().debug(colocs_per_file)
    dfs, dfs_cell = [], []
    get_run_logger().debug(f"{len(colocs_per_file)} number of files.")
    for colocs, nuc_segs in zip(colocs_per_file, segmentations):
        segmaps = {
            "nuclei": nuc_segs.get_data()[0],
        }
        df = pd.concat([coloc.get_data() for coloc in colocs])

        try:
            df, df_cell = koopa.postprocess.get_segmentation_data(
                df, segmaps, {"brains_enabled": True, "do_3d": True}
            )
            dfs.append(df)
            dfs_cell.append(df_cell)
            get_run_logger().debug(f"Merged files for {colocs[0].get_name()}")
        except ValueError as e:
            get_run_logger().debug(e)
            get_run_logger().info(f"No spots found for "
                                  f"{colocs[0].get_name()}.")

    summary = CSVTarget.from_path(join(output_path, "summary.csv"))
    summary.set_data(pd.concat(dfs, ignore_index=True))
    summary_cell = CSVTarget.from_path(join(output_path, "summary_cell.csv"))
    summary_cell.set_data(pd.concat(dfs_cell, ignore_index=True))

    return summary, summary_cell


def exlude_context_task_input_hash(
    context: "TaskRunContext", arguments: dict[str, Any]
) -> Optional[str]:
    hash_args = {}
    for k, item in arguments.items():
        if k not in ["context"]:
            hash_args[k] = item

    return task_input_hash(context, hash_args)


@task(cache_key_fn=exlude_context_task_input_hash)
def write_info_md(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    run_name: str,
    preprocess: Preprocess3D,
    spot_detection: SpotDetection,
    segment_nuclei: SegmentNuclei,
    coloc_conf: Colocalization,
    context: dict,
):
    date = datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    params = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "run_name": str(run_name),
        "preprocess": preprocess.dict(),
        "spot_detection": spot_detection.dict(),
        "segment_nuclei": segment_nuclei.dict(),
        "coloc_conf": coloc_conf.dict(),
    }
    gchao_koopa_flows_v = pkg_resources.get_distribution("koopa-flows").version
    gchao_koopa_v = pkg_resources.get_distribution("koopa").version
    prefect_v = pkg_resources.get_distribution("prefect").version
    content = "# Fly Brain Cell Analysis 3D\n" \
              "Source: [https://github.com/fmi-basel/gchao-koopa-flows](" \
              "https://github.com/fmi-basel/gchao-koopa-flows)\n" \
              f"Date: {date}\n" \
              "\n" \
              "## Parameters\n" \
              f"{json.dumps(params, indent=4)}\n" \
              "\n" \
              "## Packages\n" \
              f"* gchao-koopa-flows: {gchao_koopa_flows_v}\n" \
              f"* gchao-koopa: {gchao_koopa_v}\n" \
              f"* prefect: {prefect_v}\n" \
              "\n" \
              "## Prefect Context\n" \
              f"{str(context)}\n"

    with open(join(output_path, run_name, "README.md"), "w") as f:
        f.write(content)



@task(cache_key_fn=task_input_hash)
def write_koopa_cfg(
    path: str,
    detection_channels: list[int],
    coloc_active: bool,
    coloc_channels: list[list[int]],
):
    det_channels = [str(c) for c in detection_channels]
    content = "[General]\n" \
              "do_timeseries = False\n" \
              "do_3d = True\n" \
              "\n" \
              "[SegmentationOther]\n" \
              "sego_enabled = False\n" \
              "sego_channels = []\n" \
              "\n" \
              "[SpotsDetection]\n" \
              f"detect_channels = [{','.join(det_channels)}]\n" \
              "refinement_radius = 3\n" \
              "\n" \
              "[SpotsColocalization]\n" \
              f"coloc_enabled = {coloc_active}\n" \
              f"coloc_channels = {coloc_channels}\n"

    with open(join(path, "koopa.cfg"), "w") as f:
        f.write(content)


@flow(
    name="Fly Brain Cell Analysis 3D",
    cache_result_in_memory=False,
    persist_result=True,
    result_serializer=koopa_serializer(),
    result_storage=LocalFileSystem.load("koopa"),
)
def fly_brain_cell_analysis_3D(
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        run_name: str,
        preprocess: Preprocess3D,
        spot_detection: SpotDetection,
        segment_nuclei: SegmentNuclei,
        coloc_conf: Colocalization,
):
    raw_files = load_images(input_path, preprocess.file_extension)

    preprocessed = preprocessing(
        raw_files=raw_files,
        output_dir=join(output_path, run_name),
        preprocess=preprocess
    )

    # Deepblink runs in GPU TensorFlow env
    raw_spots: FlowRun = run_deepblink.submit(
        image_dicts=[p.serialize() for p in preprocessed],
        output_path=output_path,
        run_name=run_name,
        detection_channels=spot_detection.detection_channels,
        deepblink_models=spot_detection.deepblink_models,
        wait_for=[preprocessed],
    )

    nuclei_segmentations = cell_segmentation(
        images=preprocessed,
        cellpose_model=segment_nuclei.cellpose_model,
        brains_channel=segment_nuclei.brain_channel,
        min_intensity=segment_nuclei.min_intensity,
        min_area=segment_nuclei.min_area,
        max_area=segment_nuclei.max_area,
        dilation=segment_nuclei.dilation,
        output_dir=os.path.join(output_path, run_name),
    )

    final_spots = []
    buffer = []
    for spots_per_channels in raw_spots.result():
        buffer.append(
            non_maxima_suppression.submit(
                raw_spots=spots_per_channels,
                output_dir=os.path.join(output_path, run_name),
                search_range=spot_detection.search_range,
                gap_frames=spot_detection.gap_frames,
                min_length=spot_detection.min_length,
            )
        )

        wait_for_task_runs(
            results=final_spots,
            buffer=buffer,
            max_buffer_length=6,
        )

    wait_for_task_runs(
        results=final_spots,
        buffer=buffer,
        max_buffer_length=0
    )

    if coloc_conf.active:
        all_colocs = []
        for c_source, c_target in coloc_conf.coloc_channels:
            colocs = []
            buffer = []
            for spots_per_channels in final_spots:
                buffer.append(
                    colocalize.submit(
                        all_spots=spots_per_channels,
                        output_path=os.path.join(output_path, run_name,
                                                 f"colocalization_{c_source}-"
                                                 f"{c_target}"),
                        coloc_channels=[c_source, c_target],
                        z_distance=coloc_conf.z_distance,
                        distance_cutoff=coloc_conf.distance_cutoff
                    )
                )

                wait_for_task_runs(
                    results=colocs,
                    buffer=buffer,
                    max_buffer_length=12,
                )

            wait_for_task_runs(
                results=colocs,
                buffer=buffer,
                max_buffer_length=0,
            )
            all_colocs.append(colocs)
    else:
        spots_per_channel = final_spots[0]
        all_colocs = [[spots] for spots in spots_per_channel.values()]
        for spots_per_channel in final_spots[1:]:
            for i, spots in enumerate(spots_per_channel.values()):
                all_colocs[i].append(spots)

    merge(
        segmentations=nuclei_segmentations,
        all_colocs=all_colocs,
        output_path=join(output_path, run_name)
    )

    write_koopa_cfg(
        path=join(output_path, run_name),
        detection_channels=spot_detection.detection_channels,
        coloc_active=coloc_conf.active,
        coloc_channels=coloc_conf.coloc_channels,
    )

    write_info_md(
        input_path=input_path,
        output_path=output_path,
        run_name=run_name,
        preprocess=preprocess,
        spot_detection=spot_detection,
        segment_nuclei=segment_nuclei,
        coloc_conf=coloc_conf,
        context=get_prefect_context(get_run_context()),
    )
