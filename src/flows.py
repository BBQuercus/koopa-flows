import json
import os
import subprocess
from os.path import join
from typing import List

from distro import distro
from prefect import flow, get_run_logger, unmapped, task
from prefect.task_runners import SequentialTaskRunner
from prefect_dask import DaskTaskRunner
import koopa

import tasks_postprocess
import tasks_preprocess
import tasks_segment
import tasks_spots


# cluster_class="dask_jobqueue.SLURMCluster",
# cluster_kwargs={
#     "n_workers": 1,
#     "account": "ppi",
#     "queue": "cpu_short",
#     "cores": 4,
#     "memory": "16GB",
#     "walltime": "00:30:00",
#     "job_extra_directives": ["--ntasks=1"],
# },
# adapt_kwargs={"minimum": 1, "maximum": 1},

@task()
def save_conda_env(output_dir: str):
    """
    Save conda environment to conda-environment.yaml.
    :param output_dir:
    :param logger:
    :return:
    """
    logger = get_run_logger()
    conda_prefix = os.environ["CONDA_PREFIX"]
    outpath = join(output_dir, "conda-environment.yaml")
    cmd = f"conda list -p {conda_prefix} > {outpath}"
    result = subprocess.run(cmd, shell=True, check=True)
    result.check_returncode()

    logger.info(f"Saved conda-environment to {outpath}.")

@task()
def save_system_information(output_dir: str):
    """
    Dump system information into system-info.json.
    :param output_dir:
    :param logger:
    :return:
    """
    logger = get_run_logger()
    outpath = join(output_dir, "system-info.json")
    info = distro.info(pretty=True, best=True)
    with open(outpath, "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"Saved system information to {outpath}.")


def file_independent(config: dict):
    if not config["alignment_enabled"]:
        return None

    tasks_preprocess.align.submit(
        path_in=config["alignment_path"], path_out=config["output_path"], config=config
    ).wait()


def cell_segmentation(
    fnames: List[str], config: dict, kwargs: dict, dependencies: list
):
    if not config["brains_enabled"]:
        if config["selection"] == "both":
            return tasks_segment.segment_cells_both.map(
                fnames, **kwargs, wait_for=dependencies
            )
        return tasks_segment.segment_cells_single.map(
            fnames, **kwargs, wait_for=dependencies
        )

    brain_1 = tasks_segment.segment_cells_predict.map(
        fnames, **kwargs, wait_for=dependencies
    )
    brain_2 = tasks_segment.segment_cells_merge.map(fnames, **kwargs, wait_for=brain_1)
    return tasks_segment.dilate_cells.map(fnames, **kwargs, wait_for=brain_2)


def other_segmentation(
    fnames: List[str], config: dict, kwargs: dict, dependencies: list
):
    if not config["sego_enabled"]:
        return []

    channels = range(len(config["sego_channels"]))
    fnames_map = [f for f in fnames for _ in channels]
    index_map = [c for _ in fnames for c in channels]

    seg_other = tasks_segment.segment_other.map(
        fnames_map, **kwargs, index_list=index_map, wait_for=dependencies
    )
    return seg_other


def spot_detection(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    channels = range(len(config["detect_channels"]))
    fnames_map = [f for f in fnames for _ in channels]
    index_map = [c for _ in fnames for c in channels]
    spots = tasks_spots.detect.map(
        fnames_map, **kwargs, index_list=index_map, wait_for=dependencies
    )

    if config["do_3d"] or config["do_timeseries"]:
        fnames_map = [f for f in fnames for _ in config["detect_channels"]]
        index_map = [c for _ in fnames for c in config["detect_channels"]]
        spots = tasks_spots.track.map(
            fnames_map, **kwargs, index_channel=index_map, wait_for=spots
        )
    return spots


def colocalization(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    if not config["coloc_enabled"]:
        return []

    reference = [i[0] for _ in fnames for i in config["coloc_channels"]]
    transform = [i[1] for _ in fnames for i in config["coloc_channels"]]
    fnames_map = [f for f in fnames for _ in config["coloc_channels"]]

    if config["do_timeseries"]:
        return tasks_spots.colocalize_track.map(
            fnames_map,
            **kwargs,
            index_reference=reference,
            index_transform=transform,
            wait_for=dependencies
        )

    return tasks_spots.colocalize_frame.map(
        fnames_map,
        **kwargs,
        index_reference=reference,
        index_transform=transform,
        wait_for=dependencies
    )


def merging(fnames: List[str], config: dict, kwargs: dict, dependencies: list):
    dfs = tasks_postprocess.merge_single.map(fnames, **kwargs, wait_for=dependencies)
    tasks_postprocess.merge_all.submit(config["output_path"], dfs, wait_for=dfs)


@flow(name="Koopa",
      version=koopa.__version__,
      task_runner=DaskTaskRunner(cluster_class="dask_jobqueue.SLURMCluster",
        cluster_kwargs={
            "account": "dlthings",
            "queue": "cpu_short",
            "cores": 4,
            "processes": 1,
            "memory": "16 GB",
            "walltime": "04:00:00",
            "job_extra_directives": [
                "--gpus-per-node=1",
                "--ntasks=1",
                "--output=/tungstenfs/scratch/gmicro_share/_prefect/slurm/output/%j.out",
            ],
            "worker_extra_args": [
                "--lifetime",
                "240m",
                "--lifetime-stagger",
                "15m",
            ],
            "job_script_prologue": [
                "conda run -p /tungstenfs/scratch/gmicro_share/_prefect/miniconda3/envs/airtable python /tungstenfs/scratch/gmicro_share/_prefect/airtable/log-slurm-job.py --config /tungstenfs/scratch/gmicro/_prefect/airtable/slurm-job-log.ini"
            ],
        },
        adapt_kwargs={
            "minimum": 1,
            "maximum": 8,
        },))
def workflow(config_path: str, force: bool = False):
    """Core koopa workflow.

    Arguments:
        * config_path: Path to koopa configuration file.
            Path must be passed linux-compatible (e.g. /tungstenfs/scratch/...).
            The default configuration file can be viewed and downloaded [here](https://github.com/BBQuercus/koopa/blob/main/koopa-prefect/koopa.cfg).

        * force: If selected, the entire workflow will be re-run.
            Otherwise, only the not yet executed components (missing files) are run.

    All documentation can be found on the koopa wiki (https://github.com/BBQuercus/koopa/wiki).
    """
    logger = get_run_logger()
    logger.info("Started running Koopa!")
    koopa.util.configure_gpu(False)

    # File independent tasks
    config = tasks_preprocess.configuration(config_path, force)
    file_independent(config)

    # Workflow
    fnames = koopa.util.get_file_list(config["input_path"], config["file_ext"])
    kwargs = dict(path=unmapped(config["output_path"]), config=unmapped(config))

    # Preprocess
    preprocess = tasks_preprocess.preprocess.map(fnames, **kwargs)

    # Segmentation
    seg_cells = cell_segmentation(fnames, config, kwargs, dependencies=preprocess)
    seg_other = other_segmentation(fnames, config, kwargs, dependencies=preprocess)

    # Spots
    spots = spot_detection(fnames, config, kwargs, dependencies=preprocess)
    coloc = colocalization(fnames, config, kwargs, dependencies=spots)

    # Merge
    merging(
        fnames, config, kwargs, dependencies=[*spots, *coloc, *seg_cells, *seg_other]
    )
    logger.info("Koopa finished analyzing everything!")
