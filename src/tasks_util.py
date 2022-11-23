import json
import os
import subprocess

from distro import distro
from prefect import get_run_logger
from prefect import task
import koopa


@task()
def save_conda_env(path_out: str):
    """Save conda environment to conda-environment.yaml."""
    logger = get_run_logger()
    conda_prefix = os.environ["CONDA_PREFIX"]
    outpath = os.path.join(path_out, "conda-environment.yaml")
    cmd = f"conda list -p {conda_prefix} > {outpath}"
    result = subprocess.run(cmd, shell=True, check=True)
    result.check_returncode()

    logger.info(f"Saved conda-environment to {outpath}.")


@task()
def save_system_information(path_out: str):
    """Dump system information into system-info.json."""
    logger = get_run_logger()
    outpath = os.path.join(path_out, "system-info.json")
    info = distro.info(pretty=True, best=True)
    with open(outpath, "w") as f:
        json.dump(info, f, indent=4)

    logger.info(f"Saved system information to {outpath}.")


@task(name="Configuration")
def configuration(path: os.PathLike, force: bool):
    logger = get_run_logger()

    # Parse configuration
    cfg = koopa.io.load_config(path)
    koopa.config.validate_config(cfg)
    logger.info("Configuration file validated.")
    config = koopa.config.flatten_config(cfg)

    # Save config
    cfg = koopa.config.add_versioning(cfg)
    fname_config = os.path.join(config["output_path"], "koopa.cfg")
    koopa.io.save_config(fname_config, cfg)

    config["force"] = force
    return config
