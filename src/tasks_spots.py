import os

from prefect import get_run_logger
from prefect import task
import deepblink as pink
import koopa


@task(name="Spots (Detect)")
def detect(fname: str, path: os.PathLike, index_list: int, config: dict):
    # Config
    index_channel = config["detect_channels"][index_list]
    fname_image = os.path.join(path, "preprocessed", f"{fname}.tif")
    fname_out = os.path.join(path, f"detection_raw_c{index_channel}", f"{fname}.parq")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    image = koopa.io.load_image(fname_image)
    model = pink.io.load_model(config["detect_models"][index_channel])

    # Run
    df = koopa.detect.detect_image(
        image, index_channel, model, config["refinement_radius"]
    )
    df.insert(loc=0, column="FileID", value=fname)

    # Save
    koopa.io.save_parquet(fname_out, df)


@task(name="Spots (Track)")
def track(fname: str, path: os.PathLike, index_channel: int, config: dict):
    # Config
    fname_spots = os.path.join(path, f"detection_raw_c{index_channel}", f"{fname}.parq")
    fname_out = os.path.join(path, f"detection_final_c{index_channel}", f"{fname}.parq")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    df = koopa.io.load_parquet(fname_spots)

    # Run
    track = koopa.track.track(
        df, config["search_range"], config["gap_frames"], config["min_length"]
    )
    if config["do_3d"]:
        track = koopa.track.link_brightest_particles(df, track)
    if config["subtract_drift"]:
        track = koopa.track.subtract_drift(track)
    track = koopa.track.clean_particles(track)

    # Save
    koopa.io.save_parquet(fname_out, track)


@task(name="Spots (Colocalize Frames)")
def colocalize_frame(
    fname: str,
    path: os.PathLike,
    index_reference: int,
    index_transform: int,
    config: dict,
):
    # Config
    logger = get_run_logger()
    logger.info(f"Colocalizing {index_reference}<-{index_transform}")
    folder = "final" if config["do_3d"] else "raw"
    fname_reference = os.path.join(
        path, f"detection_{folder}_c{index_reference}", f"{fname}.parq"
    )
    fname_transform = os.path.join(
        path, f"detection_{folder}_c{index_transform}", f"{fname}.parq"
    )
    name = f"{index_reference}-{index_transform}"
    fname_out = os.path.join(path, f"colocalization_{name}", f"{fname}.parq")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    df_reference = koopa.io.load_parquet(fname_reference)
    df_transform = koopa.io.load_parquet(fname_transform)

    # Run
    df = koopa.colocalize.colocalize_frames(
        df_reference,
        df_transform,
        name,
        config["z_distance"] if config["do_3d"] else 1,
        config["distance_cutoff"],
    )

    # Save
    koopa.io.save_parquet(fname_out, df)


@task(name="Spots (Colocalize Tracks)")
def colocalize_track(
    fname: str,
    path: os.PathLike,
    index_reference: int,
    index_transform: int,
    config: dict,
):
    # Config
    logger = get_run_logger()
    logger.info(f"Colocalizing {index_reference}<-{index_transform}")
    fname_reference = os.path.join(
        path, f"detection_final_c{index_reference}", f"{fname}.parq"
    )
    fname_transform = os.path.join(
        path, f"detection_final_c{index_transform}", f"{fname}.parq"
    )
    name = f"{index_reference}-{index_transform}"
    fname_out = os.path.join(path, f"colocalization_{name}", f"{fname}.parq")
    if not config["force"] and os.path.exists(fname_out):
        return

    # Input
    df_reference = koopa.io.load_parquet(fname_reference)
    df_transform = koopa.io.load_parquet(fname_transform)

    # Run
    df = koopa.colocalize.colocalize_tracks(
        df_reference, df_transform, config["min_frames"], config["distance_cutoff"]
    )

    # Save
    koopa.io.save_parquet(fname_out, df)
