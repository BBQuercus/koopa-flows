import os


def test_output_files(config):
    """Example pipeline basic 2D cellular data."""
    files = [
        "conda-environment.yaml",
        "detection_raw_c0/20220512_EGFP_3h_2.parq",
        "koopa.cfg",
        "preprocessed/20220512_EGFP_3h_2.tif",
        "segmentation_cyto/20220512_EGFP_3h_2.tif",
        "segmentation_nuclei/20220512_EGFP_3h_2.tif",
        "summary.csv",
        "summary_cells.csv",
        "system-info.json",
    ]
    for fname in files:
        assert os.path.exists(os.path.join(config["2d"], fname))


def test_output_summary(config):
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,cell_id,nuclear"

    with open(os.path.join(config["2d"], "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line


def test_output_summary_cells(config):
    columns = (
        "FileID,cell_id,area_cyto,eccentricity_cyto,area_nuclei,eccentricity_nuclei"
    )

    with open(os.path.join(config["2d"], "summary_cells.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line
