import os


def test_output_files(config):
    """Example pipeline for live cell."""
    files = [
        "alignment.npy",
        "alignment_post.tif",
        "alignment_pre.tif",
        "colocalization_0-1/20220518_18xsm_2.parq",
        "detection_final_c0/20220518_18xsm_2.parq",
        "detection_raw_c0/20220518_18xsm_2.parq",
        "koopa.cfg",
        "preprocessed/20220518_18xsm_2.tif",
        "segmentation_cyto/20220518_18xsm_2.tif",
        "summary.csv",
    ]

    for fname in files:
        assert os.path.exists(os.path.join(config["live"], fname))


def test_output_summary(config):
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,coloc_particle,cell_id,num_cells,area_cyto,eccentricity_cyto"
    with open(os.path.join(config["live"], "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line
