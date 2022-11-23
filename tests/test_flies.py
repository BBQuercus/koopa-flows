import os


def test_output_files(config):
    """Example pipeline for Jess's flies."""
    files = [
        "colocalization_0-1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_final_c1/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "detection_raw_c0/hr38-24xPP7_hr38633_PP7546_OCT_9.parq",
        "koopa.cfg",
        "preprocessed/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
        "segmentation_nuclei/hr38-24xPP7_hr38633_PP7546_OCT_9.tif",
        "summary.csv",
    ]

    for fname in files:
        assert os.path.exists(os.path.join(config["flies"], fname))


def test_output_summary(config):
    columns = "FileID,y,x,mass,size,eccentricity,signal,frame,channel,particle,particle_0-1,coloc_particle_0-1,cell_id,num_cells,area_nuclei,other_c1"
    with open(os.path.join(config["flies"], "summary.csv"), "r") as f:
        first_line = f.readline().strip()
    assert columns == first_line
