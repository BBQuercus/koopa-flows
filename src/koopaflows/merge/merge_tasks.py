import os
from pathlib import Path
from typing import List

import pandas as pd
from cpr.csv.CSVTarget import CSVTarget
from cpr.image.ImageSource import ImageSource
from cpr.utilities.utilities import task_input_hash
from koopa.postprocess import get_segmentation_data
from prefect import task

from src.flow.cpr_parquet import ParquetSource


@task(cache_key_fn=task_input_hash)
def merge(
    coloc_spots: List[List[ParquetSource]], # for each image a list of
        # coloc´d spots
    segmentations: List[List[ImageSource]], # for each image [nuc, cyto]
    output_path: Path,
) -> tuple[CSVTarget, CSVTarget]:

    result_dfs, result_cell_dfs = [], []
    for coloc, (nuc, cyto) in zip(coloc_spots, segmentations):
        dfs = []
        for src in coloc:
            dfs.append(src.get_data())

        df = pd.concat(dfs)

        df, cell_df = get_segmentation_data(
            df,
            {
                "nuclei": nuc,
                "cyto": cyto,
            },
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