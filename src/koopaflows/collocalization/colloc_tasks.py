import os
from pathlib import Path

import koopa
from prefect import task

from src.flow.cpr_parquet import ParquetSource, ParquetTarget
from scratchpad.fixed_cell_flow_complicated import ignore_suffix_task_input_hash
from src.flow.flow_parameters import Colocalize


@task(cache_key_fn=ignore_suffix_task_input_hash)
def colocalize_spots(
    spot: dict[int, ParquetSource], colocalize: Colocalize, output_path:
        Path, suffix: str,
) -> list[ParquetTarget]:
    results = []
    for chs in colocalize.channels:
        df_1 = spot[chs[0]].get_data()
        df_2 = spot[chs[1]].get_data()
        name = f"{chs[0]}_{chs[1]}"
        df = koopa.colocalize.colocalize_frames(
            df_one=df_1,
            df_two=df_2,
            name=name,
            z_distance=1,
            distance_cutoff=colocalize.distance_cutoff,
        )

        res = ParquetTarget.from_path(os.path.join(output_path,
                                                   f"colocalization_{name}",
                                                   spot[chs[0]].get_name()))
        res.set_data(df)
        results.append(res)

    return results