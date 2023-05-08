# Build
```shell
prefect deployment build src/koopaflows/spot_detection/deepblink_flow.py:deepblink_spot_detection_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/deepblink.yaml -ib process/deepblink

prefect deployment build src/koopaflows/spot_detection/spot_detection_flow.py:run_deepblink -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/deepblink_spot_detection.yaml -ib process/koopa-orchestration

prefect deployment build src/koopaflows/preprocessing/flow.py:preprocess_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/preprocess.yaml -ib process/koopa -t koopa -t preprocess

prefect deployment build src/koopaflows/segmentation/threshold_segmentation_flow.py:threshold_segmentation_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/cell-seg-2d.yaml -ib process/koopa -t segmentation

prefect deployment build src/koopaflows/segmentation/threshold_segmentation_flow.py:run_cell_seg_threshold_2d -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/run-cell-seg-2d.yaml -ib process/koopa-orchestration -t koopa -t segmentation

prefect deployment build src/koopaflows/segmentation/other_threshold_segmentation_flow.py:other_threshold_segmentation_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/other-seg-2d.yaml -ib process/koopa -t segmentation

prefect deployment build src/koopaflows/segmentation/other_threshold_segmentation_flow.py:run_other_threshold_segmentation -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/run-other-seg-2d.yaml -ib process/koopa-orchestration -t koopa -t segmentation

prefect deployment build src/koopaflows/meta_flows/fixed_cell_flow.py:fixed_cell_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/fixed_cell_flow.yaml -ib process/koopa -t koopa -t fish -t fish-if
```

## Segmentation Models
```shell
prefect deployment build src/koopaflows/segmentation/other_dl_segmentation_flow.py:other_segmentation_DL_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/other_segmentation_models.yaml -ib process/segmentation-models 
```

## 3D fly brain analysis
```shell
prefect deployment build src/koopaflows/meta_flows/brain_cell_flow_3d.py:fly_brain_cell_analysis_3D -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/fly_brain_cell_analysis_3D.yaml -ib process/koopa -t koopa -t 3D -t fish -t fish-if
```

# Apply
```shell
prefect deployment apply deployment/*.yaml
```