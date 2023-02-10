# Build
```shell
prefect deployment build src/flow/deepblink_flow.py:deepblink_spot_detection_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/deepblink.yaml -ib process/deepblink

prefect deployment build src/flow/deepblink_flow.py:deepblink_spot_detection_flow -n "default" -q slurm -sb github/koopa-flows --skip-upload -o deployment/deepblink.yaml -ib process/deepblink
```

# Apply
```shell
prefect deployment apply deployment/*.yaml
```