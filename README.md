# Koopa-flows

Workflow logic to integrate core python code from [koopa](https://github.com/fmi-basel/gchao-koopa) and wrap it with a small CLI and prefect web interface.


## Initial deployment
Install dependencies:
```
pip install koopa prefect prefect-cloud distro dask-jobqueue
```

Build deployment(s):
```
prefect deployment build --tag "VERSION" ./src/flows.py:workflow --name "koopa"
prefect deployment build --tag "VERSION" ./src/flows.py:gpu_workflow --name "koopa-gpu"
prefect deployment build --tag "VERSION" ./src/flows.py:update_koopa --name "koopa-update"
```

Upload deployments to cloud/on-prem (potentially after logging in with `prefect cloud login -k KEY`).
```
prefect deployment apply workflow-deployment.yaml
prefect deployment apply gpu_workflow-deployment.yaml
prefect deployment apply update_koopa-deployment.yaml
```


## CLI usage
Create default configuration file with full descriptions to each parameter. Important to save/update this `koopa.cfg` file in the root of this directory (for users to find default):
```
python src/cli.py --create-config
```
Run flow from command line by passing the path to the config file. Will use the default `task_runner` and ignore the saved SLURM parameters. Useful for debugging of the flow without wating for cluster resources.
```
python src/cli.py --config PATH
```

## TODO
* GPU support / slurm scheduling
	* images w/ dask runner & dask jobqueue
	* https://prefecthq.github.io/prefect-dask/#dask-annotations