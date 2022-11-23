# Koopa-flows

Workflow logic to integrate core python code from [koopa](https://github.com/bbquercus/koopa) and wrap it with a small CLI and prefect web interface.

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