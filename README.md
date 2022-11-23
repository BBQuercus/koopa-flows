# Koopa-prefect

Workflow logic to integrate core python code from `koopa` and wrap it with a small CLI and prefect web interface.

### TODO
* Add GitHub actions hook for automatic deployment on code change
* Access by non-technical users with group login
* GPU support / slurm scheduling
	* images w/ dask runner & dask jobqueue
	* https://prefecthq.github.io/prefect-dask/#dask-annotations