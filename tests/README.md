# Tests

To run the CLI based integration tests run `pytest` from the root of this repository.

For the prefect cloud based workflows:
1. Update the following parameters in the `.cfg` files in the `config` directory:
	* `input_path`: path to respective `test_in_NAME(_images)` data directory
	* `output_path`: currently empty directory that exists
	* `alignment_path`: path to the respective `test_in_NAME(_beads)` if `alignment_enabled` is set to `True`.
1. Pass the path to the out as `config_path` in the web interface & run the flows.
1. Test the output files using pytest substituting PATH with the path to the respective config file:
	```
	pytest\
		--config-2d PATH\
		--config-3d PATH\
		--config-flies PATH\
		--config-live PATH
	```
