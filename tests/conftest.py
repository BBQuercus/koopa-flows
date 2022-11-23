import pytest
import koopa


def pytest_addoption(parser):
    parser.addoption("--config-2d", default=None)
    parser.addoption("--config-3d", default=None)
    parser.addoption("--config-flies", default=None)
    parser.addoption("--config-live", default=None)


@pytest.fixture(scope="session")
def config(pytestconfig):
    config = {}
    for name in ("2d", "3d", "flies", "live"):
        path = pytestconfig.getoption(f"config_{name}")
        if path is None:
            continue
        cfg = koopa.io.load_config(path)
        cfg = koopa.config.flatten_config(cfg)
        config[name] = cfg["output_path"]
    return config
