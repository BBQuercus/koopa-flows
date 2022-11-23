import os
import subprocess


def test_pipeline_config():
    """Create default config file."""
    path = os.path.dirname(os.path.abspath(__file__))
    cli = os.path.join(path, "..", "src", "cli.py")

    if os.path.exists("./koopa.cfg"):
        os.remove("./koopa.cfg")
    subprocess.run(["python", cli, "--create-config"])
    assert os.path.exists("./koopa.cfg")
    os.remove("./koopa.cfg")
