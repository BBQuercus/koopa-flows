import os
import subprocess


def test_pipeline_config():
    """Create default config file."""
    if os.path.exists("./koopa.cfg"):
        os.remove("./koopa.cfg")
    subprocess.run(["koopa", "--create-config"])
    assert os.path.exists("./koopa.cfg")


def test_pipeline_helptext():
    """Help only with and without being explicit."""
    process_1 = subprocess.check_output("koopa").decode()
    process_2 = subprocess.check_output(["koopa", "--help"]).decode()
    assert process_1 == process_2
