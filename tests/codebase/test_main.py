import pytest

import pkg_resources
import tempfile

from codebase.config import Args, get_args
from codebase.main import main


def test_main():
    with tempfile.TemporaryDirectory() as tmpdirname:
        args = get_args(
            [
                "-o",
                tmpdirname,
                "--conf",
                pkg_resources.resource_filename('tests.resources', 'test.conf'),
                "-M",
                "max_epochs=3"
            ]
        )

        main(args)
