import os

import pytest

from pymolecule import enable_logging

TEST_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="session", autouse=True)
def turn_on_logging():
    enable_logging(10)


@pytest.fixture
def path_4nss_pdb():
    return os.path.join(TEST_DIR, "files/pdbs/4nss.pdb")


@pytest.fixture
def path_rel1_pdb():
    return os.path.join(TEST_DIR, "files/pdbs/rel1.pdb")


@pytest.fixture
def path_rogfp2_pdb():
    return os.path.join(TEST_DIR, "files/pdbs/rogfp2.pdb")
