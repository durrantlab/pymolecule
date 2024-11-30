from pymolecule import Molecule


def test_load_4nss(path_4nss_pdb):
    mol = Molecule()
    mol.io.load_pdb_into(path_4nss_pdb)


def test_load_rel1(path_rel1_pdb):
    mol = Molecule()
    mol.io.load_pdb_into(path_rel1_pdb)


def test_load_rogfp2(path_rogfp2_pdb):
    mol = Molecule()
    mol.io.load_pdb_into(path_rogfp2_pdb, bonds_by_distance=True)
