"""
RDKit util functions.
"""
import random

import rdkit.Chem as rkc


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl

    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb

    rkrb.DisableLog("rdApp.error")


disable_rdkit_logging()


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)


def to_smiles(mol):
    """
    Converts a Mol object into a canonical SMILES string.
    :param mol: Mol object.
    :return: A SMILES string.
    """
    return rkc.MolToSmiles(mol, isomericSmiles=False)


def randomize_smiles(smile, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    mol = to_mol(smile)
    if not mol:
        return None
    if random_type == "unrestricted":
        return rkc.MolToSmiles(
            mol, canonical=False, doRandom=True, isomericSmiles=False
        )
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)
    raise ValueError(f"Type '{random_type}' is not valid")


def get_smiles_alphabet():
    """
    :return: SMiles Alphabet
    """
    numTable = rkc.rdchem.GetPeriodicTable()
    HARD_ATOM_LIMIT = 119
    ATOM_ALPHABET = [numTable.GetElementSymbol(i) for i in range(1, HARD_ATOM_LIMIT)]
    return list("#%)(+*-/.1032547698:=@[]\\cons") + ATOM_ALPHABET + ["se"]
