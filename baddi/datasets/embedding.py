import pickle
import pandas as pd
import os
from deepchem.feat import SmilesToSeq
from deepchem.data import NumpyDataset
from deepchem.trans import FeaturizationTransformer
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset

PAD_TOKEN = "<pad>"
OUT_OF_VOCAB_TOKEN = "<unk>"
from baddi.models.trainer import SimCLR


def _load_embedding(path: str, model: str):
    # with open(model, "rb") as f:
    #    model = pickle.load(f)
    model = SimCLR.load_from_checkpoint(
        "/home/srkpa/scratch/BADDI/baddi/datasets/epoch=184-step=141339.ckpt"
    )
    model = model.eval()
    print(model)
    # _data = pd.read_table(path)
    _data = pd.read_csv("/home/srkpa/scratch/BADDI/data/chembl.csv", index_col=0)
    # print(_data.columns)
    # smiles = list(set(_data['X1'].values.tolist() + _data['X2'].values.tolist()))
    print(_data.columns)
    smiles = _data.SMILES.values.tolist()
    print(len(smiles))
    max_len = max(len(s) for s in smiles)

    char_set = set()
    for smile in smiles:
        if len(smile) <= max_len:
            char_set.update(set(smile))

    unique_char_list = list(char_set)
    unique_char_list += [PAD_TOKEN, OUT_OF_VOCAB_TOKEN]
    char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}

    root, _ = os.path.splitext(path)
    s = pd.DataFrame(smiles, columns=["SMILES"])
    s.to_csv(root + "_smiles.csv")

    add_args = dict(
        char_to_idx=char_to_idx,
        max_len=max_len,
        pad_len=0,
    )

    trans = SmilesToSeq(**add_args)
    smiles = pd.read_csv(path, index_col=0).smiles.values.tolist()
    dataset = NumpyDataset(X=np.array(smiles), y=np.arange(len(smiles)))
    _trans = FeaturizationTransformer(dataset, trans)
    dataset = _trans.transform(dataset)
    print(dataset.X.shape)
    x = dataset.X
    r = []
    taille = dataset.X.shape[0]
    for i in range(0, taille, 32):
        print("debut ", i)
        start, end = i, min(i + 32, taille)
        b = x[start:end]
        b = torch.from_numpy(b)
        pred = model.base_network(b)
        r += [pred]
        print("fin ", i, pred.shape)
    print("l", len(r))
    pred = torch.cat(r, dim=0)
    print(pred.shape)
    torch.save(pred, root + ".pt")
    np.save(root + "_before.npy", dataset.X)
    # pd.DataFrame(data={"SMILES": smiles}).to_csv(root + "_ssmi.csv")


if __name__ == "__main__":
    _load_embedding(
        path="/home/srkpa/scratch/BADDI/randomized.csv",
        model="/home/srkpa/scratch/tube/simclr/version_4_b256/conv1d.pkl",
    )
