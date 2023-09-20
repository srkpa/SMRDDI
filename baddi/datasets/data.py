import os
import pickle as pk
from typing import *

import deepchem as dc
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from deepchem.feat.molecule_featurizers import create_char_to_idx
from deepchem.data import NumpyDataset
from deepchem.splits import RandomSplitter, ScaffoldSplitter
from rdkit.Chem.PandasTools import LoadSDF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import DataLoader
from baddi.datasets.sampler import SamplingConfig
from torch.utils.data import WeightedRandomSampler

from baddi.datasets.transformer import *
from baddi.models.trainer import SimCLR
from baddi.utils import instantiate


m_map = dict(
    zip(
        [
            "Mol2VecFingerprint",
            "MACCSKeysFingerprint",
            "CircularFingerprint",
            "PubChemFingerprint",
            "RDKitDescriptors",
            "MordredDescriptors",
        ],
        [300, 167, 2048, 881, 208, 1613],
    )
)

PAD_TOKEN = "<pad>"
OUT_OF_VOCAB_TOKEN = "<unk>"


def _train_test_valid_split(
    data,
    root: str,
    mode: str = "random",
    dc_split: str = None,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    seed: int = 42,
):

    if mode == "random":
        db_train, db_test = train_test_split(
            data,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=data["Y"],
        )
        db_train, db_val = train_test_split(
            db_train,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=db_train["Y"],
        )

    else:
        drugs = list(set(data["X1"].values.tolist() + data["X2"].values.tolist()))
        drugs = sorted(drugs)
        train_idx, test_idx = train_test_split(
            drugs, test_size=test_size, random_state=seed
        )
        train_idx, valid_idx = train_test_split(
            train_idx, test_size=valid_size, random_state=seed
        )

        train_dd = set(product(train_idx, repeat=2))

        if mode == "one_unseen":
            valid_dd = set(product(train_idx, valid_idx)).union(
                set(product(valid_idx, train_idx))
            )
            test_dd = set(product(train_idx, test_idx)).union(
                set(product(test_idx, train_idx))
            )
        else:
            valid_dd = set(product(valid_idx, repeat=2))
            test_dd = set(product(test_idx, repeat=2))

        train = pd.DataFrame(list(train_dd), columns=["X1", "X2"])
        valid = pd.DataFrame(list(valid_dd), columns=["X1", "X2"])
        test = pd.DataFrame(list(test_dd), columns=["X1", "X2"])

        db_train = pd.merge(data, train, on=["X1", "X2"])
        db_val = pd.merge(data, valid, on=["X1", "X2"])
        db_test = pd.merge(data, test, on=["X1", "X2"])

        shared_y = (
            set(db_train.Y.unique())
            .intersection(set(db_test.Y.unique()))
            .intersection(set(db_val.Y.unique()))
        )

        db_train = db_train[db_train["Y"].isin(list(shared_y))]
        db_test = db_test[db_test["Y"].isin(list(shared_y))]
        db_val = db_val[db_val["Y"].isin(list(shared_y))]

    return db_train, db_val, db_test


def args_setup(dataset_params: Dict, train_params: Dict = None):
    train_params = {} if train_params is None else train_params
    if dataset_params["data_file"] is None:
        data = split_from_frame(
            file=dataset_params["drugs_file"],
            is_supervised=False,
            sep=None,
            **train_params,
        )
        is_supervised = False
    else:
        data = split_from_frame(
            file=dataset_params["data_file"],
            is_supervised=True,
            sep=",",  # \t
            **train_params,
        )
        is_supervised = True
    add_args = {}
    if dataset_params["transform"] == "SmilesToSeq":
        if is_supervised:
            smi = set(data["X1"].values.tolist() + data["X2"].values.tolist())
            root = os.path.dirname(dataset_params["data_file"])
            name, _ = os.path.basename(dataset_params["data_file"]).split(".")
        else:
            smi = data["SMILES"].values.tolist()
            root = os.path.dirname(dataset_params["drugs_file"])
            name, _ = os.path.basename(dataset_params["drugs_file"]).split(".")

        smi_file = f"{root}/{name}_smiles.csv"
        print("Smiles file :", smi_file)

        if not os.path.isfile(smi_file):
            smiles = pd.DataFrame(data=list(smi), columns=["SMILES"])
        else:
            print("exists: True")
            smiles = pd.read_csv(smi_file, sep=",", index_col=0, dtype=str)
            smiles.dropna(inplace=True)

        max_len = max(len(s) for s in smiles["SMILES"].values.tolist())

        char_set = set()
        for smile in smiles["SMILES"]:
            if len(smile) <= max_len:
                char_set.update(set(smile))

        unique_char_list = list(char_set)
        unique_char_list += [PAD_TOKEN, OUT_OF_VOCAB_TOKEN]
        char_to_idx = {letter: idx for idx, letter in enumerate(unique_char_list)}

        smiles["SMILES"].to_csv(smi_file)

        add_args = dict(
            char_to_idx=char_to_idx,
            max_len=max_len,
            pad_len=0,
        )
    return add_args


def split_from_frame(
    file: str,
    mode: str = "",
    sep: Union[str, None] = "\t",
    index_col: int = 0,
    is_supervised: bool = False,
    splitter: str = None,
    seed: int = 42,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    max_epochs: int = 2,
    batch_size: int = 3,
    **kwargs,
) -> pd.DataFrame:
    root, ext = os.path.splitext(file)
    data = pd.read_csv(file, sep=sep, index_col=index_col)

    if not (os.path.isfile(f"{root}_train.csv") and os.path.isfile(f"{root}_val.csv")):
        if is_supervised:
            print("Mode = ", mode, "\n SEED = ", seed)
            db_train, db_val, db_test = _train_test_valid_split(
                data=data,
                root=root,
                mode=mode,
                test_size=test_size,
                valid_size=valid_size,
                seed=seed,
            )

            db_test.to_csv(f"{root}_test.csv")
        else:
            splitter_inst = instantiate(
                package=dc, module_name="splits", class_name=splitter, **kwargs
            )
            num_drugs = len(data)
            dc_dataset = dc.data.NumpyDataset(
                X=np.arange(num_drugs), y=np.arange(num_drugs), ids=data["SMILES"]
            )
            db_train, db_test = splitter_inst.train_test_split(
                dataset=dc_dataset, frac_train=1.0 - test_size, seed=seed
            )
            db_train = pd.DataFrame(data=db_train.ids, columns=["SMILES"])
            db_val = pd.DataFrame(data=db_test.ids, columns=["SMILES"])
        db_train.to_csv(f"{root}_train.csv")
        db_val.to_csv(f"{root}_val.csv")
    return data


def readfile(file: str, sep: str = ",") -> pd.DataFrame:
    if file.lower().endswith("pck"):
        content = pk.load(open(file, "rb"))
        content = pd.DataFrame(
            {
                "X1": [x for (x, _) in content.keys()],
                "X2": [x for (_, x) in content.keys()],
                "Y": content.values(),
            }
        )
    elif file.lower().endswith("csv"):
        content = pd.read_csv(filepath_or_buffer=file, sep=sep, index_col=0)
    else:
        content = LoadSDF(file, smilesName="SMILES")
    return content


def loadfile(
    index_col: str,
    data_file: str,
    drugs_file: Optional[str] = None,
    seed: int = 42,
    debug: bool = True,
    sep: str = ",",
) -> Tuple:
    if data_file is not None:
        datas = (
            readfile(file=data_file, sep=sep)
            if os.path.isfile(data_file)
            else pd.DataFrame(columns=["X1", "X2", "Y"])
        )
        if drugs_file is None:
            samples, drugs = datas[["X1", "X2", "Y"]], None
        else:
            drugs = readfile(file=drugs_file)
            drugs.set_index(index_col, inplace=True)
            samples = datas.loc[
                datas["X1"].isin(drugs.index) & datas["X2"].isin(drugs.index)
            ]

        if debug:
            samples = samples.groupby("Y", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 3), random_state=seed)
            )
    else:
        samples = None
        drugs = (
            readfile(file=drugs_file)
            if os.path.isfile(drugs_file)
            else pd.DataFrame(columns=["SMILES"])
        )
        if debug:
            drugs = drugs.head(n=10)
    return samples, drugs


class DrugDataset(data.Dataset):
    def __init__(
        self,
        drugs_file: str,
        data_file: str,
        dataset_name: str,
        num_rounds: int = 1,
        use_randomize_smiles: bool = False,
        use_contrastive_transform: bool = False,
        transform: Union[Optional[Callable], str] = None,
        target_transform: Optional[Callable] = LabelEncoder(),
        min_count: int = 1,
        debug: bool = False,
        index_col: str = "GENERIC_NAME",
        split: str = "train",
        embedding: list = None,
        **kwargs,
    ) -> None:
        super(DrugDataset, self).__init__()

        if data_file is None:
            root, ext = os.path.splitext(drugs_file)
            drugs_file = f"{root}_{split}{ext}"
        else:
            root, ext = os.path.splitext(data_file)
            data_file = f"{root}_{split}{ext}"

        self.data, self.drugs = loadfile(
            index_col=index_col, data_file=data_file, drugs_file=drugs_file, debug=debug
        )
        self.dataset_name = dataset_name

        self.transform = transform
        self.convert = False

        if isinstance(self.transform, str):
            self.transform = instantiate(
                package=dc, module_name="feat", class_name=transform, **kwargs
            ).featurize

            if transform in m_map:
                self.n_features = m_map[transform]
                self.convert = True

        if embedding is not None:
            emb_d_file, emb_t_file = embedding
            smiles = pd.read_csv(emb_d_file, index_col=0)
            self.smiles = smiles["SMILES"].values.tolist()
            self.embeddings = torch.load(emb_t_file)
            self.transform = self.fn
            self.n_features = self.embeddings.shape[1]

            print("N features per drugs : ", self.n_features, self.embeddings.shape)

        self.target_transform = target_transform

        if self.target_transform is None:
            self.data, self.targets, self.num_classes = (
                self.drugs["SMILES"].values,
                None,
                0,
            )
        else:
            labels: Optional[np.ndarray] = self.data.Y.values
            values, counts = np.unique(labels, return_counts=True)
            mask = np.isin(labels, values[counts >= min_count])
            self.num_classes = len(values[counts >= min_count])
            self.targets = target_transform.fit_transform(labels[mask])
            # assert same order of classes
            assert all(
                target_transform.classes_ == (values[counts >= min_count])
            ), "Not same order of y classes"

            # Add class weights
            self.class_weights = 1 / np.bincount(
                self.targets
            )  # 1 / counts[counts >= min_count] # astype np.float64, 1.0 / division en double
            assert (
                len(self.class_weights) == self.num_classes
            ), f"class weight shape = {self.class_weights.shape}, num_classes = {self.num_classes}"

            # Add sample weights
            self.sample_weights = np.array(
                [self.class_weights[index] for index in self.targets]
            )

            assert len(self.sample_weights) == len(
                self.targets
            ), f"Samples weight shape = {self.sample_weights.shape}, num_samples = {len(self.targets)}"

            self.data: Optional[np.ndarray] = self.data[["X1", "X2"]].values[mask]
            if self.drugs is not None:

                def func(mol_id):
                    return self.drugs.loc[mol_id, "SMILES"]

                vfunc = np.vectorize(func)
                self.data = vfunc(self.data)

        self.use_contrastive_transform = use_contrastive_transform

        if use_randomize_smiles:

            self.mol_transform = Randomizer(
                data=self.data, targets=self.targets, n_times=num_rounds
            )

        else:
            self.mol_transform = None

        if (
            split == "train" and self.mol_transform is not None
        ):  # or use_contrastive_transform:
            self.data, self.targets = self.mol_transform()


        print(
            self.data.shape,
            self.targets.shape
            if isinstance(self.targets, np.ndarray)
            else self.targets,
            self.num_classes,
            min_count,
            "use random = ",
            use_randomize_smiles,
            "use constrative transform",
            use_contrastive_transform,
            num_rounds,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.targets is None:
            moles, target = self.data[index], -1
        else:
            moles, target = self.data[index], int(self.targets[index])

        if self.use_contrastive_transform:
            moles = [moles, randomize_smiles(moles)]

        if self.transform is not None:
            moles = [np.squeeze(self.transform([mol])) for mol in moles]

            if self.convert:
                moles = [np.float32(x) for x in moles]

        return moles, target

    def __len__(self) -> int:
        return self.data.shape[0]

    def fn(self, drug_id):
        return self.embeddings[self.smiles.index(drug_id[-1])]


class DrugDataModule(pl.LightningDataModule):
    def __init__(self, dataset_params: Dict, train_params: Dict, seed: int = 42):
        super(DrugDataModule, self).__init__()
        self.use_sampler = train_params.pop("balance")
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        add_args = args_setup(
            dataset_params=self.hparams.dataset_params,  # type: ignore
            train_params=self.hparams.train_params,
        )

        self.train_dataset = DrugDataset(
            split="train", **self.hparams.dataset_params, **add_args
        )
        self.valid_dataset = DrugDataset(
            split="val", **self.hparams.dataset_params, **add_args
        )
        self.test_dataset = DrugDataset(
            split="test", **self.hparams.dataset_params, **add_args
        )
        assert self.train_dataset.num_classes == self.valid_dataset.num_classes

    def train_dataloader(self) -> DataLoader:

        if self.use_sampler == "equal":
            return SamplingConfig(balance=True).create_loader(
                dataset=self.train_dataset,
                batch_size=self.hparams.train_params["batch_size"],
            )
        elif self.use_sampler == "class_weight":
            return SamplingConfig(
                balance=False, class_weights=self.train_dataset.class_weights
            ).create_loader(
                dataset=self.train_dataset,
                batch_size=self.hparams.train_params["batch_size"],
            )
        elif self.use_sampler == "sample_weight":
            targets = torch.from_numpy(self.train_dataset.targets)
            _, counts = targets.unique(return_counts=True)
            classWeights = (1 / counts).double()
            sampleWeights = torch.take_along_dim(classWeights, targets)
            sampler = WeightedRandomSampler(
                weights=sampleWeights, num_samples=len(targets), replacement=True
            )

            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.train_params["batch_size"],
                sampler=sampler,
            )
        elif self.use_sampler == "auto":
            return SamplingConfig().create_loader(
                dataset=self.train_dataset,
                batch_size=self.hparams.train_params["batch_size"],
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.train_params["batch_size"],
                shuffle=True,
                pin_memory=True,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.train_params["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                self.test_dataset,
                batch_size=self.hparams.train_params["batch_size"],
                shuffle=False,
                pin_memory=True,
            )
           