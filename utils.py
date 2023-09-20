from dataclasses import dataclass, field
from functools import partial
from functools import update_wrapper
from typing import Callable, Tuple, Optional, List, Any

import click
import numpy as np
import pandas as pd
from deepchem.trans import FeaturizationTransformer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import csv
import os


@click.group()
def cli():
    pass


@dataclass
class XpResults:
    _df: Optional[List] = field(default_factory=list)

    def update(self, kwargs):
        self._df.append(kwargs)

    def to_csv(self, filepath):
        output = pd.DataFrame(self._df)
        output.to_csv(filepath)


def wrapped_partial(func, **kwargs):
    partial_func = partial(func, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def instantiate(package: Any, module_name: str = 'feat', class_name: str = 'CircularFingerprint',
                **kwargs) -> Any:
    module = getattr(package, module_name)
    if class_name not in dir(module):
        raise Exception(f"Unhandled model. The name of \
               the architecture should be one of those: {dir(module)}")

    obj = getattr(module, class_name)
    return obj(**kwargs)


def featurize(db, smiles, X):
    drugs_a, drugs_b = db['X1'].values.tolist(), db['X2'].values.tolist()
    drugs_a_ids = list(map(smiles.SMILES.tolist().index, drugs_a))
    drugs_b_ids = list(map(smiles.SMILES.tolist().index, drugs_b))
    features = np.concatenate((X[drugs_a_ids], X[drugs_b_ids]), axis=1)
    print(features.shape)
    labeler = LabelEncoder()
    targets = labeler.fit_transform(db['Y'].values.tolist())
    return features, targets

def save_experiment(experiment, filepath : str, headers: List = None):
    if headers is None:
        headers = []
    file_exist = os.path.isfile(filepath)

    with open(filepath, 'a+', encoding='UTF8') as f:
        writer =csv.DictWriter(f, delimiter=',', fieldnames=headers)
        if not file_exist :
            writer.writeheader() 
        writer.writerow(experiment)
       

@cli.command()  
@click.option(
    "-f",
    "--featurizer",
    prompt=False,
    help="featurizer name",
    required=True,
)
@click.option(
    "-s",
    "--smiles",
    prompt=False,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=True,
        resolve_path=True,
        path_type=Path,
    ),
    default="data/drugbank_smiles.csv",
)
def features(featurizer, smiles, **kwargs):
    import deepchem as dc
    featurizer_name = featurizer
    featurizer = instantiate(
                    package=dc,
                    module_name="feat",
                    class_name= featurizer_name,
                    **kwargs
                )
    smiles = pd.read_csv(smiles, index_col=0) 
    print(len(smiles))
    y = np.arange(len(smiles))
    dataset = dc.data.NumpyDataset(smiles.SMILES, y, ids=smiles.SMILES)
    trans = FeaturizationTransformer(dataset, featurizer)
    dataset = trans.transform(dataset)
    features = dataset._X
    ids = dataset.ids
    features = features[~np.isnan(features).any(axis=1),:]
    print(features.shape, len(ids))
    features = np.float32(features)

    np.save(open(f'data/{featurizer_name}.npy', 'wb'), features)

if __name__ == '__main__':
    cli()
