import functools
import os
import re
import subprocess
import types
import warnings
from collections import defaultdict, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from functools import update_wrapper
from itertools import combinations_with_replacement
from math import ceil
from typing import Callable, Tuple, Optional, List, Any

import click
import numpy as np
import pandas as pd
import six
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import accuracy_score

FUNCTYPES = (types.FunctionType, types.MethodType, functools.partial)

_use_shared_memory = False
r"""Whether to use shared memory in custom_collate"""

from difflib import SequenceMatcher


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


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def wrapped_partial(func, **kwargs):
    partial_func = partial(func, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def instantiate(
    package: Any,
    module_name: str = "feat",
    class_name: str = "CircularFingerprint",
    **kwargs,
) -> Any:
    module = getattr(package, module_name)
    if class_name not in dir(module):
        raise Exception(
            f"Unhandled model. The name of \
               the architecture should be one of those: {dir(module)}"
        )

    obj = getattr(module, class_name)
    return obj(**kwargs)


def ensemble_predictions(models, inputs, embed=None, **kwargs):
    """
    Make an ensemble prediction for multi-class classification
    """
    print(len(models))
    y_pred = []
    for i, model in enumerate(models):
        if embed is not None:
            inputs = embed[i].transform(inputs).toarray()

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        y_hat = model(inputs)
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().numpy()
            y_pred.append(y_hat)

    y_pred = np.array(y_pred)
    print(y_pred.shape)
    # sum across ensemble members
    summed = np.sum(y_pred, axis=0)
    print(summed.shape)
    # argmax across classes
    result = np.argmax(summed, axis=1)
    print("result", result.shape)
    return result


def evaluate_n_members(models, n_models, inputs, embed=None, **kwargs):
    """
    Evaluate a specific number of members in an ensemble
    """

    inputs, labels = inputs[:, :-1], inputs[:, -1]
    print("debut eval", inputs.shape, labels.shape)
    # select a subset of members
    single_pred = ensemble_predictions([models[n_models - 1]], inputs, embed=embed)
    print("ensemble prediction")
    # make prediction
    ens_pred = ensemble_predictions(models[:n_models], inputs, embed=embed)
    # calculate accuracy
    return accuracy_score(labels, single_pred), accuracy_score(labels, ens_pred)


# TODO
# def get_tanimoto_similarity_scores(drugs_info, pairs): # pairs de smiles directement c,est mieux car j,ai les ids
#     smiles = drugs_info['SMILES'].values.tolist()
#     drug_ids =  drugs_info['GENERIC_NAME'].values.tolist()
#     ms = [Chem.MolFromSmiles(smiles) for smile in smiles]
#     fps = [Chem.RDKFingerprint(x) for x in ms]
#
#     n = len(pairs)
#     matx = np.zeros((n, n))
#     for index, x in np.ndenumerate(a):
#         pair_1 = pairs[index[0]]
#         pair_2 = pairs[index[1]]
#         fps[drug_ids.index(pair_1[0])]
#         fps[drug_ids.index(pair_1[1])]
#
#         DataStructs.FingerprintSimilarity(fps[0], fps[1])
#     # (a, b ) , (c, d)   0.5 * max(t(a, c), t(a, d)) + 0.5 * max(t(b, c), t(b, d)


def sample(
    file: str,
    samples,
    r_length: float = 0.8,
    threshold: float = 0.0,
    scorer: Callable = similar,
    sampler: Callable = combinations_with_replacement,
) -> Tuple:
    r_length = ceil(r_length * len(samples))

    i = 0
    with open(file, "w") as fe:
        print("START #i =", i, "\n")
        for comb in sampler(iterable=samples, r=r_length):
            fe.write("".join(str(e) for e in comb) + "\n")
            i += 1
            print(f"#{i}\n")
        print("END #i =", i, "\n")
    r_combinations = sampler(iterable=samples, r=r_length)
    n_comb = len(r_combinations)
    print(n_comb)
    matrix = np.zeros((n_comb, n_comb))
    for (ix, iy), _ in np.ndenumerate(matrix):
        seq_1 = "".join(str(e) for e in r_combinations[ix])
        seq_2 = "".join(str(e) for e in r_combinations[iy])
        matrix[ix, iy] = scorer(seq_1, seq_2)

    matrix = np.mean(matrix, axis=1)
    min_mat, max_mat = np.amin(matrix), np.amax(matrix)
    matrix = matrix[matrix <= threshold]
    print(len(matrix))
    return len(samples), n_comb, min_mat, max_mat, threshold, len(matrix)


@cli.command()
@click.option("--input_path", "-f", type=str, help="Training dataset file path.")
@click.option("--r_length", "-r", type=float, default=0.8)
@click.option("--threshold", "-t", type=float, default=0.0)
def sample_without_replacement(input_path, r_length, threshold):
    Xp = XpResults()
    headers = (
        "number of samples",
        "number of combinations (ini)",
        "CS (min)",
        "CS (max)",
        "threshold",
        "number of combinations (end)",
    )
    data = pd.read_csv(input_path, index_col=0)
    print(len(data))
    result = sample(
        samples=data.index.to_list(),
        r_length=r_length,
        scorer=similar,
        threshold=threshold,
        sampler=combinations_with_replacement,
        file=f"{os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])}_swr_thres_{threshold}.txt",
    )

    Xp.update(dict(zip(headers, result)))
    Xp.to_csv(
        filepath=f"{os.path.join(os.path.dirname(input_path), os.path.splitext(os.path.basename(input_path))[0])}_swr_thres_{threshold}.csv"
    )


def batch_index_select(input, dim, index):
    r"""
    Pytorch index_select for batched tensor. the main difference is index can be a 2D tensor now.

    This function returns a new tensor which indexes the input tensor along dimension `dim`
    using the entries in index which is a LongTensor. The returned tensor has the same number of dimensions
    as the original tensor (input). The dimth dimension has the same size as the length of index;
    other dimensions have the same size as in the original tensor.

    Arguments
    ----------
        input: `torch.Tensor`
            The input tensor (B, D1, D2, D3)
        dim: int
            The dimension in which we index
        index : `torch.LongTensor`
            The 2-D tensor (B, Dx) containing the indices to index

    Returns
    -------
        out: `torch.Tensor`
            The output tensor

    See Also
    --------
        :func:`~torch.index_select`

    """
    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def is_callable(func):
    r"""
    Check if input is a function or a callable

    Arguments
    ----------
        func: type
            The python variable/type that will be checked

    Returns
    -------
        is_func: bool
            whether input argument is a python function or callable
    """
    return func and (isinstance(func, FUNCTYPES) or callable(func))


def reduce_dimension(data, method="pca", **kwargs):
    r"""
    Reduce the dimension of a data using either PCA or TSNE

    Arguments
    ----------
        data: array
            Original data in high dimension
        method: str, optional
            Projection method to use. Either `pca` or `tsne`.
            (Default value = 'pca')
        **kwargs:
            Named parameter for the underlying sklearn model
            to use. For example, 'tsne' has n_components, perplexity,
            early_exaggeration, learning_rate, etc.
            The full list of parameter is found in sklearn documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Returns
    -------
        res:
            Data in a lower dimension
        model:
            Model used to reduce dimensionality. This is returned for convenience,
            so you could check for example the percent of variance explained.
    """
    n = kwargs.pop("n_components", 2)
    if method.lower() == "pca":
        model = PCA(n_components=n, **kwargs)
    elif method.lower() == "tsne":
        perplx = kwargs.pop("perplexity", 50)
        v = kwargs.pop("verbose", 0)
        n_iter = kwargs.pop("n_iter", 250)  # hmmm no too much of this
        model = TSNE(
            n_components=n, perplexity=perplx, verbose=v, n_iter=n_iter, **kwargs
        )
    else:
        raise NotImplementedError(f"Provided method ({method}) is not supported !")

    res = model.fit_transform(data)
    return res, model


def get_gpu_memory_map():
    r"""
    Get the current gpu usage.

    Returns
    -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
    """
    # nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"]
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    return dict(zip(range(len(gpu_memory)), gpu_memory))


def to_categorical(y, num_classes=None, dtype=float):
    r"""Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.
    .. deprecated:: 3.1
    Use :func:`sklearn.preprocessing.LabelEncoder` instead.

    Arguments
    ----------
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: int, optional
            total number of classes.
            (Default value = None), elements in y are used to infer the number
            of classes
        dtype: type, optional
            data type of the the return.
            (Default value = float). Other possible types are int, bool, ...

    Returns
    -------
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    if len(input_shape) > 1 and any(np.array(input_shape[1:]) > 1):
        raise ValueError(f"The input y should be a 1D array, provided shape: {y.shape}")
    y = y.ravel()

    num_classes_min = np.max(y) + 1
    if not num_classes:
        num_classes = num_classes_min
    if num_classes < num_classes_min:
        raise ValueError(
            f"Provided num_classes ({num_classes}) is too low to fit all the classes from provided y ({y}) !"
        )

    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def make_hash(depth=None, type=None):
    r"""
    Make a multilevel dict (A dict, of dict, of dict, ...) with `depth`
    recursion level. The deepest dict is forced to have an object
    of the class defined by `type`.

    Arguments
    ----------
        depth: int, optional
            the lowest depth of the dict. A depth of 0
            corresponds to a standard dict. If depth is set to None, then
            the dict is recursively defined for new keywords.
            (Default value = None)
        type: object, optional
            enforces an object type for the content of
            the dict. By default, any object type will be accepted as value
            (Default value = None)

    Returns
    -------
        Multiple level dict
    """
    if (depth, type) == (None, None):
        return defaultdict(make_hash)
    elif depth == 0 or depth is None:
        return defaultdict(type)
    return defaultdict(partial(make_hash, depth - 1, type))


def one_of_k_encoding(val, num_classes, dtype=int):
    r"""Converts a single value to a one-hot vector.

    Arguments
    ----------
        val: int
            class to be converted into a one hot vector
            (integers from 0 to num_classes).
        num_classes: iterator
            a list or 1D array of allowed
            choices for val to take
        dtype: type, optional
            data type of the the return.
            (Default value = int). Other possible types are float, bool, ...
    Returns
    -------
        A numpy 1D array of length len(num_classes) + 1
    """

    encoding = np.zeros(len(num_classes) + 1, dtype=dtype)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(num_classes):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
    return encoding


def params_getter(param_dict, prefix):
    r"""
    Filter a parameter dict to keep only a list of relevant parameters

    Arguments
    ----------
        param_dict: dict
            Input dictionary to be filtered
        prefix (str or iterable): A string or a container that contains
            the required string prefix(es) to select parameters of interest

    Returns
    -------
        filtered_dict: dict
            Dict of (param key, param value) after
            filtering, and with the prefix removed
    """

    # Check the instances of the prefix
    if isinstance(prefix, str):
        prefix = [prefix]
    if not (isinstance(prefix, (list, tuple))):
        raise ValueError(
            f"Expect a string, a tuple of strings or a list of strings, got {type(prefix)}"
        )
    if not all(isinstance(this_prefix, str) for this_prefix in prefix):
        raise ValueError("All the prefix must be strings")

    # Create the new dictionary
    new_dict = {}
    for pkey, pval in param_dict.items():
        for this_prefix in prefix:
            if pkey.startswith(this_prefix):
                new_dict[pkey.split(this_prefix, 1)[-1]] = pval

    return new_dict


def to_sparse(x, dtype=None):
    r"""
    Converts dense tensor x to sparse format

    Arguments
    ----------
        x: torch.Tensor
            tensor to convert
        dtype: torch.dtype, optional
            Enforces new data type for the output. If None, it keeps the same datatype as x
            (Default value = None)
    Returns
    -------
        new torch.sparse Tensor

    """

    if dtype is not None:
        x = x.type(dtype)

    x_typename = torch.typename(x).split(".")[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def to_tensor(x, gpu=True, dtype=None):
    r"""
    Convert a numpy array to tensor. The tensor type will be
    the same as the original array, unless specify otherwise

    Arguments
    ----------
        x: numpy.ndarray
            Numpy array to convert to tensor type
        gpu: bool optional
            Whether to move tensor to gpu.
        dtype: torch.dtype, optional
            Enforces new data type for the output

    Returns
    -------
        New torch.Tensor

    """
    x = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.Tensor(x)
    if dtype is not None:
        x = x.type(dtype)
    if torch.cuda.is_available() and gpu:
        x = x.cuda()
    return x


def custom_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if re.search("[SaUO]", elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith("float") else int
            return np.array(list(map(py_type, batch)), dtype=py_type)
    elif isinstance(batch[0], six.integer_types):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], six.string_types):
        return batch
    elif isinstance(batch[0], Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], Sequence):
        print(batch)
        transposed = zip(*batch)
        print(transposed)
        exit()
        return [custom_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def mol2img(
    mol,
    mol_w=300,
    mol_h=100,
    kekulize=True,
    explicit_H=False,
    hightlightAtoms=None,
    **kwargs,
):
    r"""
    Convert a molecule into an image. Can be used to highlight a list of atoms in the molecule.

    Arguments
    ----------
        mol: rdkit.Chem.Molecule or str
            molecule to draw
        mol_w: int, optional
            width of the image
            (Default value = 300)
        mol_h: int, optional
            height of the image
            (Default value = 100)
        kekulize: bool, optional
            Whether to kekulize molecule or not
            (Default value = false)
        explicit_H: bool, optional
            Whether to show hydrogen atom
            (Default value = False)
        hightlightAtoms: list of int, optional
            List of integers representing the atoms to highlight in the image
            (Default value = [])
        **kwargs: dict
            Named additional parameters for Chem.Draw.MolToImage

    Returns
    -------
        A PIL image, that can be displayed (eg.: in jupyter-notebook)

    Examples
    -------
        In the following, the mol2img function will be used for interactive exploration
        of molecule in a given dataset

        .. code-block:: python

            from rdkit import Chem
            from ipywidgets import interact
            from ivbase.utils.commons import mol2img
            mols = [
                "CCOc1c(OC)cc(CCN)cc1OC",
                "COc1cc(CCN)cc(OC)c1OC",
                "COc1c2OCCc2c(CCN)c2CCOc12",
                "COc1cc(CCN)c2CCOc2c1OC",
                "C[C@H](N)Cn1ncc2ccc(O)cc12",
                "COc1ccc2cnn(C[C@H](C)N)c2c1",
                "C[C@H](N)Cn1ncc2ccc(O)c(C)c12",
            ]

        .. code-block:: python

            def show_molecule(index):
                pil_im = mol2img(mols[index], mol_w=400, mol_h=250) # explicit_H=True to show hydrogen
                return display(pil_im)

            interact(show_molecule,  index=(0, len(smiles)- 1, 1))
    """
    if hightlightAtoms is None:
        hightlightAtoms = []
    if isinstance(mol, str):  # allow failing in any other circumstances
        mol_str = mol
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            warnings.warn(f"Error converting molecule {mol_str} to image")
    if mol is None:
        warnings.warn("Cannot convert NoneType molecule to image")
        return None
    if explicit_H:
        mol = Chem.AddHs(mol)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(
        mol, (mol_w, mol_h), kekulize=kekulize, highlightAtoms=hightlightAtoms, **kwargs
    )


def is_dtype_torch_tensor(dtype):
    r"""
    Verify if the dtype is a torch dtype

    Arguments
    ----------
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns
    -------
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


def is_dtype_numpy_array(dtype):
    r"""
    Verify if the dtype is a numpy dtype

    Arguments
    ----------
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns
    -------
        A boolean saying if the dtype is a numpy dtype
    """
    is_torch = is_dtype_torch_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, "__module__"):
        is_numpy = dtype.__module__ == "numpy"
    else:
        is_numpy = False

    return (is_num or is_numpy) and not is_torch


def map_vocab(vocab):
    r"""
    Map element of a vocabulary to their index

    Arguments
    ----------
        vocab: list
            List of elements in the vocabulary. Should be a list, as order is important
    Returns
    -------
        mapped_vocab: dict
            Mapping of each element of `vocab` to its position in the list.
    """
    return dict(zip(vocab, range(len(vocab))))


if __name__ == "__main__":
    cli()
