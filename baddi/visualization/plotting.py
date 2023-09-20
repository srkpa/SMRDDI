import pandas as pd
import numpy as np
from torch import load, device
from sklearn.manifold import TSNE
import plotly.express as px
from typing import List, Union
from pathlib import Path
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdMolHash
import rdkit
from typing import Tuple


def load_file(path: str) -> Union[np.ndarray, pd.DataFrame]:
    filepath = Path(path)
    assert filepath.exists(), "File not found"
    if filepath.suffix == ".pt":
        return load(filepath, map_location=device("cpu")).detach().numpy()
    elif filepath.suffix == ".npy":
        return np.load(open(filepath, "rb"))
    else:
        return pd.read_csv(filepath, index_col=0)


def plot_bar(df: pd.DataFrame):
    return px.bar(df, orientation="h")


def plot_scatter(z: np.ndarray, x: int, y: int, labels: List, title: str = "", **args):
    fig = px.scatter(z, x=x, y=y, color=labels, labels=labels, symbol=labels, **args)
    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis_title=title,
        showlegend=False,
    )

    return fig


def tsne(d: np.ndarray, **args) -> np.ndarray:
    z = TSNE(**args)
    z = z.fit_transform(d)
    z = z.astype(float)
    return z


def plot_heatmap():
    fig = make_subplots(1, 1)
    fig.add_trace(go.Heatmap(z=cm(X, y_true), type="heatmap", colorscale="rainbow"))
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    fig.show()


def draw(
    smiles: List,
    mostFreq_murckoHash: str = "c1ccccc1",
    molsPerRow: int = 3,
    imgsize: Tuple = (250, 250),
):
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    murckoHashList = [
        rdMolHash.MolHash(mMol, rdkit.Chem.rdMolHash.HashFunction.MurckoScaffold)
        for mMol in mols
    ]

    mostFreq_murckoHash_mol = Chem.MolFromSmiles(mostFreq_murckoHash)
    highlight_mostFreq_murckoHash = [
        mMol.GetSubstructMatch(mostFreq_murckoHash_mol) for mMol in mols
    ]
    return Draw.MolsToGridImage(
        mols,
        legends=list(murckoHashList),
        highlightAtomLists=highlight_mostFreq_murckoHash,
        subImgSize=imgsize,
        useSVG=False,
        molsPerRow=molsPerRow,
    )
