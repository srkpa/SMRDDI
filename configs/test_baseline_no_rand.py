from sklearn.model_selection import ParameterGrid
from os import getcwd, path

data_dir = f"{path.dirname(getcwd())}/data"

dataset_params = list(
    ParameterGrid(
        dict(
            drugs_file=[None],
            data_file=['/home/srkpa/scratch/BADDI/data/drugbank.csv'],
            dataset_name=['drugbank'],
            transform=['SmilesToSeq'],
            min_count=[0],
            num_rounds=[1],
            debug=[False],
        )
    )
)

expt_config = dict(
    dataset_params=dataset_params,
    pretrained_model_path=[
        "/home/srkpa/scratch/BARe/NO_RAND/conv1d/86e29a46fdfd7b412cc5f9abf56a904b84c9b6fc"]
)
