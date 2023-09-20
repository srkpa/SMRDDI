from sklearn.model_selection import ParameterGrid
from os import getcwd, path

data_dir = f"{path.dirname(getcwd())}/data"

dataset_params = list(ParameterGrid(
    dict(
        drugs_file=[None],
        data_file=['/home/srkpa/scratch/BADDI/data/drugbank.csv'],
        dataset_name=['drugbank'],
        use_randomize_smiles=[True],
        transform=['SmilesToSeq'],
        min_count=[0],
        num_rounds=[1, 2],  # aller jusqu,a 1
        debug=[False]
    )
))

train_params = list(ParameterGrid(
    dict(max_epochs=[100],
         batch_size=[256],
         test_size=[0.2],
         valid_size=[0.2]
         )))

model_params = list(ParameterGrid(dict(
    base_network=['conv1d'],
    projection_head=['feedforward']

)))

expt_config = dict(
    model_params=model_params,
    dataset_params=dataset_params,
    train_params=train_params,
    pruning=[True],
    n_trials=[10],
    experiment_name=['baselines_with_rand'],
)
