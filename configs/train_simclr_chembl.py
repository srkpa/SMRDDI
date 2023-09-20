from os import getcwd, path

from sklearn.model_selection import ParameterGrid

data_dir = "/home/srkpa/scratch/BADDI/data"

dataset_params = list(ParameterGrid(
    dict(
        drugs_file=['/home/srkpa/scratch/BADDI/data/chembl.csv'],
        data_file=[None],
        dataset_name=['chembl'],
        use_randomize_smiles=[False],
        use_contrastive_transform=[True],
        transform=['SmilesToSeq'],
        target_transform=[None],
        min_count=[0],
        num_rounds=[1],  # aller jusqu,a 10
        debug=[False]
    )
))

train_params = list(ParameterGrid(
    dict(max_epochs=[200],
         batch_size=[512],
         test_size=[0.2],
         valid_size=[0.2],
         splitter=['MaxMinSplitter']
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
    experiment_name=['simclr_512'],
    pred_file=['/home/srkpa/scratch/BADDI/data/drugbank_smiles.csv']
)
