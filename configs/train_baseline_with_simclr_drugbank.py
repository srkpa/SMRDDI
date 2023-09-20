from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(
        drugs_file=[None],
        data_file=['/home/srkpa/scratch/BADDI/data/chem_random/drugbank.csv',
                   '/home/srkpa/scratch/BADDI/data/chem_maxmin/drugbank.csv',
                   '/home/srkpa/scratch/BADDI/data/chem_molecularw/drugbank.csv',
                   '/home/srkpa/scratch/BADDI/data/chem_scaffold/drugbank.csv',
                   '/home/srkpa/scratch/BADDI/data/chem_butina/drugbank.csv',
                   '/home/srkpa/scratch/BADDI/data/chem_fingerprint/drugbank.csv'],
        dataset_name=['drugbank'],
        transform=[None],
        min_count=[0],
        num_rounds=[1],
        debug=[False],
        embedding=[['/home/srkpa/scratch/BADDI/data/drugbank_smiles.csv',
                    "/home/srkpa/scratch/tube/simclr/version_4_b256/predictions.pt"]]
    )
))

train_params = list(ParameterGrid(
    dict(max_epochs=[200],
         batch_size=[256],
         test_size=[0.2],
         valid_size=[0.2]
         )))

model_params = list(ParameterGrid(dict(
    base_network=["identity"],
    projection_head=['feedforward']

)))

expt_config = dict(
    model_params=model_params,
    dataset_params=dataset_params,
    train_params=train_params,
    pruning=[True],
    n_trials=[50],
    experiment_name=['trained_simclr_drugbank'],
)
