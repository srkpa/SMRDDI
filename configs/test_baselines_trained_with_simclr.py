from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(drugs_file=[None],
         data_file=['/home/rogia/datasets/drugbank.csv'],
         dataset_name=['drugbank'],
         transform=['SmilesToSeq'],
         min_count=[0],
         num_rounds=[1],
         debug=[True]
         )
))
expt_config = dict(
    dataset_params=dataset_params,
    pretrained_model_path=[
        '/home/rogia/Documents/git/RuRe/baselines_with_trained_simclr/conv1d_2de489e0645bba5a2e8bcbd7ed76185bba340655/a1c48b7d76b5e83bc27e4dc60bd7b7215886f33a'],
    experiment_name=['baselines_with_trained_simclr']
)
