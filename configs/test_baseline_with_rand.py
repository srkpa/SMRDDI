from sklearn.model_selection import ParameterGrid

dataset_params = list(ParameterGrid(
    dict(
        drugs_file=[None],
        data_file=['/home/rogia/datasets/drugbank.csv'],
        dataset_name=['drugbank'],
        transform=['SmilesToSeq'],
        min_count=[0],
        num_rounds=[1],  # aller jusqu,a 10
        debug=[True]
    )
))

expt_config = dict(
    dataset_params=dataset_params,
    pretrained_model_path=[
        "/home/rogia/Documents/git/RuRe/baselines_with_rand/conv1d_08b11d7789e62972bd51acf10d382f3100cd8e6c"]
)
