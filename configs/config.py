import json
import os

from sklearn.model_selection import ParameterGrid


def generate_config_file(cfg_file, nw_params, exp):
    with open(cfg_file, 'r') as tc:
        params_grid = json.load(tc)
        params_grid.update(nw_params)

    for k, v in params_grid.items():
        params_grid[k] = v if isinstance(v, list) else [v]

    task_grid = list(ParameterGrid(params_grid))

    for task_id, task in enumerate(task_grid):
        task_id += 1
        with open(f"{exp}_{task_id}.json", 'w') as f:
            json.dump(task, f, indent=4)
        assert os.path.exists(f"{exp}_{task_id}.json")


if __name__ == '__main__':
    config_file = 'siam_no_rand.json'
    # new_params = {
    #     "embedding": None,
    #     "embedding_smiles": None,
    #     "transform": ["Mol2VecFingerprint", "MACCSKeysFingerprint", "CircularFingerprint", "PubChemFingerprint",
    #                   "RDKitDescriptors", "MordredDescriptors"]
    # }
    # exp_name = 'siam_featurizers'
    # new_params = {
    #     "data_file": "/home/srkpa/scratch/BADDI/data/chem_random_ou/drugbank.csv",
    #     "embedding": "/home/srkpa/scratch/tube/simclr/version_4_b256/predictions.pt"
    # }
    new_params = {
        'data_file': [
            "/home/srkpa/scratch/BADDI/data/chem_random_ou/drugbank.csv",
            "/home/srkpa/scratch/BADDI/data/chem_random/drugbank.csv",
            "/home/srkpa/scratch/BADDI/data/random/drugbank.csv"
        ]
    }
    exp_name = 'siam_no_rand'
    generate_config_file(config_file, new_params, exp_name)
