
 from deepchem.molnet import load_chembl 
 
def load_molnet(dataset_name: str = "chembl"):

    load_chembl(
        set="sparse", data_dir="../../data", save_dir="../../data", reload=False
    )
    
if __name__ == "__main__":
    load_molnet()