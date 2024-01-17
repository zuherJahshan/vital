import os
from typing import List, Tuple
from mh_genome_ds_v2 import MHGenomeDS
from dataset import Dataset, Label, Filepath

Dirpath = str


def create_dataset(
    type: str, 
    mapping: List[Tuple[Filepath, Label]],
    labels: List[Label] = None,
):
    if type == "MHGenomeDS":
        return MHGenomeDS(mapping, labels)
    else:
        # throw exception, there is no dataset like this
        raise Exception(f"Dataset type {type} does not exist.")
    return None
        

def get_dataset_types() -> List[str]:
    return [
        "MHGenomeDS"
    ]


def load_dataset(dirpath: Dirpath) -> Dataset:
    if os.path.exists(f"{dirpath}/dataset.json") and os.path.exists(f"{dirpath}/config.txt"):
        # read the config file to understand which class should be retrieved
        with open(f"{dirpath}/config.txt", 'r') as f:
            class_name = f.readline().strip()
        if class_name == "MHGenomeDS":
            return MHGenomeDS(mapping = None, load=(True, dirpath))
        else:
            raise Exception(f"Dataset type {class_name} does not exist.")
        return None
    else:
        raise Exception(f"File {dirpath}/dataset.json does not exist or {dirpath}/config.txt does not exist.")


def remove_dataset(dirpath: Dirpath) -> None:
    if os.path.exists(f"{dirpath}/dataset.json"):
        os.remove(f"{dirpath}/dataset.json")
        os.remove(f"{dirpath}/config.txt")