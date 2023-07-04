import tensorflow as tf
import os
import json
import random
from typing import List, Tuple, Dict
from enum import Enum
import shutil

__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../utils/")
from utils import sub_dirpath_of_dirpath

os.chdir(f"{__ORIG_WD__}/../dataset/")
from dataset import Dataset, load_dataset, remove_dataset

os.chdir(f"{__ORIG_WD__}/ml_models/")

from ml_model import MLModel
from ml_models import get_ml_model_constructor

os.chdir(__ORIG_WD__)


Filepath = str
Label = str

class DatasetName(Enum):
    trainset = "trainset"
    validset = "validset"
    testset = "testset"

AccessionPath = str


def make_soft_links_to_existing_files(dest_dirpath: str, existing_filepaths: List[str]):
    os.makedirs(dest_dirpath, exist_ok=True)
    
    # check all files exists, if not raise an error
    for filepath in existing_filepaths:
        if not os.path.exists(filepath):
            raise Exception(f"File {filepath} does not exist.")
        
    # create soft links to all files
    for filepath in existing_filepaths:
        os.symlink(filepath, f"{dest_dirpath}/{os.path.basename(filepath)}")



class Model(object):
    def __init__(self, name, data_dir = "./data", load = False):
        # create models_dir in case it does not exist
        os.makedirs(data_dir, exist_ok=True)
        
        if sub_dirpath_of_dirpath(os.getcwd(), data_dir) == False:
            raise Exception(f"data_dir {data_dir} is not a subdirectory of the current working directory {os.getcwd()}")
        
        self._createModelState(name, data_dir)
        if os.path.exists(self._get_model_path()):
            if load:
                self._loadModelState()
            else:
                raise Exception(f"Model {name} already exists, please use load_model(name) instead")
            

    def create_datasets(
        self,
        dataset: List[Tuple[Label, List[Filepath]]],
        portions: Dict[DatasetName, float],
        force_creation: bool = False,
        minhash_dataset: bool = False 

    ):
        '''
        1. check validity of parameters
        2. Check that all files exists
        3. remove old datasets folders (only if force_creation is True, else raise Exception)
        4. divide large dataset into trainset, validset and testset
        5. create dataset for each of trainset, validset and testset.
        '''

        # 1. check validity of parameters
        for label, accessions in dataset:
            # 2. check that all files exists
            for filepath in accessions:
                if not os.path.exists(filepath):
                    raise Exception(f"File {filepath} does not exist.")
                
        if sum(portions.values()) != 1.0:
            raise Exception(f"Sum of portions must be 1.0, but is {sum(portions.values())}")
        
        # 3. remove old datasets folders (only if forcd_creation is True, else raise Exception)
        if not os.path.exists(self._get_model_path()) or force_creation:
            # Create new folder if needed.
            os.makedirs(self._get_model_path(), exist_ok=True)

            trainset, validset, testset = self._create_example_label_mapping(dataset, portions)

            # create datasets
            self.datasets[DatasetName.trainset.name] = Dataset(trainset, minhash_dataset=minhash_dataset)
            self.datasets[DatasetName.trainset.name].save(f"{self._get_model_path()}/{DatasetName.trainset.name}")
            self.datasets[DatasetName.validset.name] = Dataset(
                mapping=validset,
                labels=self.datasets[DatasetName.trainset.name].get_labels(),
                minhash_dataset=minhash_dataset
            )
            self.datasets[DatasetName.validset.name].save(f"{self._get_model_path()}/{DatasetName.validset.name}")
            self.datasets[DatasetName.testset.name] = Dataset(
                testset,
                labels=self.datasets[DatasetName.trainset.name].get_labels(),
                minhash_dataset=minhash_dataset
            )
            self.datasets[DatasetName.testset.name].save(f"{self._get_model_path()}/{DatasetName.testset.name}")


    def set_coverage(self, coverage: int):
        # refresh datasets, to force the creation of the tf.graph
        self._load_datasets()
        
        # then change the coverage
        for dataset in self.datasets:
            self.datasets[dataset].set_coverage(coverage)
            self.datasets[dataset].save(f"{self._get_model_path()}/{dataset}")


    def set_frag_len(self, frag_len: int):
            # refresh datasets, to force the creation of the tf.graph
            self._load_datasets()
            
            # then change the coverage
            for dataset in self.datasets:
                self.datasets[dataset].set_frag_length(frag_len)    
                self.datasets[dataset].save(f"{self._get_model_path()}/{dataset}")


    def add_examples(self, dataset, portions):
        # refresh datasets, to force the creation of the tf.graph
        self._load_datasets()

        # add examples according to the separation portions
        trainset, validset, testset = self._create_example_label_mapping(dataset, portions)
        self.datasets[DatasetName.trainset.name].add_examples(trainset)
        self.datasets[DatasetName.validset.name].add_examples(validset)
        self.datasets[DatasetName.testset.name].add_examples(testset)

        # save the datasets
        self.datasets[DatasetName.trainset.name].save(f"{self._get_model_path()}/{DatasetName.trainset.name}")
        self.datasets[DatasetName.validset.name].save(f"{self._get_model_path()}/{DatasetName.validset.name}")
        self.datasets[DatasetName.testset.name].save(f"{self._get_model_path()}/{DatasetName.testset.name}")



    def add_ml_model(
        self,
        ml_model_type: str,
        name: str,
        hps: Dict = {},
    ):
        os.makedirs(self._get_ml_models_path(), exist_ok=True)
        
        # check if exists and return an appropriate error
        if self._ml_model_exists(name):
            raise Exception(f"ML model {name} already exists.")
        
        os.makedirs(self._get_ml_model_path(name, ml_model_type), exist_ok=True)

        # prepare hps for the model
        base_hps = {
            "labels": self.datasets[DatasetName.trainset.name].get_number_of_labels(),
            "d_model": self.datasets[DatasetName.trainset.name].get_frag_length(),
        }
        for key, value in hps.items():
            base_hps[key] = value

        # create the ml_model (creation also saves the model)
        ml_model = get_ml_model_constructor(ml_model_type)(
            dirpath=self._get_ml_model_path(name, ml_model_type),
            hps=base_hps,
            name=name
        )

        # add to the dictionary of ml_models
        self.ml_models[name] = ml_model
        

    def change_ml_hps(
        self,
        name,
        hps: Dict
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")
        
        if name not in self.ml_models:
            # load the model
            self._load_ml_model(name)
        
        self.ml_models[name].change_hps(hps)


    def remove_ml_model(
        self,
        name: str
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")
        
        if name in self.ml_models:
            del self.ml_models[name]
        
        shutil.rmtree(self._get_ml_model_path(name, self._get_model_type(name)))


    def list_ml_models(
        self
    ):
        if not os.path.exists(self._get_ml_models_path()):
            return []
        return [dirname.split("-")[0] for dirname in os.listdir(self._get_ml_models_path())]


    def save_ml_model(
        self,
        name
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")

        if name not in self.ml_models:
            return
        
        self.ml_models[name].save()
    

    def train(
        self,
        name: str,
        epochs: int
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")
        
        if not name in self.ml_models:
            # load the model
            self._load_ml_model(name)

        self.ml_models[name].train(
            epochs=epochs,
            trainset=self.datasets[DatasetName.trainset.name],
            trainset_size=self.datasets[DatasetName.trainset.name].get_size(),
            validset=self.datasets[DatasetName.validset.name],
            validset_size=self.datasets[DatasetName.validset.name].get_size()
        )


    def evaluate(
        self,
        name,
        dataset_name = DatasetName.validset.name
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")
        
        if not name in self.ml_models:
            # load the model
            self._load_ml_model(name)

        self.set_frag_len(self.ml_models[name].get_hps().get("d_model"))
            
        return self.ml_models[name].evaluate(
            self.datasets[dataset_name],
            self.datasets[dataset_name].get_size()
        )


    def test(
        self,
        name
    ):
        return self.evaluate(name, dataset_name=DatasetName.testset.name)


    def predict(
        self,
        name,
        fasta_dirpath: str
    ):
        pass


    def transfer(
        self,
        transfer_from_name,
        transfer_to_name,
    ) -> List[str]:
        if not self._ml_model_exists(transfer_from_name):
            raise Exception(f"ML model {transfer_from_name} does not exist.")
        if not self._ml_model_exists(transfer_to_name):
            raise Exception(f"ML model {transfer_to_name} does not exist.")
        
        self._load_ml_model(transfer_from_name)
        self._load_ml_model(transfer_to_name)

        return self.ml_models[transfer_to_name].transfer(self.ml_models[transfer_from_name])
    ############################
    #### private functions #####
    ############################

    def _createModelState(
        self,
        name,
        data_dir
    ):
        self.name = name
        self.data_dir = data_dir

        # Will be updated together when create_datasets() or _load_datasets() are called.
        self.datasets: Dict[DatasetName, tf.data.Dataset] = {}
        self.ml_models: Dict[str, MLModel] = {}

    
    def _loadModelState(
        self
    ):
        self._load_datasets() # Will fill self.datasets


    # def _create_dataset(
    #     self,
    #     mapping: List[Tuple[Label, List[Filepath]]],
    #     dataset_name: str
    # ):
    #     # Create and save the mapping.
    #     print(f"Creating dataset {dataset_name}...")
    #     self.mappings[dataset_name] = []
    #     json_obj = {}
    #     for label, accessions in mapping:
    #         json_obj[label] = accessions
    #         for accession in accessions:
    #             self.mappings[dataset_name].append((accession, label))

    #     random.shuffle(self.mappings[dataset_name]) # should be discarded and be handled in the dataset

    #     # Create the dataset itself
    #     dataset = Dataset()
    #     dataset.set_labels(list(json_obj.keys()))
    #     self.datasets[dataset_name] = dataset.get_tf_dataset(self.mappings[dataset_name])
        
    #     # Persist the mapping to disk
    #     with open(self._get_dataset_json_filename(dataset_name), "w") as f:
    #         f.write(json.dumps(json_obj))


    def _load_datasets(
        self
    ):
        for dataset_name in [DatasetName.trainset.name, DatasetName.validset.name, DatasetName.testset.name]:
            self.datasets[dataset_name] = load_dataset(f"{self._get_model_path()}/{dataset_name}")


    def _create_example_label_mapping(self, dataset, portions):
        # create trainset, validset and testset according to seperation portions
        trainset = []
        validset = []
        testset = []
        for label, accessions in dataset:
            # shuffle accessions randomly
            random.shuffle(accessions)

            # figure indexes separating trainset, validset and testset
            accessions_for_validset_idx = int(len(accessions) * portions[DatasetName.trainset.name])
            accessions_for_validset = int(len(accessions) * portions[DatasetName.validset.name])
            accessions_for_testset_idx = accessions_for_validset_idx + accessions_for_validset

            # create trainset, validset and testset
            for accession_idx, accession in enumerate(accessions):
                if accession_idx < accessions_for_validset_idx:
                    trainset.append((accession, label))
                elif accession_idx < accessions_for_testset_idx:
                    validset.append((accession, label))
                else:
                    testset.append((accession, label))
        return trainset, validset, testset


    def _get_model_path(
        self,
    ):
        return f"{self.data_dir}/{self.name}"
    

    def _get_ml_model_path(self, name, model_type):
        return f"{self._get_ml_models_path()}/{name}-{model_type}"


    def _load_ml_model(
        self,
        name
    ):
        if not self._ml_model_exists(name):
            raise Exception(f"ML model {name} does not exist.")
        
        model_type = self._get_model_type(name)
        self.ml_models[name] = get_ml_model_constructor(model_type)(
            dirpath=self._get_ml_model_path(name, model_type),
            name=name,
            load=True
        )
    
    
    def _get_ml_models_path(self):
        return f"{self._get_model_path()}/ml_models/"


    def _ml_model_exists(self, name):
        for ml_model_dirname in os.listdir(self._get_ml_models_path()):
            if ml_model_dirname.split("-")[0] == name:
                return True
        return False


    def _get_model_type(self, name):
        for ml_model_dirname in os.listdir(self._get_ml_models_path()):
            if ml_model_dirname.split("-")[0] == name:
                return ml_model_dirname.split("-")[1]
        return None


def load_model(name, data_dir = "./data") -> Model:
    if os.path.exists(f"{data_dir}/{name}"):
        return Model(name, load=True)
    else:
        raise Exception(f"Model {name} does not exist.")


def remove_model(name, data_dir = "./data") -> None:
    # delete the folder containing the model
    os.system(f"rm -rf {data_dir}/{name}")
    