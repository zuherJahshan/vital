import tensorflow as tf
import os
import json
from typing import Dict, List, Tuple, Callable
import h5py
import numpy as np
from hps import HPs
from vit import VitStructure
from irene import IreneStructure
from sandy import SandyStructure
from ml_model_structure import MLModelStructure
from enum import Enum


__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../../dataset/")
from dataset import Dataset

os.chdir(f"{__ORIG_WD__}/../../utils/")
from utils import print_progress_bar

os.chdir(__ORIG_WD__)


ml_model_structures = {
    "VitStructure": VitStructure,
    "IreneStructure": IreneStructure,
    "SandyStructure": SandyStructure,
}

class CopyTrainable(Enum):
    COPY = 0
    DONT_COPY = 1
    ALL_TRAINABLE = 2
    NONE_TRAINABLE = 3


def get_structure(structure_name: str) -> MLModelStructure:
    if structure_name in ml_model_structures:
        return ml_model_structures[structure_name]
    raise Exception(f"structure_name {structure_name} is not supported")


MetricName = str
MetricComparisonFunc = Callable[[float, float], bool]
ModelViews = List[Tuple[MetricName, MetricComparisonFunc]]


class MLModel(object):
    def __init__(
        self,
        dirpath: str,
        name: str,
        hps: HPs = None,
        load: bool = False
    ):
        if not load and not hps:
            raise Exception("hps must be provided if load is False")
        self._create_state(dirpath, name, hps)
        if self._ml_model_exists():
            self._load_state()
        else:
            self._create_ml_model()
            self.save()


    @staticmethod
    def get_structure_hps(ml_model_structure_name: str):
        if ml_model_structure_name in ml_model_structures:
            return ml_model_structures[ml_model_structure_name].get_hps()
        raise Exception(f"ml_model_structure_name {ml_model_structure_name} is not supported")


    def get_hps(self):
        return self.hps.get_dict()


    def get_weights_views(self):
        # get all weights views that is a metric in the hps
        weights_views = ["loss"]
        for metric in self.hps.get(HPs.attributes.metrics.name):
            # check if the metric is part of the model views
            if metric in self.model_views:
                weights_views.append(metric)
        return weights_views


    def save(self, weights_view: str = "manual", weights_view_value: float = 0):
        os.makedirs(self._get_ml_model_path(), exist_ok=True)

        # save the hyperparams
        self.hps.save_hps(self._get_ml_model_path())

        # save layers configuration
        self._save_layers_configuration()

        # save the weights
        self._save_weights(weights_view=weights_view, new_value=weights_view_value)

        self._save_ml_model_history()


    def record_metric(
        self,
        epoch: int,
        metric: str,
        value: float
    ):
        if metric in self.ml_model_history["metrics"]:
            self.ml_model_history["metrics"][metric].append([epoch, value])
        else:
            self.ml_model_history["metrics"][metric] = [[epoch, value]]


    def record_hps_change(
        self,
        hps: Dict
    ):
        self.ml_model_history["hps_changes"].append({
            self.ml_model_history["current_epoch"]: hps
        })


    def record_transfer(
        self,
        name: str,
    ):
        self.ml_model_history["transfers"].append({
            self.ml_model_history["current_epoch"]: name
        })


    def increment_epoch_and_save_history(self):
        self.ml_model_history["current_epoch"] += 1


    def get_name(self):
        return self.name


    def get_net(self):
        return self.net


    def transfer(
        self,
        other,
        copied_trainable: bool = CopyTrainable.NONE_TRAINABLE
    ):
        self._transfer(other.get_net(), copied_trainable)
        self.record_transfer(other.get_name())


    def change_hps(
        self,
        hps: Dict
    ):
        # save a sanpshot of the old model
        old_net = self.net
        
        # create a new model with the new hps

        self.hps.set_hps(hps)
        self._create_ml_model()

        # transfer the weights from the old model to the new one
        self._transfer(old_net, copied_trainable=CopyTrainable.COPY)
        self.record_hps_change(hps)
        


    def get_layers_by_names(
        self,
    ):
        return {layer.name: layer for layer in self.net.layers}


    def train(
        self,
        epochs: int,
        trainset: Dataset,
        shuffle_buffer_size: int = 2048,
        validset: Dataset = None,
    ):
        if validset is None:
            return self.net.fit(
                trainset.get_tf_dataset(shuffle_buffer_size=shuffle_buffer_size),
                batch_size=self.hps.get(HPs.attributes.batch_size.name),
                epochs=epochs,
                steps_per_epoch=int(trainset.get_size() / trainset.get_batch_size()),
                callbacks=[MLModelSaveCallback(self)]
            )
        else:
            return self.net.fit(
                trainset.get_tf_dataset(shuffle_buffer_size=shuffle_buffer_size),
                epochs=epochs,
                steps_per_epoch=int(trainset.get_size() / trainset.get_batch_size()),
                validation_data=validset.get_tf_dataset(repeats=1),
                callbacks=[MLModelSaveCallback(self)]
            )


    def evaluate(
        self,
        dataset: tf.data.Dataset,
    ):
        return self.net.evaluate(
            dataset.get_tf_dataset(
                repeats=1
            ),
        )
    

    def predict(
        self,
        dataset: tf.data.Dataset,
    ):
        results = []
        for examples_batch_id, examples_batch in dataset.get_tf_dataset(no_labels=True):
            predictions = self.net.predict_on_batch(examples_batch)
            for i, prediction in enumerate(predictions):
                example_id = examples_batch_id[i].numpy().decode("utf-8")
                results.append((example_id, prediction))
        return results


    def get_model_summary(self):
        return self.net.summary()

    ############################
    #### private functions #####
    ############################
    def _create_state(
        self,
        dirpath: str,
        name: str,
        hps: Dict
    ):
        self.name       : str       = name
        self.dirpath    : str       = dirpath
        
        self.model_views: ModelViews = {
            "loss": lambda old, new: new < old,
            "accuracy": lambda old, new: new > old,
            "manual": lambda old, new: True,
        }
        self.model_views_ranking = [
            "loss",
            "accuracy",
            "manual",
        ]

        if not hps:
            self.hps                = None
        else:
            self.hps    : HPs    = HPs(hps)

        self.ml_model_history: Dict = {
            "current_epoch": 0,
            "metrics": {},
            "hps_changes": [],
            "transfers": [],
        }

        self.layers_config = {
            "layers_trainability": []
        }


    def _create_ml_model(self):
        # create the model
        self.net = get_structure(self.hps.get(HPs.attributes.structure.name))(self.hps)

        # compile the model
        self.net.compile(
            optimizer = self.hps.get(HPs.attributes.optimizer.name),
            loss = self.hps.get(HPs.attributes.loss.name),
            metrics = self.hps.get(HPs.attributes.metrics.name)
        )

        # run dummy example
        self.net(tf.zeros((2, 2, self.hps.get("d_model"), 4)))

        self.layers_config["layers_trainability"] = [layer.trainable for layer in self.net.layers]


    def _load_state(self):
        # load the hyper parameters
        self.hps = HPs(filepath=self._get_ml_model_path())

        # create the model
        self._create_ml_model()

        # load the weights
        self._load_weights()

        self._load_layers_configuration()

        if os.path.exists(self._get_ml_model_history_path()):
            self._load_ml_model_history()


    def _get_ml_model_history_path(self):
        return f"{self._get_ml_model_path()}/history.json"


    def _get_weights_path(self):
        return f"{self._get_ml_model_path()}/weights/"


    def _get_ml_model_path(self):
        return f"{self.dirpath}/{self.name}"
    

    def _get_manual_view_name(self):
        return "manual"


    def _get_model_view_path_and_value(self, model_view: str):
        if not model_view in self.model_views.keys():
            raise Exception(f"model_view must be one of {self.model_views.keys()}")
        
        for model_view_path in os.listdir(self._get_weights_path()):
            current_model_view = model_view_path.split("-")[0]
            value = model_view_path.split("|")[1]
            if current_model_view == model_view:
                return os.path.abspath(f"{self._get_weights_path()}/{model_view_path}"), float(value)
        return None, None


    def _get_model_view_and_value_filename(self, model_view: str, value):
        return "-".join([model_view, f"|{value}|", ".hdf5"])


    def _save_weights(
        self,
        weights_view: str = "manual", new_value=0
    ):
        weights_dir = self._get_weights_path()
        
        # if the weights directory does not exist, create it.
        os.makedirs(weights_dir, exist_ok=True)

        assert weights_view in self.model_views.keys(), f"weights_view must be one of {self.model_views.keys()}"

        save_needed = False
        weights = self.net.get_weights()
        model_view_path, old_value = self._get_model_view_path_and_value(weights_view)
        if model_view_path is not None and self.model_views[weights_view](old_value, new_value):
            # remove old model_view_path
            os.remove(model_view_path)
            save_needed = True

            # create new model_view_path
        if model_view_path is None:
            save_needed = True
        
        if save_needed:
            with h5py.File(f"{weights_dir}/{self._get_model_view_and_value_filename(weights_view, new_value)}", 'w') as f:
                for i, weight in enumerate(weights):
                    f.create_dataset(f'weight_{i}', data=weight)


    def _load_weights(self, weights_view: str = None):
        if weights_view is not None:
            if not weights_view in self.model_views.keys():
                raise Exception(f"weights_view must be one of {self.model_views.keys()}")
            weights_view_path, _ = self._get_model_view_path_and_value(weights_view)
        else:
            # if weights_view is not stated, load the weights as the ranking of the model_views states.
            for weight_view in self.model_views_ranking:
                weights_view_path, _ = self._get_model_view_path_and_value(weight_view)
                if weights_view_path is not None:
                    break
            if weights_view_path is None:
                raise Exception("No weights were found")
            
        with h5py.File(weights_view_path, 'r') as f:
            # order them according to the keys
            weights = [np.array(f[f'weight_{i}'][:]) for i in range(len(f.keys()))]
        self.net.set_weights(weights)


    def _save_ml_model_history(self):
        with open(self._get_ml_model_history_path(), "w") as f:
            f.write(json.dumps(self.ml_model_history))


    def _load_ml_model_history(self):
        with open(self._get_ml_model_history_path(), "r") as f:
            ml_model_history = json.loads(f.read())
        
        for key, value in ml_model_history.items():
            if key in self.ml_model_history:
                self.ml_model_history[key] = value


    def _save_layers_configuration(self):
        with open(f"{self._get_ml_model_path()}/layers_configuration.json", "w") as f:
            f.write(json.dumps(self.layers_config))


    def _load_layers_configuration(self):
        if not os.path.exists(f"{self._get_ml_model_path()}/layers_configuration.json"):
            return
        try:
            with open(f"{self._get_ml_model_path()}/layers_configuration.json", "r") as f:
                config = json.loads(f.read())
        except Exception as e:
            print(f"Failed to load layers_configuration.json: {e}")
            return
        
        # load layers_trainable
        self.layers_config = config
        trainability = self.layers_config["layers_trainability"]
        for i, layer in enumerate(self.net.layers):
            layer.trainable = trainability[i]

               
    def _ml_model_exists(self):
        if os.path.exists(self._get_ml_model_path()):
            return True
        return False


    def _transfer(
        self,
        other,
        copied_trainable: bool
    ):
        transfered_layers = []
        for i, layer in enumerate(other.layers):
            # check if the same kind of layer as the i'th layer of the current model
            if isinstance(layer, self.net.layers[i].__class__):
                # transfer weights
                # check if the weights have the same shape
                if len(layer.get_weights()) > 0 and \
                    not layer.get_weights()[0].shape == self.net.layers[i].get_weights()[0].shape:
                    break

                self.net.layers[i].set_weights(layer.get_weights())
                if copied_trainable == CopyTrainable.ALL_TRAINABLE:
                    self.net.layers[i].trainable = True
                elif copied_trainable == CopyTrainable.NONE_TRAINABLE:
                    self.net.layers[i].trainable = False
                elif copied_trainable == CopyTrainable.COPY:
                    self.net.layers[i].trainable = layer.trainable
                transfered_layers.append(layer.name)
            else:
                break
        trainability = []
        for layer in self.net.layers:
            trainability.append(layer.trainable)
            self.layers_config["layers_trainability"] = trainability
        return transfered_layers
    

class MLModelSaveCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        ml_model: MLModel
    ):
        super(MLModelSaveCallback, self).__init__()
        self.ml_model = ml_model


    def on_epoch_end(self, epoch, logs=None):
        for weights_view in self.ml_model.get_weights_views():
            if not weights_view in logs:
                raise Exception(f"weights_view {weights_view} not found in logs")
            if f"val_{weights_view}" in logs:
                self.ml_model.save(weights_view=weights_view, weights_view_value=logs[f"val_{weights_view}"])
            else:
                self.ml_model.save(weights_view=weights_view, weights_view_value=logs[f"val_{weights_view}"])


class MLModelSaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        ml_model: MLModel
    ):
        super(MLModelSaveHistoryCallback, self).__init__()
        self.ml_model = ml_model


    def on_epoch_end(self, epoch, logs=None):
        for metric in logs.keys():
            self.ml_model.record_metric(epoch, metric, logs[metric])
        self.ml_model.increment_epoch_and_save_history()