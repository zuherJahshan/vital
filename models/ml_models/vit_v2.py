"""
Every class of an ml_model should have the following core functionalities:
    ##### Functions for initialization #####
    1. create(hps) #### ALWAYS
    2. load(dirpath) #### ALWAYS
    3. save(dirpath) #### ALWAYS
    
    ##### Functions for changing network state #####
    4. train(epochs) #### ALWAYS
    5. additional updates to the state that youj may define, like changing predictor head, or deapening the network

    ##### Functions for getting data from the network #####
    6. predict #### ALWAYS
    9. evaluate #### ALWAYS
    7. get_state
    8. print_network
"""


import tensorflow as tf
import os
from enum import Enum
import json
from typing import Dict, List, Tuple, Callable
from ml_model import MLModel
import copy
import h5py
import numpy as np


__ORIG_WD__ = os.getcwd()

os.chdir(f"{__ORIG_WD__}/../../dataset/")
from dataset import Dataset

os.chdir(__ORIG_WD__)


class VitHPs:
    class attributes(Enum):
        encoder_repeats = "encoder_repeats",
        labels = "labels",
        d_model = "d_model",
        d_val = "d_val",
        d_key = "d_key",
        d_ff = "d_ff",
        heads = "heads",
        dropout_rate = "dropout_rate",
        regularizer = "regularizer",
        initializer = "initializer",
        activation = "activation",
        optimizer = "optimizer",
        learning_rate = "learning_rate",
        loss = "loss",
        metrics = "metrics",
        batch_size = "batch_size"


    def __init__(
        self,
        hps = None,
        filepath = None
    ):
        # one of the two must be defined
        if hps is None and filepath is None:
            raise Exception("Either hps or filepath must be defined")
        
        # if filepath is defined, load the hps from the file
        if filepath is not None:
            # check that the file exists
            if not os.path.exists(filepath):
                raise Exception(f"File {filepath} does not exist")
            hps = self._load_hps(filepath)

        if VitHPs.attributes.labels.name in hps and VitHPs.attributes.d_model.name in hps:
            self.hps = self.get_default()
            self.set_hps(hps)
            self.hps_get_behavior = self._get_default_get_attribute_behavior()
        else:
            raise Exception("Model hyper parameters must have \"labels\" and \"d_model\" defined")
        

    def save_hps(self, dirpath):
        with open(f"{dirpath}/hps.csv", "w") as f:
            f.write(json.dumps(self.hps))


    def set_hps(self, hps):
        for key, _ in self.hps.items():
            if key in hps:
                self.hps[key] = hps[key]


    def get(self, attribute):
        return self.hps_get_behavior[attribute](attribute)


    def get_default(self):
        return {
            VitHPs.attributes.encoder_repeats.name: 1,
            VitHPs.attributes.labels.name: None,
            VitHPs.attributes.d_model.name: None,
            VitHPs.attributes.d_val.name: 64,
            VitHPs.attributes.d_key.name: 64,
            VitHPs.attributes.d_ff.name: 1024,
            VitHPs.attributes.heads.name: 8,
            VitHPs.attributes.dropout_rate.name: 0.1,
            VitHPs.attributes.regularizer.name: {
                "name": "l2",
                "params": {
                    "l2": 0.01,
                }
            },
            VitHPs.attributes.initializer.name: "glorot_normal",
            VitHPs.attributes.optimizer.name: {
                "name": "Adam",
                "params": {
                    "learning_rate": 0.001,
                }
            },
            VitHPs.attributes.loss.name: "categorical_crossentropy",
            VitHPs.attributes.metrics.name: ["accuracy", "AUC"],
            VitHPs.attributes.activation.name: "relu",
            VitHPs.attributes.batch_size.name: 32,
        }

    
    def _get_default_get_attribute_behavior(self):
        def default(attribute):
            if attribute in self.hps:
                return self.hps[attribute]
            else:
                raise Exception(f"Attribute {attribute} not found in model hyper parameters")
            
        def get_from_tf_class(attribute, tf_class):
            if attribute in self.hps:
                return getattr(tf_class, self.hps[attribute]["name"])(**self.hps[attribute]["params"])
            else:
                raise Exception(f"Attribute {attribute} not found in model hyper parameters")
            

        return {
            VitHPs.attributes.encoder_repeats.name: default,
            VitHPs.attributes.labels.name: default,
            VitHPs.attributes.d_model.name: default,
            VitHPs.attributes.d_val.name: default,
            VitHPs.attributes.d_key.name: default,
            VitHPs.attributes.d_ff.name: default,
            VitHPs.attributes.heads.name: default,
            VitHPs.attributes.dropout_rate.name: default,
            VitHPs.attributes.regularizer.name: lambda attribute: get_from_tf_class(attribute, tf.keras.regularizers),
            VitHPs.attributes.initializer.name: default,
            VitHPs.attributes.optimizer.name: lambda attribute: get_from_tf_class(attribute, tf.keras.optimizers),
            VitHPs.attributes.loss.name: default,
            VitHPs.attributes.metrics.name: default,
            VitHPs.attributes.activation.name: default,
            VitHPs.attributes.batch_size.name: default,
        }
    

    def _load_hps(self, filepath):
        with open(f"{filepath}/hps.csv", "r") as f:
            return json.loads(f.read())


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        d_val,
        d_key,
        d_ff,
        heads,
        dropout_rate,
        regularizer,
        initializer,
        activation,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.d_val = d_val
        self.d_key = d_key
        self.d_ff = d_ff
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer
        self.initializer = initializer
        self.activation = activation


    def build(self, input_shape):
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization() # Could be combined with the first one
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=self.d_key,
            value_dim=self.d_val,
            dropout=self.dropout_rate,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer
        )
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(
            self.d_ff,
            activation=self.activation,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(
            self.d_model,
            activation="linear",
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
        )


    def call(self, inputs):
        x = self.multihead_attention(inputs, inputs)
        x = self.dropout1(x)
        x = self.layernorm1(x + inputs)
        y = self.dense1(x)
        y = self.dropout2(y)
        y = self.dense2(y)
        return self.layernorm2(x + y)
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "d_model": self.d_model,
            "d_val": self.d_val,
            "d_key": self.d_key,
            "d_ff": self.d_ff,
            "heads": self.heads,
            "dropout_rate": self.dropout_rate,
            "regularizer": self.regularizer,
            "initializer": self.initializer,
            "activation": self.activation,
        })
        return config


class VitStructure(tf.keras.Model):
    def __init__(
        self,
        hps: VitHPs
    ):
        super(VitStructure, self).__init__()
        self.hps = hps

        # define the different netowrk layers
        self.embedding = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.hps.get(VitHPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(VitHPs.attributes.regularizer.name),
        )
        self.encoders = [Encoder(
            d_model=self.hps.get(VitHPs.attributes.d_model.name),
            d_val=self.hps.get(VitHPs.attributes.d_val.name),
            d_key=self.hps.get(VitHPs.attributes.d_key.name),
            d_ff=self.hps.get(VitHPs.attributes.d_ff.name),
            heads=self.hps.get(VitHPs.attributes.heads.name),
            dropout_rate=self.hps.get(VitHPs.attributes.dropout_rate.name),
            regularizer=self.hps.get(VitHPs.attributes.regularizer.name),
            initializer=self.hps.get(VitHPs.attributes.initializer.name),
            activation=self.hps.get(VitHPs.attributes.activation.name),
        ) for _ in range(self.hps.get(VitHPs.attributes.encoder_repeats.name))]
        self.softmax_dense = tf.keras.layers.Dense(
            units=self.hps.get(VitHPs.attributes.labels.name),
            activation="softmax",
            kernel_initializer=self.hps.get(VitHPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(VitHPs.attributes.regularizer.name),
            name="softmax_dense"
        )


    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.squeeze(x, -1)
        for encoder in self.encoders:
            x = encoder(x)
        x  = tf.squeeze(tf.split(x, x.shape[-2], axis=-2)[0], axis=-2)
        return self.softmax_dense(x)
    

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "hps": self.hps,
        })
        return config
 
    
    def from_config(cls, config):
        config['hps'] = VitHPs.from_config(config['hps'])
        return cls(**config)


MetricName = str
MetricComparisonFunc = Callable[[float, float], bool]
ModelViews = List[Tuple[MetricName, MetricComparisonFunc]]

class Vit(MLModel):
    def __init__(
        self,
        dirpath: str,
        name: str,
        hps: Dict = None,
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
        self._transfer(old_net, copied_trainable=True)


    def get_hps(self):
        return self.hps


    def get_weights_views(self):
        # get all weights views that is a metric in the hps
        weights_views = ["loss"]
        for metric in self.hps.get(VitHPs.attributes.metrics.name):
            # check if the metric is part of the model views
            if metric in self.model_views:
                weights_views.append(metric)
        return weights_views


    def save(self, weights_view: str = "manual", weights_view_value: float = 0):
        os.makedirs(self._get_ml_model_path(), exist_ok=True)

        # save the hyperparams
        self.hps.save_hps(self._get_ml_model_path())

        # save the weights
        self._save_weights(weights_view=weights_view, new_value=weights_view_value)


    def get_net(self):
        return self.net


    def transfer(
        self,
        other: MLModel,
        copied_trainable: bool = False
    ):
        self._transfer(other.get_net(), copied_trainable)


    def get_layers_by_names(
        self,
    ):
        return {layer.name: layer for layer in self.net.layers}


    def train(
        self,
        epochs: int,
        trainset: Dataset,
        trainset_size: int,
        shuffle_buffer_size: int = 8192,
        validset: Dataset = None,
        validset_size: int = None,
    ):
        if validset is None:
            return self.net.fit(
                trainset.get_tf_dataset(batch_size=self.hps.get(VitHPs.attributes.batch_size.name), shuffle_buffer_size=shuffle_buffer_size),
                batch_size=self.hps.get(VitHPs.attributes.batch_size.name),
                epochs=epochs,
                steps_per_epoch=int(trainset_size / self.hps.get(VitHPs.attributes.batch_size.name)),
                callbacks=[VitSaveCallback(self)]
            )
        else:
            if validset_size is None:
                raise ValueError("validset_size must be provided if validset is provided")
            return self.net.fit(
                trainset.get_tf_dataset(batch_size=self.hps.get(VitHPs.attributes.batch_size.name), shuffle_buffer_size=shuffle_buffer_size),
                epochs=epochs,
                steps_per_epoch=int(trainset_size / self.hps.get(VitHPs.attributes.batch_size.name)),
                validation_data=validset.get_tf_dataset(repeats=1, batch_size=32),
                callbacks=[VitSaveCallback(self)]
            )


    def evaluate(
        self,
        dataset: tf.data.Dataset,
        dataset_size: int
    ):
        return self.net.evaluate(
            dataset.get_tf_dataset(
                repeats=1,
                batch_size=self.hps.get(VitHPs.attributes.batch_size.name)
            ),
        )


    def predict(
        self,
        dataset: tf.data.Dataset,
    ):
        pass


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
            self.hps    : VitHPs    = VitHPs(hps)


    def _create_ml_model(self):
        # create the model
        self.net = VitStructure(self.hps)

        # compile the model
        self.net.compile(
            optimizer = self.hps.get(VitHPs.attributes.optimizer.name),
            loss = self.hps.get(VitHPs.attributes.loss.name),
            metrics = self.hps.get(VitHPs.attributes.metrics.name)
        )

        # run dummy example
        self.net(tf.zeros((1, 1, self.hps.get(VitHPs.attributes.d_model.name), 4)))


    def _load_state(self):
        # load the hyper parameters
        self.hps = VitHPs(filepath=self._get_ml_model_path())

        # create the model
        self._create_ml_model()

        # load the weights
        self._load_weights()


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

               
    def _ml_model_exists(self):
        if os.path.exists(self._get_ml_model_path()):
            return True
        return False


    def _transfer(
        self,
        other: VitStructure,
        copied_trainable: bool
    ):
        transfered_layers = []
        for i, layer in enumerate(other.layers):
            # check if the same kind of layer as the i'th layer of the current model
            if isinstance(layer, self.net.layers[i].__class__):
                # transfer weights
                # check if the weights have the same shape
                if not layer.get_weights()[0].shape == self.net.layers[i].get_weights()[0].shape:
                    break
                self.net.layers[i].set_weights(layer.get_weights())
                transfered_layers.append(layer.name)
                layer.trainable = copied_trainable
            else:
                break
        return transfered_layers
    

class VitSaveCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        vit: Vit
    ):
        super(VitSaveCallback, self).__init__()
        self.vit = vit


    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} ended, saving weights")
        for weights_view in self.vit.get_weights_views():
            if not weights_view in logs:
                raise Exception(f"weights_view {weights_view} not found in logs")
            print(f"Saving weights for {weights_view}")
            if f"val_{weights_view}" in logs:
                self.vit.save(weights_view=weights_view, weights_view_value=logs[f"val_{weights_view}"])
            else:
                self.vit.save(weights_view=weights_view, weights_view_value=logs[f"val_{weights_view}"])