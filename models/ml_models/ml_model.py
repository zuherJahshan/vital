import tensorflow as tf
from typing import Dict
import os
from abc import ABC, abstractmethod

class MLModel(ABC):
    @abstractmethod
    def __init__(
        self,
        dirpath: str,
        name: str,
        hps: Dict,
    ):
        pass
    
    @abstractmethod
    def save():
        pass


    @abstractmethod
    def get_name():
        pass


    @abstractmethod
    def get_net():
        pass


    @abstractmethod
    def transfer(self, other):
        pass


    @abstractmethod
    def train(
        self,
        epochs: int,
        trainset: tf.data.Dataset,
        trainset_size: int,
        validset: tf.data.Dataset = None,
        validset_size: int = None
    ):
        pass

    
    @abstractmethod
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        dataset_size: int
    ):
        # dataset here comes with the labels
        pass


def exists(dirpath, name):
    if os.path.exists(f"{dirpath}/{name}"):
        return True
    return False