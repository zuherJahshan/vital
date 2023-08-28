import tensorflow as tf
from abc import abstractmethod

class MLModelStructure(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLModelStructure, self).__init__(**kwargs)
        

    @abstractmethod
    def call(self, inputs):
        pass


    @staticmethod
    @abstractmethod
    def get_hps() -> dict:
        pass