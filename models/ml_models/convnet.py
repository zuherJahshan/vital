import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure

class ConvStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs,
    ):
        super(ConvStructure, self).__init__()
        self.hps = hps
        
        try:
            convclass = getattr(tf.keras.applications, hps.get("convclass"))
        except:
            raise Exception(f"Convolutional class {hps.get('convclass')} not found in tf.keras.applications")

        try:
            pascal_name = "".join([word.capitalize() for word in hps.get("convnet").split("_")])
            convnet = getattr(convclass, pascal_name)
            self.convnet = convnet(
                include_top=True,
                input_shape=[hps.get("seq_len"), hps.get("d_model"), 4],
                weights=None,
                pooling="avg",
                classes=hps.get("labels"),
                classifier_activation="softmax",
            )
        except:
            raise Exception(f"Convolutional net {hps.get('convnet')} not found in {hps.get('convclass')}")


    def call(self, inputs):
        return self.convnet(inputs)
    

    def get_weights(self):
        return self.convnet.get_weights()


    def set_weights(self, weights):
        self.convnet.set_weights(weights)


    @staticmethod
    def get_hps():
        return {
            "convclass": "required",
            "convnet": "required",
            "labels": "required",
            "seq_len": "required",
            "d_model": "required",
        }