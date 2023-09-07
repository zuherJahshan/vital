import tensorflow as tf
from vit import Encoder
from hps import HPs
from ml_model_structure import MLModelStructure
from irene import CustomFlattenLayer

class SandyStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs
    ):
        super(SandyStructure, self).__init__()
        self.hps = hps

        # define the different netowrk layers
        self.embeddings = [tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation=self.hps.get(HPs.attributes.activation.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            padding="same",
        ) for i in range(3)]
        self.layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(3)]
        self.flatten = CustomFlattenLayer()
        self.mlp = tf.keras.layers.Dense(
            units=self.hps.get("d_model"),
            activation=self.hps.get(HPs.attributes.activation.name),
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
        )
        self.encoders = [Encoder(
            d_model=self.hps.get("d_model"),
            d_val=self.hps.get("d_val"),
            d_key=self.hps.get("d_key"),
            d_ff=self.hps.get("d_ff"),
            heads=self.hps.get("heads"),
            dropout_rate=self.hps.get(HPs.attributes.dropout_rate.name),
            regularizer=self.hps.get(HPs.attributes.regularizer.name),
            initializer=self.hps.get(HPs.attributes.initializer.name),
            activation=self.hps.get(HPs.attributes.activation.name),
        ) for _ in range(self.hps.get("encoder_repeats"))]
        self.softmax_dense = tf.keras.layers.Dense(
            units=self.hps.get("labels"),
            activation="softmax",
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
            name="softmax_dense"
        )


    def call(self, inputs):
        # Ensure that inputs are float32
        inputs = tf.cast(inputs, dtype=tf.float32)

        # Get the batch shape and sequential lengths
        batch_shape = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]

        # Reshape inputs to 3D tensor to match the input shape that Conv1D expects
        inputs = tf.reshape(inputs, [-1, d_model, 4])

        x = self.embeddings[0](inputs)
        x = self.layer_norms[0](x)
        print(x.shape, flush=True)
        x = self.embeddings[1](x)
        x = self.layer_norms[1](x)
        print(x.shape, flush=True)
        x = self.embeddings[2](x)
        x = self.layer_norms[2](x)
        print(x.shape, flush=True)

        x = tf.reshape(x, [batch_shape, seq_len, d_model, 32])
        print(x.shape, flush=True)
        x = tf.nn.max_pool(
            x,
            ksize=[1, 1, 2, 1],  # Pooling size: No pooling on batch1, batch2, and the last dimension. Pool every 2 elements in the third dimension.
            strides=[1, 1, 2, 1],  # Stride: Move by 2 elements in the third dimension.
            padding='VALID'
        )

        print(x.shape, flush=True)
        x = self.flatten(x)
        print(x.shape, flush=True)
        x = self.mlp(x)
        print(x.shape, flush=True)
        for encoder in self.encoders:
            x = encoder(x)
        print(x.shape, flush=True)
        x  = tf.squeeze(tf.split(x, x.shape[-2], axis=-2)[0], axis=-2)
        print(x.shape, flush=True)
        x = self.softmax_dense(x)
        return x
    

    @staticmethod
    def get_hps() -> dict:
        return {
            "d_model": "required",
            "d_val": "required",
            "d_key": "required",
            "d_ff": "required",
            "heads": "required",
            "dropout_rate": "optional",
            "regularizer": "optional",
            "initializer": "optional",
            "activation": "optional",
            "encoder_repeats": "required",
            "labels": "required",
        }

# class SandyStructure(MLModelStructure):
#     def __init__(self, hps: HPs):
#         super(SandyStructure, self).__init__()
#         self.hps = hps

#         # Define the Conv1D layer with 32 filters
#         self.conv1d = tf.keras.layers.Conv1D(
#             filters=32,  # Setting the number of output filters to 32
#             kernel_size=3,  # Adjust kernel size as needed
#             activation=self.hps.get(HPs.attributes.activation.name),
#             kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
#             kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
#             padding="same",
#         )

#     def call(self, inputs):
#         # Ensure that inputs are float32
#         inputs = tf.cast(inputs, dtype=tf.float32)

#         # Get the batch shape and sequential lengths
#         batch_shape = tf.shape(inputs)[0]
#         seq_len = tf.shape(inputs)[1]
#         d_model = tf.shape(inputs)[2]

#         # Reshape inputs to 3D tensor to match the input shape that Conv1D expects
#         inputs_reshaped = tf.reshape(inputs, [-1, d_model, 4])
        
#         # Apply the Conv1D layer
#         x = self.conv1d(inputs_reshaped)
        
#         # Reshape the output back to a 4D tensor
#         x_reshaped = tf.reshape(x, [batch_shape, seq_len, d_model, 32])
        
#         return x_reshaped


    
#     @staticmethod
#     def get_hps() -> dict:
#         return {
#             "d_model": "required",
#             "d_val": "required",
#             "d_key": "required",
#             "d_ff": "required",
#             "heads": "required",
#             "dropout_rate": "optional",
#             "regularizer": "optional",
#             "initializer": "optional",
#             "activation": "optional",
#             "encoder_repeats": "required",
#             "labels": "required",
#         }

