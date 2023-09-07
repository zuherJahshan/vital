import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure
from vit import Encoder
from typing import Set


import tensorflow as tf

class CustomFlattenLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomFlattenLayer, self).__init__(**kwargs)
        self.flatten_layer = tf.keras.layers.Flatten()


    def call(self, inputs):
        # Get the shape of the input tensor dynamically
        input_shape = tf.shape(inputs)
        b1, b2 = input_shape[0], input_shape[1]

        # Temporarily merge the first two dimensions
        merged_tensor = tf.reshape(inputs, [-1, input_shape[2], input_shape[3]])

        # Flatten using Keras Flatten layer
        flattened_merged_tensor = self.flatten_layer(merged_tensor)

        # Reshape the tensor to separate the two batch dimensions
        flattened_tensor = tf.reshape(flattened_merged_tensor, [b1, b2, -1])
        return flattened_tensor



class CnnBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        activation,
        regularizer,
        initializer,
        dropout_rate
    ):
        super(CnnBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.regularizer = regularizer
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        
        self.conv2d_layers = [tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            padding='same') for _ in range(3)]
        self.layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(3)]


    def call(self, inputs):
        x = inputs
        for i in range(2):
            x = self.conv2d_layers[i](x)
            x = self.layer_norms[i](x)
        y = self.conv2d_layers[2](x)
        y = self.layer_norms[2](y + x)
        y = tf.nn.max_pool(
            y,
            ksize=[1, 1, 2, 1],  # Pooling size: No pooling on batch1, batch2, and the last dimension. Pool every 2 elements in the third dimension.
            strides=[1, 1, 2, 1],  # Stride: Move by 2 elements in the third dimension.
            padding='VALID'
        )
        return y


    @staticmethod
    def get_hps():
        return {
            "filters",
            "kernel_size",
            "activation",
            "regularizer",
            "initializer",
        }


class IreneStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs
    ):
        super(IreneStructure, self).__init__()
        self.hps = hps
        self.embedding = tf.keras.layers.Dense(
            units=1,
            activation="linear",
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
        )
        self.cnn_blocks = [CnnBlock(
            filters=16*(2 ** i),
            kernel_size=3,
            activation=self.hps.get(HPs.attributes.activation.name),
            regularizer=self.hps.get(HPs.attributes.regularizer.name),
            initializer=self.hps.get(HPs.attributes.initializer.name),
            dropout_rate=self.hps.get(HPs.attributes.dropout_rate.name),
        ) for i in range(3)]
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
        x = self.embedding(inputs)
        for i, cnn_block in enumerate(self.cnn_blocks):
            x = cnn_block(x)
        x = self.flatten(x)
        x = self.mlp(x)
        for encoder in self.encoders:
            x = encoder(x)
        x  = tf.squeeze(tf.split(x, x.shape[-2], axis=-2)[0], axis=-2)
        return self.softmax_dense(x)
    

    @staticmethod
    def get_hps() -> Set:
        return {
            "d_model",
            "d_val",
            "d_key",
            "d_ff",
            "heads",
            "dropout_rate",
            "regularizer",
            "initializer",
            "activation",
            "encoder_repeats",
            "labels",
        }

