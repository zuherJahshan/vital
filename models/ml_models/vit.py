import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure


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
    
    @staticmethod
    def get_hps():
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
        }


class VitStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs
    ):
        super(VitStructure, self).__init__()
        self.hps = hps

        # define the different netowrk layers
        self.embedding = tf.keras.layers.Dense(
            units=1,
            activation="linear",
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
        config['hps'] = HPs.from_config(config['hps'])
        return cls(**config)
    

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
