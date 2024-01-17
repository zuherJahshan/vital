import tensorflow as tf
from hps import HPs
from ml_model_structure import MLModelStructure

class ConvNextStructure(MLModelStructure):
    def __init__(
        self,
        hps: HPs,
    ):
        super(ConvNextStructure, self).__init__()
        self.hps = hps
        self.embedding = tf.keras.layers.Dense(
            units=3,
            activation="relu",
            kernel_initializer=self.hps.get(HPs.attributes.initializer.name),
            kernel_regularizer=self.hps.get(HPs.attributes.regularizer.name),
        )
        self.convnet = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=True,
            input_shape=[hps.get("seq_len"), hps.get("d_model"), 3],
            weights=None,
            pooling="avg",
            classes=hps.get("labels"),
            classifier_activation="softmax",
        )

    def compile(self, optimizer, loss, metrics):
        super(ConvNextStructure, self).compile(optimizer=optimizer, loss=loss)
        self.own_metrics = []
        for metric in metrics:
            # transform metric name from snake_case to PascalCase
            pascal_metric = "".join([word.capitalize() for word in metric.split("_")])
            try:
                metric_func = getattr(tf.keras.metrics, pascal_metric)
                self.own_metrics.append(metric_func(name=metric))
            except:
                pass



    @property
    def metrics(self):
        return self.own_metrics + super(ConvNextStructure, self).metrics


    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            embedding = self.embedding(inputs, training=True)
            predictions = self.convnet(embedding, training=True)
            loss = self.compute_loss(y=labels, y_pred=predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        #transform predictions to labels
        predictions = tf.one_hot(tf.argmax(predictions, axis=1), depth=labels.shape[1])
        
        return self.compute_metrics(inputs, labels, predictions, None)


    def call(self, inputs):
        return self.convnet(self.embedding(inputs))
    

    def get_weights(self):
        return self.convnet.get_weights() + self.embedding.get_weights()


    def set_weights(self, weights):
        self.convnet.set_weights(weights[:len(self.convnet.get_weights())])
        self.embedding.set_weights(weights[len(self.convnet.get_weights()):])


    @staticmethod
    def get_hps():
        return {
            "labels": "required",
            "seq_len": "required",
            "d_model": "required",
        }
    

    def compute_metrics(self, x, y, y_pred, sample_weight):

    # This super call updates `self.compiled_metrics` and returns
    # results for all metrics listed in `self.metrics`.
        metric_results = super(ConvNextStructure, self).compute_metrics(x, y, y_pred, sample_weight)

        # Note that `self.custom_metric` is not listed in `self.metrics`.
        y_pred_aligned = tf.one_hot(tf.argmax(y_pred, axis=1), depth=y.shape[1])
        for metric in self.own_metrics:
            if metric.name == "loss":
                loss = self.compute_loss(y=y, y_pred=y_pred)
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred_aligned, sample_weight)
            metric_results[metric.name] = metric.result()
        return metric_results