import tensorflow as tf
from hps import HPs
from abc import abstractmethod

class MLModelStructure(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLModelStructure, self).__init__(**kwargs)
        self.grad_accum_steps = tf.Variable(0, trainable=False)
        

    def compile(self, optimizer, loss, metrics):
        super(MLModelStructure, self).compile(optimizer=optimizer, loss=loss)
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
        return self.own_metrics + super(MLModelStructure, self).metrics


    def build(self, input_shape):
        super(MLModelStructure, self).build(input_shape)
        self.init_grad_accum()


    def init_grad_accum(self):
        self.accumulated_grads = [
            tf.Variable(
                tf.zeros_like(
                    self.trainable_variables[i]
                ), trainable=False,

            ) for i in range(len(self.trainable_variables))
        ]


    @tf.function
    def reset_grad_accum(self):
        self.grad_accum_steps.assign(0)
        for i in range(len(self.trainable_variables)):
            self.accumulated_grads[i].assign(tf.zeros_like(self.trainable_variables[i]))


    @tf.function
    def update_grad_accum(self, gradients):
        accum_steps = int(self.hps.get(HPs.attributes.batch_size.name) / self.hps.get(HPs.attributes.mini_batch_size.name))
        frac = 1 / accum_steps
        for i in range(len(self.trainable_variables)):
            self.accumulated_grads[i].assign_add(frac * gradients[i])
        self.grad_accum_steps.assign_add(1)
        
        # Function that actually updates the weights
        def update_weights():
            self.optimizer.apply_gradients(zip(self.accumulated_grads, self.trainable_variables))
            self.reset_grad_accum()

        # when it reaches the number of accumulated gradients, update the weights
        tf.cond(
            self.grad_accum_steps == accum_steps,
            update_weights,
            lambda: None
        )


    def train_step(self, data):
        inputs, labels = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compute_loss(y=labels, y_pred=predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.update_grad_accum(gradients)
        #transform predictions to labels
        predictions = tf.one_hot(tf.argmax(predictions, axis=1), depth=labels.shape[1])
        
        return self.compute_metrics(inputs, labels, predictions, None)

    def compute_metrics(self, x, y, y_pred, sample_weight):

        # This super call updates `self.compiled_metrics` and returns
        # results for all metrics listed in `self.metrics`.
        metric_results = super(MLModelStructure, self).compute_metrics(x, y, y_pred, sample_weight)

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

    @abstractmethod
    def call(self, inputs):
        pass


    @staticmethod
    @abstractmethod
    def get_hps() -> dict:
        pass