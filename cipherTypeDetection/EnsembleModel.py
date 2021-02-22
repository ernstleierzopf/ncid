import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import cipherTypeDetection.config as config


class EnsembleModel:
    def __init__(self, models, architectures, strategy):
        self.models = models
        self.architectures = architectures
        self.strategy = strategy

    def load_model(self):
        for j in range(len(self.models)):
            if self.architectures[j] in ("FFNN", "CNN", "LSTM", "Transformer"):
                model_ = tf.keras.models.load_model(self.models[j])
                optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon,
                                 amsgrad=config.amsgrad)
                model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                               metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
                self.models[j] = model_
            else:
                with open(self.models[j], "rb") as f:
                    self.models[j] = pickle.load(f)

    def evaluate(self, batch, labels, batch_size):
        results = []
        for i in range(len(self.models)):
            if self.architectures[i] in ("FFNN", "CNN", "LSTM", "Transformer"):
                results.append(self.models[i].evaluate(batch, labels, batch_size=batch_size))
            elif self.architectures[i] in ("DT", "NB", "RF", "ET"):
                results.append(self.models[i].score(batch, labels))
        

    def predict(self):