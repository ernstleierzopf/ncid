import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
import cipherTypeDetection.config as config
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding


class EnsembleModel:
    def __init__(self, models, architectures, strategy):
        self.models = models
        self.architectures = architectures
        self.strategy = strategy
        self.load_model()

    def load_model(self):
        for j in range(len(self.models)):
            if self.architectures[j] in ("FFNN", "CNN", "LSTM", "Transformer"):
                if self.architectures[j] == 'Transformer':
                    model_ = tf.keras.models.load_model(self.models[j], custom_objects={
                        'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
                        'TransformerBlock': TransformerBlock})
                else:
                    model_ = tf.keras.models.load_model(self.models[j])
                optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon,
                                 amsgrad=config.amsgrad)
                model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                               metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
                self.models[j] = model_
            else:
                with open(self.models[j], "rb") as f:
                    self.models[j] = pickle.load(f)

    def evaluate(self, batch, batch_ciphertexts, labels, batch_size, verbose=0):
        correct_all = 0
        prediction = self.predict(batch, batch_ciphertexts, batch_size, verbose=0)
        for i in range(0, len(prediction)):
            if labels[i] == np.argmax(prediction[i]):
                correct_all += 1
        if verbose == 1:
            print("Accuracy: %f" % (correct_all / len(prediction)))
        return correct_all / len(prediction)

    def predict(self, batch, ciphertexts, batch_size, verbose=0):
        results = []
        for i in range(len(self.models)):
            if self.architectures[i] == "FFNN":
                results.append(self.models[i].predict(batch, batch_size=batch_size, verbose=verbose))
            elif self.architectures[i] in ("CNN", "LSTM", "Transformer"):
                results.append(self.models[i].predict(ciphertexts, batch_size=batch_size, verbose=verbose))
            elif self.architectures[i] in ("DT", "NB", "RF", "ET"):
                results.append(self.models[i].predict_proba(batch))
        if self.strategy == 'mean':
            res = [0.] * len(results[0])
            for result in results:
                for i in range(len(result)):
                    res[i] += result[i]
            for i in range(len(results[0])):
                res[i] = res[i] / len(results)
            return res
