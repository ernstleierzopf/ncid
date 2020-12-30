from cipherTypeDetection import config
from tensorflow.keras.callbacks import LearningRateScheduler


def time_based_decay_schedule(iterations, _lr):
    lr = config.learning_rate * (1. / (1. + config.decay * iterations))
    return lr


def custom_step_decay_schedule(drop, lr):
    if drop:
        lr *= (1 - config.drop)
    return lr


class TimeBasedDecayLearningRateScheduler(LearningRateScheduler):
    def __init__(self, train_dataset_size, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = time_based_decay_schedule
        self.train_dataset_size = train_dataset_size
        self.iteration = 0
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        super(TimeBasedDecayLearningRateScheduler, self).on_epoch_begin(self.iteration, logs)

    def on_epoch_end(self, epoch, logs=None):
        super(TimeBasedDecayLearningRateScheduler, self).on_epoch_end(self.iteration, logs)


class CustomStepDecayLearningRateScheduler(LearningRateScheduler):
    def __init__(self, early_stopping_callback, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = custom_step_decay_schedule
        self.early_stopping_callback = early_stopping_callback
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        drop = False
        if self.early_stopping_callback.wait >= 100:
            drop = True
        super(CustomStepDecayLearningRateScheduler, self).on_epoch_begin(drop, logs)

    def on_epoch_end(self, epoch, logs=None):
        super(CustomStepDecayLearningRateScheduler, self).on_epoch_end(epoch, logs)
