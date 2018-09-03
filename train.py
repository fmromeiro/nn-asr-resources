import tensorflow as tf
import tensorflow.keras as keras
from data import AudioPrep
import numpy as np

NUM_PHONEMES = 44

def ctc_loss(y_true, y_predicted):
    return tf.nn.ctc_loss(y_true, y_predicted, NUM_PHONEMES + 1) # Phonemes + epsilon

class Trainer:
    """ Abstraction of the training process for a Keras acoustic model for automated speech recognition"""
    DEFAULT_OPTIMIZER = keras.optimizers.SGD()

    def __init__(self, model, data_path, masking_value = 2):
        """Class constructor

        Initializes class attributes.

        Args:
            model: the Keras model which will be trained by this instance
            data_path: the path to the audio and transcription data
            masking_value: the value used by the model's mask layer for padded sequences
        """
        self.__model = model
        self.__data_path = data_path
        self.__mask = masking_value

    def __get_input_tensor(self, batch):
        """Returns an input tensor of shape (batch_size, timesteps, feature_dims)
        
        Args:
            batch: the batch object returned by AudioPrep.get_data()
        """
        audios = [value[1] for value in batch.values()]
        max_size = max(audios, key=lambda a: a.shape[0]).shape[0]
        features = audios[0].shape[1]
        padded_audios = []
        for pair in audios:
            result = np.full((max_size, features), self.__mask)
            result[:pair.shape[0], :features] = pair
            padded_audios.append(result)
        
        padded_audios = np.array(padded_audios)
        return padded_audios

    def train(self, epochs, optimizer=DEFAULT_OPTIMIZER, sliding_windows=False, window_size=20):
        """Trains the model.

        Args:
            epochs: the number of epochs of training.
            optimizer: Defaults to SGD. the optimizer to be used during training.
            sliding_windows: Defaults to False. determines whether or not to slice the batch into windows.
            window_size: Defaults to 20. the size of the window to be used if sliding_windows is set to True
        """
        self.__model.compile(loss=ctc_loss, optimizer=optimizer, metrics=['acc'])
        
        audio_interface = AudioPrep(self.__data_path)

        for epoch in range(epochs):
            for _ in range(audio_interface.batch_count):
                batch = audio_interface.get_data()
                X = self.__get_input_tensor(batch)
                if sliding_windows:
                    for i in range(len(X)):
                        X[i] = [X[i][j:j+window_size] for j in range(len(X[i]) - window_size + 1)]
                    np.reshape(X, X.shape + (1,))
                y = [value[0] for value in batch.values()]

                self.__model.fit(X, y, batch_size = X.shape[0], shuffle=False, verbose=0)
                self.__model.reset_states()

