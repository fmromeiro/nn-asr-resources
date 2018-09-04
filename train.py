import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda, Input
from data import AudioPrep
import numpy as np

NUM_PHONEMES = 44

def _dummy_loss(y_true, y_pred):
    """As the loss is being computed inside a Lambda layer of the model,
    that returns the already computed loss, this dummy_loss returns that same loss"""
    return y_pred

def _ctc_loss(args):
    y_true, y_predicted, label_length, input_length = args
    """The actual CTC loss, used inside a Lambda layer of a wrapper model"""
    return keras.backend.ctc_batch_cost(y_true, y_predicted, input_length, label_length)

class Trainer:
    LABEL_MASK = 0

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
        self._model = model
        self._data_path = data_path
        self._mask = masking_value

    def _build_wrapper_model(self, optimizer):
        """Builds a wrapper model for CTC loss using a Lambda layer"""
        # actual model's data
        x_input = self._model.layers[0].input # the X training data
        y_pred = self._model.layers[-1].output # the actual predicted output

        # CTC wrapper data
        labels = Input(name='labels', shape=(None,), dtype='int32')
        label_length = Input(name='label_length', shape=(None,), dtype='int32')
        input_length = Input(name='input_length', shape=(None,), dtype='int32')

        loss_layer = Lambda(_ctc_loss, output_shape=(1,), name='ctc')([labels, 
                                                                       y_pred, 
                                                                       label_length, 
                                                                       input_length])

        self._wrapper_model = keras.Model(inputs=[x_input, labels, input_length, label_length], outputs=loss_layer)
        self._wrapper_model.compile(loss=_dummy_loss, optimizer=optimizer, metrics=['acc'])

    def _get_input_tensor(self, batch):
        """Returns a tuple of an 1D tensor containing the lengths of each sample in the batch and
        and the input tensor of shape (batch_size, timesteps, feature_dims)
        
        Args:
            batch: the batch object returned by AudioPrep.get_data()
        """
        audios = [value[1] for value in batch.values()]
        max_size = max(audios, key=lambda a: a.shape[0]).shape[0]
        features = audios[0].shape[1]
        padded_audios = []
        input_lengths = []
        for audio in audios:
            input_lengths.append(audio.shape[0])
            result = np.full((max_size, features), self._mask)
            result[:audio.shape[0], :features] = audio
            padded_audios.append(result)
        
        padded_audios = np.array(padded_audios)
        input_lengths = np.array(input_lengths)
        return input_lengths, padded_audios

    def _get_output_tensor(self, batch):
        """Returns a tuple of an 1D tensor containing the lengths of each sample in the batch and
        and the input tensor of shape (batch_size, timesteps, feature_dims)
        
        Args:
            batch: the batch object returned by AudioPrep.get_data()
        """
        labels = [value[0] for value in batch.values()]
        max_size = len(max(labels, key=lambda label: len(label)))

        label_lengths = []
        padded_labels = []
        for label in labels:
            label_lengths.append(len(label))
            result = np.full((max_size), Trainer.LABEL_MASK)
            result[:len(label)] = label
            padded_labels.append(result)

        padded_labels = np.array(padded_labels)
        label_lengths = np.array(label_lengths)

        return label_lengths, padded_labels

    def train(self, epochs, optimizer=DEFAULT_OPTIMIZER, sliding_windows=False, window_size=20):
        """Trains the model.

        Args:
            epochs: the number of epochs of training.
            optimizer: Defaults to SGD. the optimizer to be used during training.
            sliding_windows: Defaults to False. determines whether or not to slice the batch into windows.
            window_size: Defaults to 20. the size of the window to be used if sliding_windows is set to True
        """
        self._build_wrapper_model(optimizer)
        self._wrapper_model.summary()
        
        audio_interface = AudioPrep(self._data_path)

        for epoch in range(epochs):
            for _ in range(audio_interface.batch_count):
                batch = audio_interface.get_data()
                input_lengths, X = self._get_input_tensor(batch)
                label_lengths, y = self._get_output_tensor(batch)
                if sliding_windows:
                    for i in range(len(X)):
                        X[i] = [X[i][j:j+window_size] for j in range(len(X[i]) - window_size + 1)]
                    np.reshape(X, X.shape + (1,))

                self._wrapper_model.fit([X, y, input_lengths, label_lengths],
                                        y,
                                        batch_size = X.shape[0],
                                        shuffle=False,
                                        verbose=0)
                self._wrapper_model.reset_states()
