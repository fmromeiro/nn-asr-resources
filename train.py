import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda, Input
from data import AudioPrep
import numpy as np

NUM_PHONEMES = 44
INPUT_FEATURES = 13

def _dummy_loss(y_true, y_pred):
    """As the loss is being computed inside a Lambda layer of the model,
    that returns the already computed loss, this dummy_loss returns that same loss"""
    return y_pred

def _ctc_loss(args):
    y_true, y_predicted, label_length, input_length = args
    """The actual CTC loss, used inside a Lambda layer of a wrapper model"""
    return keras.backend.ctc_batch_cost(y_true, y_predicted, input_length, label_length)

class Trainer:
    """ Abstraction of the training process for a Keras acoustic model for automated speech recognition"""
    
    """The default optimizer used in training"""
    DEFAULT_OPTIMIZER = keras.optimizers.SGD()

    def __init__(self, model, training_data_path, validation_data_path, masking_value = 2):
        """Class constructor

        Initializes class attributes.

        Args:
            model: the Keras model which will be trained by this instance
            training_data_path: the path to the training audio and transcription data
            validation_data_path: the path to the validation audio and transcription data
            masking_value: the value used by the model's mask layer for padded sequences
        """
        self._model = model
        self._training_path = training_data_path
        self._validation_path = validation_data_path
        self._mask = masking_value
        self._built = False

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

        self._built = True

    def train(self, epochs, callbacks=None, optimizer=DEFAULT_OPTIMIZER):
        """Trains the model.

        Args:
            epochs: the number of epochs of training.
            optimizer: Defaults to SGD. the optimizer to be used during training.
            sliding_windows: Defaults to False. determines whether or not to slice the batch into windows.
            window_size: Defaults to 20. the size of the window to be used if sliding_windows is set to True
        """
        self._build_wrapper_model(optimizer)
        
        train_set = AudioPrep(self._training_path)
        validation_set = AudioPrep(self._validation_path)

        self._wrapper_model.fit_generator(generator=train_set.batch_generator(self._mask, 0),
                                          steps_per_epoch=train_set.batch_count,
                                          validation_data=validation_set.batch_generator(),
                                          validation_steps=validation_set.batch_count,
                                          epochs=epochs,
                                          shuffle=False,
                                          callbacks=callbacks)
