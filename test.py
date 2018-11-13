import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, GRU, TimeDistributed, Dense

import data
import train

def main():
    UNITS = 512
    BATCH_SIZE = 1
    model = keras.models.Sequential()
    model.add(GRU(UNITS, return_sequences=True, input_shape=(None, train.INPUT_FEATURES), name="gru_input"))
    model.add(GRU(UNITS, return_sequences=True, name="gru_2"))
    model.add(TimeDistributed(Dense(train.NUM_PHONEMES + 1), name="output"))

    TRAINING_DATA_PATH = "C:\\Temp\\cache_test_2"
    VALIDATION_DATA_PATH = "C:\\Temp\\cache_test_2"
    trainer = train.Trainer(model, BATCH_SIZE, TRAINING_DATA_PATH, VALIDATION_DATA_PATH)

    CHECKPOINT_PATH = "C:\\Temp\checkpoints"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH, verbose=1, save_weights_only=False, period=1
    )

    EPOCHS = 10
    trainer.train(EPOCHS, True, [checkpoint_callback])
    

if __name__ == "__main__":
    main()