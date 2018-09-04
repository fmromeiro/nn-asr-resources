import data
import train
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed
from tensorflow.keras.preprocessing import sequence

def main():
    UNITS = 50
    MASKING_VALUE = 2
    INPUT_FEATURES = 13
    NUM_PHONEMES = 44

    model = keras.models.Sequential()
    model.add(Masking(MASKING_VALUE, input_shape=(None, INPUT_FEATURES)))
    model.add(LSTM(UNITS, return_sequences=True))
    model.add(TimeDistributed(Dense(NUM_PHONEMES + 1)))

    # model.compile(loss='binary_crossentropy', optimizer='adam')
    # print(model.summary())

    trainer = train.Trainer(model, "C:\\Users\\u16187\\Desktop\\TCC\\OpenSLR\\dev-clean\\LibriSpeech\\dev-clean")
    trainer.train(2)

if __name__ == "__main__":
    main()