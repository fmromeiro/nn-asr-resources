import data
import train
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM, Dense, Masking, TimeDistributed
from tensorflow.keras.preprocessing import sequence

def main():
    RAW_DATA_PATH = "C:\\Users\\u16187\\Desktop\\TCC\\OpenSLR\\dev-clean\\LibriSpeech\\dev-clean"
    CACHE_PATH = "C:\\Temp\\cache_test_2\\"
    data.AudioPrep.generate_cache(RAW_DATA_PATH, CACHE_PATH)
    prep = data.AudioPrep(CACHE_PATH)
    for batch in prep.batch_generator():
        print(batch.shape)
    

if __name__ == "__main__":
    main()