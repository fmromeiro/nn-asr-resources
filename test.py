import data
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import sequence

def main():
    prep = data.AudioPrep("C:\\Users\\u16187\\Desktop\\TCC\\OpenSLR\\dev-clean\\LibriSpeech\\dev-clean")
    batch = prep.get_data()
    
    audios = [value[1] for value in batch.values()]

    max_size = max(audios, key=lambda a: a.shape[0]).shape[0]
    features = audios[0].shape[1]
    padded_audios = []
    for pair in audios:
        result = np.full((max_size, features), 2)
        result[:pair.shape[0], :features] = pair
        padded_audios.append(result)
    
    padded_audios = np.array(padded_audios)
    print(padded_audios.shape)

if __name__ == "__main__":
    main()