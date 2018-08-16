# I'm just happy that everything here works as of 08/09/2018 11:14:28

#TODONE: Create a class that feeds audio in batches, dividing by authors or by transcripts
#TODONE: Switch the variables to private
#TODONE: If there are no wav files to available, try to convert all non-txt files to wav (or use a find_audio_files method perhaps)

from os import listdir
from os.path import isdir, join
from pydub import AudioSegment
import soundfile as sf
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
from python_speech_features import mfcc
from sklearn.preprocessing import minmax_scale


# constants
PRE_EMPHASIS = 0.97 # Value of the amplification filter applied to high frequencies of the audio
FRAME_SIZE = 0.025 # Size of the audio window in ms
FRAME_STRIDE = 0.01 # Step between audio windows in ms (notice that it's shorter than the frame_size, resulting in overlap)
nfft = 512 # Not sure what this is, but it stands for the N in N-point FFT
NFILT = 40 # Number of triangular filters to be applied on FFT, creating a Mel-scale
NUM_CEPS = 12 # Number of cepstral coefficients to retain after compression using Discrete Consine Transform (DCT)
CEP_LIFTER = 22
AUDIO_ORIGIN_FORMAT = "flac"
AUDIO_TARGET_FORMAT = "wav"

class AudioPrep:

    def __init__(self, path, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, cep_lifter = None):
        self.__path = path
        self.__pre_emphasis = pre_emphasis if pre_emphasis is not None else PRE_EMPHASIS
        self.__frame_size = frame_size if frame_size is not None else FRAME_SIZE
        self.__frame_stride = frame_stride if frame_stride is not None else FRAME_STRIDE
        self.__NFFT = NFFT if NFFT is not None else nfft
        self.__nfilt = nfilt if nfilt is not None else NFILT 
        self.__num_ceps = num_ceps if num_ceps is not None else NUM_CEPS
        self.__cep_lifter = cep_lifter if cep_lifter is not None else CEP_LIFTER
        self.__origin_format = None
        self.__target_format = None
        self.__get_files()
        
    def __get_files(self):
        self.__files = dict()
        author_folders = [d for d in listdir(self.__path) if isdir(join(self.__path, d))]
        for author in author_folders:
            self.__files[author] = dict()
            author_path = join(self.__path, author)
            print ("1: ", author_path)
            audio_folders = [d for d in listdir(author_path) if isdir(join(author_path, d))]
            for audio in audio_folders:
                self.__files[author][audio] = list()
                curr_path = join(author_path, audio)
                print ("2:", curr_path)
                transcript_file = [file for file in listdir(curr_path) if file.endswith('.txt')][0]
                transcript_lines = [line.rstrip('\n') for line in open(join(curr_path, transcript_file))]
                for line in transcript_lines:
                    split = line.find(' ')
                    self.__files[author][audio].append(line[:split])
                    #self.files.append(line[:split])
        self.__index = -1
        self.__author_indexes = [key for key in self.__files.keys()]

    def __get_mfcc (self, audios):
        keys = list()
        mfccs = list()

        for key, value in audios.items():
            audio = value[0]
            samplerate = value[1]
            audio_mfcc = mfcc(audio, samplerate)
            keys.append(key)
            mfccs.append(audio_mfcc)

        return (keys, mfccs)

    def __scale_data (self, keys, mfccs):
        #print(mfccs)
        #audios = [minmax_scale(mfcc, feature_range=(-1,1), axis = 1) for mfcc in mfccs]
        audios = list()
        for mfcc in mfccs:
            scaled_audio = minmax_scale(mfcc, feature_range=(-1,1), axis = 1)
            #print(mfcc)
            #print(scaled_audio)
            audios.append(scaled_audio)

        scaled_mfcc = dict()

        for i in range(len(keys)):
            scaled_mfcc[keys[i]] = audios[i]
        
        print (scaled_mfcc)
        
        return scaled_mfcc

    def __convert_audios(self, path):
        if self.__origin_format is None:
            self.__origin_format = AUDIO_ORIGIN_FORMAT
        if self.__target_format is None:
            self.__target_format = AUDIO_TARGET_FORMAT
        audio_parts = [d for d in listdir(path) if d.endswith('.' + self.__origin_format)]
        for audio_part in audio_parts:                
            print("3: ",join(path, audio_part))
            audio = AudioSegment.from_file(join(path, audio_part), self.__origin_format)
            audio.export(join(path, audio_part).split('.')[0] + '.' + self.__target_format, format = self.__target_format)

    def get_data(self, format = "wav"):
        self.__index += 1

        author = self.__author_indexes[self.__index]
        author_path = join(self.__path, author)
        transcripts = dict()
        audios = dict()
        for audio in self.__files[author]:
            curr_path = join(author_path, audio)            

            transcript_file = [file for file in listdir(curr_path) if file.endswith('.txt')][0]
            transcript_lines = [line.rstrip('\n') for line in open(join(curr_path, transcript_file))]
            for line in transcript_lines:
                split = line.find(' ')
                transcripts[line[:split]] = line[split:]

            audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + format)]
            if not audio_parts:
                if self.__target_format is None:
                    self.__target_format = format
                self.__convert_audios(curr_path)            
            audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + format)]
            for audio_part in audio_parts:                
                data, samplerate = sf.read(join(curr_path, audio_part))
                audios[audio_part.split('.')[0]] = (data, samplerate)
        
        print (curr_path)
        keys, mfcc = self.__get_mfcc(audios)
        scaled_mfcc = self.__scale_data(keys, mfcc)
        result = dict()
        for key, transcript in transcripts.items():
            result[key] = (transcript, scaled_mfcc[key])
        return result    

    def convert_audios(self, origin_format = None, target_format = None):
        self.__origin_format = origin_format if origin_format is not None else AUDIO_ORIGIN_FORMAT
        self.__target_format = target_format if target_format is not None else AUDIO_TARGET_FORMAT
