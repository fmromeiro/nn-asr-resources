from os import listdir
import os
from pydub import AudioSegment
import soundfile as sf
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import python_speech_features
from sklearn.preprocessing import minmax_scale
import beep

# constants
PRE_EMPHASIS = 0.97 # Value of the amplification filter applied to high frequencies of the audio
FRAME_SIZE = 0.025 # Size of the audio window in seconds
FRAME_STRIDE = 0.01 # Step between audio windows in seconds (notice that it's shorter than the frame_size, resulting in overlap)
nfft = 512 # Not sure what this is, but it stands for the N in N-point FFT
NFILT = 40 # Number of triangular filters to be applied on FFT, creating a Mel-scale
NUM_CEPS = 12 # Number of cepstral coefficients to retain after compression using Discrete Consine Transform (DCT)
CEP_LIFTER = 22
AUDIO_ORIGIN_FORMAT = "flac"
AUDIO_TARGET_FORMAT = "wav"

class AudioPrep(object):
    """Get audios, convert them into numpy arrays, do the MFCC transformation, translate the transcriptions into phonemes and make xy pairments."""

    def __init__(self, path, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, cep_lifter = None, dict_path = None, phonemes_path = None):
        """Class constructor

        Initializes MFCC parameters, reads the phoneme dictionary and get the list of files.

        Args:
            path: The path in which the audio authors' folders are located. The folder format should be as follows:
                |__84 [Author]
                |  |__121123 [Chapter]
                      |__{name}.txt [Transcriptions file]
                      |__{name}.{flac, wav} [Audio files] [{name} must be in transcriptions file]
                      |__[...]                      
                      |__{name}.{flac, wav} [Audio files] [{name} must be in transcriptions file]
                |  |__121550 [Chapter]
                |__174 [Author]
                |  |__50561 [Chapter]
                |  |__84280 [Chapter]
                |  |__168635 [Chapter]

            pre_emphasis: Defaults to 0.97. Defines the value of the amplification filter applied to the high frequencies of the audio.
            frame_size: Defaults to 0.025. Defines the window size in seconds.
            frame_stride: Defaults to 0.01. Defines the step between adjacent windows.
            NFFT: Not sure, leave it as it is.
            nfilt: Defaults to 40. Number of triangular filters to be applied on FFT, obtaining the Mel Scale.
            num_ceps: Defaults to 12. Number of cepstral coefficients to maintain after compression by DCT.
            cep_lifter: Not sure either.
            dict_path: Optional. Indicates the path in which the phonetic dictionary is located.
            phonemes_path: Optional, recommended to be set alongside dict_path. Indicates the path in which the phonemes are indicated
        
        Throws:
            IndexError: The folder specified in path was not in the expected format
        """
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
        self.__dict_path = dict_path
        self.__phonemes_path = phonemes_path

        if dict_path == None:
            if phonemes_path == None:
                self.__phon_dict = beep.get_phoneme_dict()
            else:
                self.__phon_dict = beep.get_phoneme_dict(phonemes_index=phonemes_path)
        elif phonemes_path is not None:
            self.__phon_dict = beep.get_phoneme_dict(path = dict_path, phonemes_index = phonemes_path)
        else:
            self.__phon_dict = beep.get_phoneme_dict(path = dict_path)
            
        self.__get_files()
        self.__batch_count = len(self.__files)
        
    def __get_files(self):
        """Prepares the folders' dict, so that we can return the audio in batches."""
        self.__files = dict()

        # listdir returns a list of all the items in a folder.
        # So, we need to use os.path.isdir to check whether a certain item is a folder or not.
        author_folders = [d for d in listdir(self.__path) if os.path.isdir(os.path.join(self.__path, d))]
        for author in author_folders:  # First, we go through the authors folders
            self.__files[author] = list()  # Inside the authors' dict, we have the chapters' dict
            author_path = os.path.join(self.__path, author)
            audio_folders = [d for d in listdir(author_path) if os.path.isdir(os.path.join(author_path, d))]
            for audio in audio_folders:  # Then, inside the authors folder we have the chapters folder
                self.__files[author].append(audio)

        self.__index = -1  # Start the authors' index before the first
        self.__author_indexes = [key for key in self.__files.keys()]  # Allows us to index the batches by number of the author

    def __get_mfcc (self, audios):
        """Do the Mel Scale transform on the audios.
        
        Args:
            audios: a dict provided by get_data() that maps the keys to tuple (numpy_arrays audio, samplerate).

        Returns:
            A tuple containing a list of the keys and a list of the audios after the MFCC transform.
        """
        keys = list()
        mfccs = list()

        for key, value in audios.items():
            audio = value[0]
            samplerate = value[1]
            audio_mfcc = python_speech_features.mfcc(audio, samplerate)
            keys.append(key)
            mfccs.append(audio_mfcc)

        return (keys, mfccs)

    def __scale_data (self, keys, mfccs):
        """Scale the MFCC coefficients into a range of (-1, 1), making it easier for the neural networks to interpret the data
        
        Args:
            keys: a list that provides the keys to be used in the dict.
            mfccs: a list that provides the mfcc transformed audios, aligned to the keys list.

        Returns:
            A dict pairing the keys to the scaled audios.
        """
        audios = list()
        for mfcc in mfccs:
            scaled_audio = minmax_scale(mfcc, feature_range=(-1,1), axis = 1)  # the axis parameter indicates that the audio will be scaled in each window, instead of each feature
            audios.append(scaled_audio)

        scaled_mfcc = dict()
        for i in range(len(keys)):
            scaled_mfcc[keys[i]] = audios[i]
        
        return scaled_mfcc

    def __convert_audios(self, path):
        """Convert all the audios in path from self.__origin_format to self.__target_format.

        Args:
            path: The path in which are the files to be converted
        """
        if self.__origin_format is None:
            self.__origin_format = AUDIO_ORIGIN_FORMAT
        if self.__target_format is None:
            self.__target_format = AUDIO_TARGET_FORMAT
        audio_parts = [d for d in listdir(path) if d.endswith('.' + self.__origin_format)]  # Get all the files that are in origin_format
        for audio_part in audio_parts:           
            audio = AudioSegment.from_file(os.path.join(path, audio_part), self.__origin_format)  # Create a object that represents the audio file
            audio.export(os.path.join(path, audio_part).split('.')[0] + '.' + self.__target_format, format = self.__target_format)  # Use the object to convert the audio

    def __get_phoneme_transcription(self, transcriptions):
        """Convert the transcripts into its respective phonemes.
        
        Args:
            transcriptions: a dict containing the audio transcripts

        Returns:
            A dict containing the key of the audio and its respective phoneme representation. Unknown words are replaced by the silence symbol
        """
        phoneme_transcripts = dict()
        phoneme_indexes = dict()
        if self.__phonemes_path != None:
            phoneme_indexes = beep.get_phoneme_indexes(self.__phonemes_path)
        else:
            phoneme_indexes = beep.get_phoneme_indexes()
        
        phoneme_indexes = beep.get_phoneme_indexes()
        for key, transcript in transcriptions.items():
            words = transcript.split()
            phrase = []
            for word in words:
                if word in self.__phon_dict.keys():
                    phrase += self.__phon_dict[word]
                else:
                    phrase += phoneme_indexes
            phoneme_transcripts[key] = phrase[:-1]
        return phoneme_transcripts


    def get_data(self, format = "wav"):
        """Gets the audios, do the transforms, get the phonetic transcriptions and pair it all up
        
        Args:
            format: the format of the audio to be read.

        Returns:
            A dict pairing the keys to tuples that contain (phonetic transcriptions, scaled mfcc converted audios)
        """
        self.__index += 1

        author = self.__author_indexes[self.__index]
        author_path = os.path.join(self.__path, author)
        transcripts = dict()
        audios = dict()
        for audio in self.__files[author]:
            curr_path = os.path.join(author_path, audio)            

            # Gets the transcripts
            transcript_file = [file for file in listdir(curr_path) if file.endswith('.txt')][0]
            transcript_lines = [line.rstrip('\n') for line in open(os.path.join(curr_path, transcript_file))]
            for line in transcript_lines:
                split = line.find(' ')
                transcripts[line[:split]] = line[split:]

            # Gets the audios and convert them
            audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + format)]
            if not audio_parts:
                if self.__target_format is None:
                    self.__target_format = format
                self.__convert_audios(curr_path)            
                audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + format)]
            for audio_part in audio_parts:                
                data, samplerate = sf.read(os.path.join(curr_path, audio_part))
                audios[audio_part.split('.')[0]] = (data, samplerate)
        
        keys, mfcc = self.__get_mfcc(audios)
        scaled_mfcc = self.__scale_data(keys, mfcc)
        result = dict()
        phon_transcripts = self.__get_phoneme_transcription(transcripts)
        for key, transcript in phon_transcripts.items():
            result[key] = (transcript, scaled_mfcc[key])
            
        if self.__index >= self.__batch_count - 1:
            self.__index = -1

        return result    

    def convert_audios(self, origin_format = None, target_format = None):
        """Sets the formats to convert the audios from and to.
        
        Args:
            origin_format: The format in which the audio is.
            target_format: The format to which the audio will be converted.
        """
        self.__origin_format = origin_format if origin_format is not None else AUDIO_ORIGIN_FORMAT
        self.__target_format = target_format if target_format is not None else AUDIO_TARGET_FORMAT

    @property
    def batch_count(self):
        return self.__batch_count

def translate_indexes(input, phonemes_path = None):
    if phonemes_path != None:
        return beep.decode_phonemes(input, phonemes_path)
    else:
        return beep.decode_phonemes(input)