from os import listdir
from pathlib import Path
import os
import random
from collections import Generator
from pydub import AudioSegment
import soundfile as sf
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import python_speech_features
from sklearn.preprocessing import minmax_scale
import beep
import numpy as np

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

    def __init__(self, path, convert_files=True, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, cep_lifter = None, dict_path = None, phonemes_path = None):
        """Class constructor

        Initializes MFCC parameters, reads the phoneme dictionary and get the list of files.

        Args:
            path: The path in which the audio authors' folders are located. The folder format should be as follows:
                |_84 [Author]
                |  |_121123 [Chapter]
                      |_{name}.txt [Transcriptions file]
                      |_{name}.{flac, wav} [Audio files] [{name} must be in transcriptions file]
                      |_[...]                      
                      |_{name}.{flac, wav} [Audio files] [{name} must be in transcriptions file]
                |  |_121550 [Chapter]
                |_174 [Author]
                |  |_50561 [Chapter]
                |  |_84280 [Chapter]
                |  |_168635 [Chapter]

            convert_files: Defaults to True. Whether it should convert all files to the target format before starting.
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
        self._path = path
        self._pre_emphasis = pre_emphasis if pre_emphasis is not None else PRE_EMPHASIS
        self._frame_size = frame_size if frame_size is not None else FRAME_SIZE
        self._frame_stride = frame_stride if frame_stride is not None else FRAME_STRIDE
        self._NFFT = NFFT if NFFT is not None else nfft
        self._nfilt = nfilt if nfilt is not None else NFILT 
        self._num_ceps = num_ceps if num_ceps is not None else NUM_CEPS
        self._cep_lifter = cep_lifter if cep_lifter is not None else CEP_LIFTER
        self._origin_format = AUDIO_ORIGIN_FORMAT # TODO: swa this with constructor value
        self._target_format = AUDIO_TARGET_FORMAT
        self._dict_path = dict_path
        self._phonemes_path = phonemes_path

        if dict_path == None:
            if phonemes_path == None:
                self._phon_dict = beep.get_phoneme_dict()
            else:
                self._phon_dict = beep.get_phoneme_dict(phonemes_index=phonemes_path)
        elif phonemes_path is not None:
            self._phon_dict = beep.get_phoneme_dict(path = dict_path, phonemes_index = phonemes_path)
        else:
            self._phon_dict = beep.get_phoneme_dict(path = dict_path)
            
        self._get_files_dict()
        if convert_files:
            self._convert_files()
        
    def _get_files_dict(self):
        """Prepares the _files dict which maps audio path to phoneme transcripts."""
        self._files = dict()

        # listdir returns a list of all the items in a folder.
        # So, we need to use os.path.isdir to check whether a certain item is a folder or not.
        author_folders = [d for d in listdir(self._path) if os.path.isdir(os.path.join(self._path, d))]
        for author in author_folders:  # First, we go through the authors folders
            author_path = os.path.join(self._path, author)
            chapter_folders = [d for d in listdir(author_path) if os.path.isdir(os.path.join(author_path, d))]
            for chapter in chapter_folders:  # Then, inside the authors folder we have the chapters folder
                files_path = os.path.join(author_path, chapter)
                all_files = [file for file in listdir(files_path)]
                transcript_file = [file for in all_files if file.endswith(".txt")][0]
    
                # gets the transcripts
                transcript_lines = [line.rstrip('\n') for line in open(transcript_file)]
                for line in transcript_lines:
                    split = line.find(' ')
                    alias = line[:split]
                    transcript = line[split:]
                    phoneme_transcript = self._get_phoneme_transcript(transcript)
                    if phoneme_transcript != None:
                        path = os.path.join(files_path, alias + self._target_format)
                        self._files[alias] = phoneme_transcript

        self._audio_count = len(self._files)

    def _get_mfcc (self, audio, samplerate):
        """Do the Mel Scale transform on a single audio.
        
        Args:
            audio: a numpy array containing the audio data.
            samplerate: the samplerate associated with the audio

        Returns:
            The MFCC scaled audio.
        """
        return python_speech_features.mfcc(audio, samplerate)

    def _scale_data (self, mfcc_audio):
        """Scale the MFCC coefficients of an audio into a range of (-1, 1), making it easier for the neural networks to interpret the data
        
        Args:
            mfcc_audio: a single mfcc_audio in the format of a numpy array

        Returns:
            The scaled audio.
        """
        # the axis parameter indicates that the audio will be scaled in each window, instead of each feature
        return minmax_scale(mfcc, feature_range=(-1,1), axis = 1)

    def _convert_files(self):
        """Converts all files listed as the _files dict's keys"""
        for file in self._files.keys():
            self._convert_file(path)

    def _convert_file(self, path):
        """Convert the the file to target format in path if not exists

        Args:
            path: The audio file path to be converted
        """
        file = Path(path)
        if !file.exists(): # only converts if it needs to be converted
            # Create a object that represents the audio file
            source_path = os.path.join(path, audio_part).split('.')[0] + '.' + self._source_format
            audio = AudioSegment.from_file(source_path)
            # Use the object to convert the audio 
            audio.export(path, format = self._target_format)

    def _get_phoneme_transcript(self, transcript):
        """Convert a single transcript into its respective phonemes.
        
        Args:
            transcript: a string containing an audio transcript

        Returns:
            A string with the transcript's respective phoneme representation. Unknown words are replaced by the silence symbol
            Returns None if the transcript is not valid.
        """
        phoneme_transcript = ''
        for word in transcript.split():
            if word in self._phon_dict.keys():
                phoneme_transcript += self._phon_dict[word]
            else:
                return None

        return phoneme_transcript

    def _get_input_tensor(self, batch, mask):
        """Returns a tuple of an 1D tensor containing the lengths of each sample in the batch and
        and the input tensor of shape (batch_size, timesteps, feature_dims)
        
        Args:
            batch: the batch object returned by AudioPrep._get_data()
        """
        audios = [value[1] for value in batch.values()]
        max_size = max(audios, key=lambda a: a.shape[0]).shape[0]
        features = audios[0].shape[1]
        padded_audios = []
        input_lengths = []
        for audio in audios:
            input_lengths.append(audio.shape[0])
            result = np.full((max_size, features), mask)
            result[:audio.shape[0], :features] = audio
            padded_audios.append(result)
        
        padded_audios = np.array(padded_audios)
        input_lengths = np.array(input_lengths)
        return input_lengths, padded_audios

    def _get_output_tensor(self, batch, mask):
        """Returns a tuple of an 1D tensor containing the lengths of each sample in the batch and
        and the input tensor of shape (batch_size, timesteps, feature_dims)
        
        Args:
            batch: the batch object returned by AudioPrep._get_data()
        """
        labels = [value[0] for value in batch.values()]
        max_size = len(max(labels, key=lambda label: len(label)))

        label_lengths = []
        padded_labels = []
        for label in labels:
            label_lengths.append(len(label))
            result = np.full((max_size), mask)
            result[:len(label)] = label
            padded_labels.append(result)

        padded_labels = np.array(padded_labels)
        label_lengths = np.array(label_lengths)

        return label_lengths, padded_labels


    def _get_batch_data(self):
        """Get the audios for a single batch, do the transforms, get the phonetic transcriptions and pair it all up
        
        Args:
            format: the format of the audio to be read.

        Returns:
            A dict pairing the keys to tuples that contain (phonetic transcriptions, scaled mfcc converted audios)
        """
        

        return result

    def get_batch(self, file_set=None, input_mask=2, output_mask=0):
        """Gets a single batch from a set of file paths
        
        Args:
            file_set: a set containing paths to the audios in batch
            input_mask: the input mask for this batch
            output_mask: the output mask for this batch

        Returns:
            A tuple equivalent to the input of a Keras CTC network read from the files in file_set
        """
        batch = self._get_batch_data(file_set)
        input_lengths, X = self._get_input_tensor(batch, input_mask)
        label_lengths, y = self._get_output_tensor(batch, output_mask)

        return X, y, input_lengths, label_lengths

    class BatchGenerator(Generator):
        """Batch generator class for using it in fit_generator"""
        def __init__(self, audio_prep=None, randomize=False, batch_size=1, input_mask=2, output_mask=0):
            if audio_prep == None or not isinstance(audio_prep, AudioPrep):
                raise ValueError("No audioprep provided to generator")
            self._audio_prep = audio_prep
            self._file_queue = audio_prep._files.keys()
            if randomize:
                self._file_queue = random.shuffle(self._file_queue)
            self._batch_size = batch_size
            self._batch_count = self._audio_prep.batch_count(self._batch_size)
            self._input_mask = input_mask
            self._output_mask = output_mask
            self._index = 0

        def __len__(self):
            return self._batch_count

        def send(self, _):
            if self._index >= self._batch_count:
                raise StopIteration

            file_set = set()
            for _ in range(self._batch_size):
                file_set.add(self._file_queue.pop())

            self._index += 1
                
            return self._audio_prep.get_batch(file_set, self._input_mask, self._output_mask)

        def throw(self, type=None, value=None, traceback=None)
            raise StopIteration


    def batch_generator(self, batch_size=1, input_mask=2, output_mask=0):
        while True:
            X, y, input_lengths, label_lengths = self.get_batch(input_mask, output_mask)
            for i in range(0, X.shape[0]-batch_size, batch_size):
                tmp_x = np.atleast_3d(X[i:i+batch_size])
                tmp_y = np.atleast_2d(y[i:i+batch_size])
                tmp_input_lengths = input_lengths[i:i+batch_size]
                tmp_label_lengths = label_lengths[i:i+batch_size]
                yield ([tmp_x, 
                        tmp_y, 
                        tmp_input_lengths, 
                        tmp_label_lengths], 
                        tmp_y)

    def convert_audios(self, origin_format = None, target_format = None):
        """Sets the formats to convert the audios from and to.
        
        Args:
            origin_format: The format in which the audio is.
            target_format: The format to which the audio will be converted.
        """
        self._origin_format = origin_format if origin_format is not None else AUDIO_ORIGIN_FORMAT
        self._target_format = target_format if target_format is not None else AUDIO_TARGET_FORMAT

    def batch_count(self, batch_size):
        return self._audio_count // batch_size

    

def translate_indexes(input, phonemes_path = None):
    if phonemes_path != None:
        return beep.decode_phonemes(input, phonemes_path)
    else:
        return beep.decode_phonemes(input)