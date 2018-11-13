from os import listdir
from pathlib import Path
import os
import glob
import random
import time
import yaml
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
AUDIO_SOURCE_FORMAT = "flac"
AUDIO_TARGET_FORMAT = "wav"

class AudioPrep(object):
    """Get audios, convert them into numpy arrays, do the MFCC transformation, translate the transcriptions into phonemes and make xy pairments."""

    @staticmethod
    def _get_mfcc (audio, samplerate):
        """Do the Mel Scale transform on a single audio.
        
        Args:
            audio: a numpy array containing the audio data.
            samplerate: the samplerate associated with the audio

        Returns:
            The MFCC scaled audio.
        """
        return python_speech_features.mfcc(audio, samplerate)

    @staticmethod
    def _scale_data (mfcc_audio):
        """Scale the MFCC coefficients of an audio into a range of (-1, 1), making it easier for the neural networks to interpret the data
        
        Args:
            mfcc_audio: a single mfcc_audio in the format of a numpy array

        Returns:
            The scaled audio.
        """
        # the axis parameter indicates that the audio will be scaled in each window, instead of each feature
        return minmax_scale(mfcc_audio, feature_range=(-1,1), axis = 1)

    @staticmethod
    def _convert_file(path, source_format = AUDIO_SOURCE_FORMAT, target_format=AUDIO_TARGET_FORMAT):
        """Convert the the file to target format in path if not exists

        Args:
            path: The audio file path to be converted
        """
        file = Path(path)
        if not file.exists(): # only converts if it needs to be converted
            # Create a object that represents the audio file
            source_path = path.split('.')[0] + '.' + source_format
            audio = AudioSegment.from_file(source_path)
            # Use the object to convert the audio 
            audio.export(path, format = target_format)

    @staticmethod
    def _get_phoneme_transcript(transcript, phon_dict):
        """Convert a single transcript into its respective phonemes.
        
        Args:
            transcript: a list containing an audio transcript

        Returns:
            A list with the transcript's respective phoneme representation. Unknown words are replaced by the silence symbol
            Returns None if the transcript is not valid.
        """
        phoneme_transcript = list()
        for word in transcript.split():
            if word in phon_dict.keys():
                phoneme_transcript += phon_dict[word]
            else:
                return None

        return phoneme_transcript

    @staticmethod
    def generate_cache(raw_data_path, cache_path, phonemes_path = None):
        """ Generates precached binary data for faster object usage
        Args:
            raw_data_path: The path in which the audio authors' folders are located. The folder format should be as follows:
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
            cache_path: The path where the cache should be located
            phonemes_path: Optional. Indicates which phonetical dictionary should be loaded.
        """
        start_time = time.time()

        directory = Path(cache_path)
        directory.mkdir(exist_ok=True) # creates directory if not exists
        word_transcript_path = os.path.join(cache_path, "word_transcripts.txt")
        phoneme_transcript_path = os.path.join(cache_path, "phoneme_transcripts.txt")
        
        metadata_path = os.path.join(cache_path, ".meta.yaml")
        if os.path.isfile(metadata_path):
            with open(metadata_path, 'r') as file: # if cache has already been generated, returns
                metadata = yaml.load(file)
                if metadata["done"] and metadata["source-path"] == raw_data_path:
                    return

        metadata = {
            "source-path": raw_data_path,
            "author-count": 0,
            "chapter-count": 0,
            "file-count": 0,
            "done": False
        }

        phon_dict = beep.get_phoneme_dict() # TODO: Phonemes path

        # listdir returns a list of all the items in a folder.
        # So, we need to use os.path.isdir to check whether a certain item is a folder or not.
        word_transcript_file = open(word_transcript_path, 'w')
        phoneme_transcript_file = open(phoneme_transcript_path, 'w')
        author_folders = [d for d in listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, d))]
        metadata["author-count"] += len(author_folders)
        for author in author_folders:  # First, we go through the authors folders
            author_path = os.path.join(raw_data_path, author)
            chapter_folders = [d for d in listdir(author_path) if os.path.isdir(os.path.join(author_path, d))]
            metadata["chapter-count"] += len(chapter_folders)
            for chapter in chapter_folders:  # Then, inside the authors folder we have the chapters folder
                files_path = os.path.join(author_path, chapter)
                all_files = [file for file in listdir(files_path)]
                transcript_file = [file for file in all_files if file.endswith(".txt")][0]
                transcript_file = os.path.join(files_path, transcript_file)
    
                # gets the transcripts
                transcript_lines = [line.rstrip('\n') for line in open(transcript_file)]
                metadata["file-count"] += len(transcript_lines)
                for line in transcript_lines:
                    split = line.find(' ')
                    alias = line[:split]
                    transcript = line[split:]
                    phoneme_transcript = AudioPrep._get_phoneme_transcript(transcript, phon_dict)
                    
                    if phoneme_transcript == None: # audio is invalid
                        continue

                    audio_path = os.path.join(files_path, alias + ".wav")
                    AudioPrep._convert_file(audio_path)
                    
                    audio, samplerate = sf.read(audio_path)
                    mfcc = AudioPrep._get_mfcc(audio, samplerate)
                    scaled = AudioPrep._scale_data(mfcc)
                    np.save(os.path.join(cache_path, alias), scaled)

                    word_transcript_file.write(line)
                    phoneme_transcript_file.write(alias + ' ')
                    for phoneme in phoneme_transcript:
                        phoneme_transcript_file.write(str(phoneme) + ' ')
                    phoneme_transcript_file.write('\n')
            word_transcript_file.flush()
            phoneme_transcript_file.flush()

        metadata["generation-time"] = int(time.time() - start_time)
        metadata["finish-time"] = int(time.time())
        metadata["done"] = True

        with open(metadata_path, 'w') as file:
            yaml.dump(metadata, file, default_flow_style=False)


        

    def __init__(self, cache_path, pre_emphasis = None, frame_size = None, frame_stride = None, NFFT = None, nfilt = None, num_ceps = None, cep_lifter = None):
        """Class constructor

        Initializes MFCC parameters, reads the phoneme dictionary and get the list of files.

        Args:
            cache_path: The path in which the data to be read is cached. If this folder has not been generated, see AudioPrep.generate_cache().
            pre_emphasis: Defaults to 0.97. Defines the value of the amplification filter applied to the high frequencies of the audio.
            frame_size: Defaults to 0.025. Defines the window size in seconds.
            frame_stride: Defaults to 0.01. Defines the step between adjacent windows.
            NFFT: Not sure, leave it as it is.
            nfilt: Defaults to 40. Number of triangular filters to be applied on FFT, obtaining the Mel Scale.
            num_ceps: Defaults to 12. Number of cepstral coefficients to maintain after compression by DCT.
            cep_lifter: Not sure either.
        
        Throws:
            IndexError: The folder specified in path was not in the expected format
        """
        self._cache_path = cache_path
        self._pre_emphasis = pre_emphasis if pre_emphasis is not None else PRE_EMPHASIS
        self._frame_size = frame_size if frame_size is not None else FRAME_SIZE
        self._frame_stride = frame_stride if frame_stride is not None else FRAME_STRIDE
        self._NFFT = NFFT if NFFT is not None else nfft
        self._nfilt = nfilt if nfilt is not None else NFILT 
        self._num_ceps = num_ceps if num_ceps is not None else NUM_CEPS
        self._cep_lifter = cep_lifter if cep_lifter is not None else CEP_LIFTER
        self._source_format = AUDIO_SOURCE_FORMAT # TODO: swa this with constructor value
        self._target_format = AUDIO_TARGET_FORMAT

        self._transcript_file = os.path.normpath(os.path.join(self._cache_path, "phoneme_transcripts.txt"))
        self._load_transcripts()

    def _load_transcripts(self):
        self._transcripts = dict()
        for line in open(self._transcript_file):
            tokens = line.split(' ')
            alias = tokens[0]
            transcripts = [int(token) for token in tokens[1:-1]] # remove alias and '\n' and cast to int

            self._transcripts[alias] = transcripts

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
            result = np.full((max_size, features), mask, dtype=np.float32)
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
            result = np.full((max_size), mask, dtype=np.int32)
            result[:len(label)] = label
            padded_labels.append(result)

        padded_labels = np.array(padded_labels)
        label_lengths = np.array(label_lengths)

        return label_lengths, padded_labels


    def _get_batch_data(self, data_set):
        """Get the audios for a single batch, do the transforms, get the phonetic transcriptions and pair it all up
        
        Args:
            data_set: a set containing data IDs to be contained in batch

        Returns:
            A dict pairing the keys to tuples that contain (phonetic transcriptions, scaled mfcc converted audios)
        """
        batch_data = dict()
        for data_id in data_set:
            audio_data = np.load(os.path.join(self._cache_path, data_id + ".npy"))

            transcription = self._transcripts[data_id]
            batch_data[data_id] = (transcription, audio_data)

        return batch_data

    def get_batch(self, data_set=None, input_mask=2, output_mask=0):
        """Gets a single batch from a set of file paths
        
        Args:
            data_set: a set containing data IDs to be contained in batch
            input_mask: the input mask for this batch
            output_mask: the output mask for this batch

        Returns:
            A tuple equivalent to the input of a Keras CTC network read from the files in file_set
        """
        batch = self._get_batch_data(data_set)
        input_lengths, X = self._get_input_tensor(batch, input_mask)
        label_lengths, y = self._get_output_tensor(batch, output_mask)

        return ([X, y, input_lengths, label_lengths], y)

    class BatchGenerator(Generator):
        """Batch generator class for using it in fit_generator"""
        def __init__(self, audio_prep=None, batch_size=1, randomize=False, input_mask=2, output_mask=0):
            if audio_prep == None or not isinstance(audio_prep, AudioPrep):
                raise ValueError("No audioprep provided to generator")
            self._audio_prep = audio_prep
            self._data_queue = list(audio_prep._transcripts.keys())
            if randomize:
                random.shuffle(self._data_queue)
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

            data_set = set()
            for _ in range(self._batch_size):
                data_set.add(self._data_queue.pop())

            self._index += 1
                
            return self._audio_prep.get_batch(data_set, self._input_mask, self._output_mask)

        def throw(self, type=None, value=None, traceback=None):
            raise StopIteration


    def batch_generator(self, batch_size=1, randomize=False, input_mask=2, output_mask=0):
        return AudioPrep.BatchGenerator(self, batch_size, randomize, input_mask, output_mask)
        # while True:
        #     X, y, input_lengths, label_lengths = self.get_batch(input_mask, output_mask)
        #     for i in range(0, X.shape[0]-batch_size, batch_size):
        #         tmp_x = np.atleast_3d(X[i:i+batch_size])
        #         tmp_y = np.atleast_2d(y[i:i+batch_size])
        #         tmp_input_lengths = input_lengths[i:i+batch_size]
        #         tmp_label_lengths = label_lengths[i:i+batch_size]
        #         yield ([tmp_x, 
        #                 tmp_y, 
        #                 tmp_input_lengths, 
        #                 tmp_label_lengths], 
        #                 tmp_y)

    def convert_audios(self, source_format = None, target_format = None):
        """Sets the formats to convert the audios from and to.
        
        Args:
            source_format: The format in which the audio is.
            target_format: The format to which the audio will be converted.
        """
        self._source_format = source_format if source_format is not None else AUDIO_SOURCE_FORMAT
        self._target_format = target_format if target_format is not None else AUDIO_TARGET_FORMAT

    def batch_count(self, batch_size):
        return len(self._transcripts) // batch_size

    

def translate_indexes(input, phonemes_path = None):
    if phonemes_path != None:
        return beep.decode_phonemes(input, phonemes_path)
    else:
        return beep.decode_phonemes(input)