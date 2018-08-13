# I'm just happy that everything here works as of 08/09/2018 11:14:28

#TODO: Create a class that feeds audio in batches, dividing by authors or by transcripts

from os import listdir
from os.path import isdir, join
from pydub import AudioSegment
import soundfile as sf
import numpy
import scipy.io.wavfile
from scipy.fftpack import dct

# constants
pre_emphasis = 0.97 # Value of the amplification filter applied to high frequencies of the audio
frame_size = 0.025 # Size of the audio window in ms
frame_stride = 0.01 # Step between audio windows in ms (notice that it's shorter than the frame_size, resulting in overlap)
NFFT = 512 # Not sure what this is, but it stands for the N in N-point FFT
nfilt = 40 # Number of triangular filters to be applied on FFT, creating a Mel-scale
num_ceps = 12 # Number of cepstral coefficients to retain after compression using Discrete Consine Transform (DCT)

def get_transcriptions (path):
    """ This method gets the transcriptions for all the audios within a data folder
        Parameters:
            path: The path to the folder where the authors folders are
    """
    author_folders = [d for d in listdir(path) if isdir(join(path, d))]
    transcripts = dict()
    for author in author_folders:
        author_path = join(path, author)
        print ("1: ", author_path)
        audio_folders = [d for d in listdir(author_path) if isdir(join(author_path, d))]
        for audio in audio_folders:
            curr_path = join(author_path, audio)
            print ("2:", curr_path)
            transcript_file = [file for file in listdir(curr_path) if file.endswith('.txt')][0]
            transcript_lines = [line.rstrip('\n') for line in open(join(curr_path, transcript_file))]
            for line in transcript_lines:
                split = line.find(' ')
                transcripts[line[:split]] = line[split:]
    return transcripts

def get_audios (path, format = "wav"):
    """ This method gets all the audios within a data folder
        Parameters:
            path: The path to the folder where the authors folders are
            format: Defaults to "wav". Can't ensure it will work with other formats
    """
    author_folders = [d for d in listdir(path) if isdir(join(path, d))]
    audios = dict()
    for author in author_folders:
        author_path = join(path, author)
        print ("1: ", author_path)
        audio_folders = [d for d in listdir(author_path) if isdir(join(author_path, d))]
        for audio_f in audio_folders:
            curr_path = join(author_path, audio_f)
            print("2: ", curr_path)
            audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + format)]
            for audio_part in audio_parts:                
                print("3: ",join(curr_path, audio_part))
                data, samplerate = sf.read(join(curr_path, audio_part))                
                print("3: ",join(curr_path, audio_part))
                audios[audio_part.split('.')[0]] = (data, samplerate)
    return audios

def convert_audios(path, origin_format = "flac", target_format = "wav"):
    """
        If the method get_audios() fail, try running this and formating the audios to wav
    """
    author_folders = [d for d in listdir(path) if isdir(join(path, d))]
    audios = dict()
    for author in author_folders:
        author_path = join(path, author)
        print ("1: ", author_path)
        audio_folders = [d for d in listdir(author_path) if isdir(join(author_path, d))]
        for audio_f in audio_folders:
            curr_path = join(author_path, audio_f)
            print("2: ", curr_path)
            audio_parts = [d for d in listdir(curr_path) if d.endswith('.' + origin_format)]
            for audio_part in audio_parts:                
                print("3: ",join(curr_path, audio_part))
                audio = AudioSegment.from_file(join(curr_path, audio_part), origin_format)
                audio.export(join(curr_path, audio_part).split('.')[0] + '.' + target_format, format = target_format)

def get_mfcc (path):
    # Clearly based on the methods described at: https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    # Obtain audios
    audios = get_audios(path)
    mfcc_result = dict()
    filter_banks_result = dict()

    for key, value in audios.items():

        # Pre-Emphasis - amplify the high frequencies
        audio = value[0]
        samplerate = value[1]
        emphasized_audio = numpy.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        #audios[key] = (emphasized_audio, value[1])

        # Framing - split the audio into short-time frames
        frame_length, frame_step = frame_size * samplerate, frame_stride * samplerate # Converts from seconds to samples
        audio_length = len(emphasized_audio)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(audio_length - frame_length)) / frame_step)) # Make sure that we have at least 1 frame

        pad_audio_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_audio_length - audio_length))
        pad_audio = numpy.append(emphasized_audio, z) # Pad audio to make sure that all frames have equal number of samples without truncating any samples from the original audio

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T # It truly misses me what's happening here
        frames = pad_audio[indices.astype(numpy.int32, copy = False)]

        # Windowing - avoid spectral leakage (as far as I could gather, FFT tends to notice frequencies that aren't really in the audio if we don't do this. I'm no specialist anyway)
        frames *= numpy.hamming(frame_length)
        # frames *= 0.54 - 0.46 * numpycos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

        # Fourier-Transform and Power Spectrum
        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT)) # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2)) # Power Spectrum

        # Filter Banks - creates the Mel-scale of the audio
        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (samplerate / 2) / 200)) # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt+2) # Equally space in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
        bin = numpy.floor ((NFFT + 1) * hz_points / samplerate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range (1, nfilt + 1):
            f_m_minus = int(bin[m - 1]) # left
            f_m = int(bin[m])           # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks = 0, numpy.finfo(float).eps, filter_banks) # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks) # dB
        # By here we already have the Mel-scale filter banks, but now we keep going to the...

        # Mel-frequency Cepstral Coefficients (MFCCs)
        mfcc = dct(filter_banks, type = 2, axis = 1, norm = 'ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
        # sinusoidal liftering - improve recognition in noisy signals
        (nframes, mcoeff) = mfcc.shape
        n = numpy.arange(ncoeff)
        lift = 1 + (cep_lifter / 2)  * numpy.sin(numpy.pi * n / cep_lifter)
        mfcc *= lift

        # Mean Normalization
        filter_banks -= (numpy.mean(filter_banks, axis = 0) + 1e-8)
        mfcc -= (numpy.mean(mfcc, axis = 0) + 1e-8)
        mfcc_result[key] = mfcc
        filter_banks_result[key] = filter_banks

    return mfcc_result

def get_x_y_mfcc(path):
    trancripts = get_transcriptions(path)
    audios = get_mfcc(path)
    result = dict()
    for key, transcript in transcripts:
        result[key] = (transcript, audios[key])
    return result