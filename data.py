# I'm just happy that everything here works as of 08/09/2018 11:14:28

from os import listdir
from os.path import isdir, join
from pydub import AudioSegment
import soundfile as sf

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

def get_mfcc