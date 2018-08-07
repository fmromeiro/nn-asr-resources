from os import listdir
from os.path import isdir, join
import soundfile as sf

def get_transcriptions (path):
    """ This method gets the transcriptions for all the audios within a data folder
        Parameters:
            path: The path to the folder where the authors folders are
    """
    author_folders = [d for d in listdir(path) if isdir(join(path, d))]
    transcripts = dict()
    for author in author_folders:
        curr_path = join(path, author)
        audio_folders = [d for d in listdir(curr_path) if isdir(join(curr_path, d))]
        for audio in audio_folders:
            curr_path = join(curr_path, audio)
            transcript_file = [file for file in listdir(curr_path) if file.endswith('.txt')][0]
            transcript_lines = [line.rstrip('\n') for line in open(join(curr_path, transcript_file))]
            for line in transcript_lines:
                split = line.find(' ')
                transcripts[line[:split]] = line[split:]
    return transcripts

def get_audios (path):
    """ This method gets all the audios within a data folder
        Parameters:
            path: The path to the folder where the authors folders are
    """
    author_folders = [d for d in listdir(path) if isdir(join(path, d))]
    audios = dict()
    for author in author_folders:
        curr_path = join(path, author)
        audio_folders = [d for d in listdir(curr_path) if isdir(join(curr_path, d))]
        for audio_f in audio_folders:
            curr_path = join(curr_path, audio_f)
            audio_parts = [d for d in listdir(curr_path) if d.endswith('.flac')]
            for audio_part in audio_parts:                
                data, samplerate = sf.read(join(curr_path, audio_part))
                audios[audio_part.split('.')[0]] = (data, samplerate)
    return audios




