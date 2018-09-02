import os
__package_directory = os.path.dirname(os.path.abspath(__file__))
__default_path = os.path.join(__package_directory, "beep/beep-1.0")

def get_phoneme_dict(path = __default_path):
    """ Makes a word to phoneme dict

        Returns:
            A dict mapping words to its corresponding phonetic translation in
            string format. The phonemes are separated by spaces. For example:

            {'AACHEN': 'aa k ax n',
             'AACHEN'S': 'aa k ax n z',
             'AARDVARK': 'aa d v aa k',
             'AARDVARKS': 'aa d v aa k s',
             'AARHUS': 'aa hh uw s'
             [...]}
    """
    beep = dict()
    with open(path,  "r", encoding = "utf-8") as f:
        for line in f:
            splitLine = line.split( )  # Each line in the file has a single word and it's respective phoneme translation
            word = splitLine[0]  # The first word is always the one being translated                                               
            word = word.replace("_", " ")  # We use replace because the dictionary file substitutes spaces for underlines
            word = word.strip()  # Some word may have have spaces at their end, so we need to remove those
            phonemes = splitLine[1:]  # After the word we have the phoneme translation, separated by spaces
            phonemes_str = ""
            for phon in phonemes:  # transform the list of phonemes into a string
                phonemes_str += phon + " "
            beep[word] = phonemes_str[:-1]
    return beep