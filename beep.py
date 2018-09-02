import os
__package_directory = os.path.dirname(os.path.abspath(__file__))
__default_path = os.path.join(__package_directory, "beep/beep-1.0")
__phonemes_path = os.path.join(__package_directory, "beep/phone45.tab")

def get_phoneme_dict(path = __default_path, phonemes_index = __phonemes_path):
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
        Args:
            path: Optional. Indicates which phonetic/pronunciation dictionary to use
            phonemes: Optional, but recommended if path is non-default. Indicates which are the phonemes in translation
    """
    beep = dict()
    phoneme_indexes = get_phoneme_indexes(phonemes_index)
    with open(path,  "r", encoding = "utf-8") as f:
        for line in f:
            splitLine = line.split( )  # Each line in the file has a single word and it's respective phoneme translation
            word = splitLine[0]  # The first word is always the one being translated                                               
            word = word.replace("_", " ")  # We use replace because the dictionary file substitutes spaces for underlines
            word = word.strip()  # Some word may have have spaces at their end, so we need to remove those
            phonemes = splitLine[1:]  # After the word we have the phoneme translation, separated by spaces
            phonemes = [phoneme_indexes[phonemes[i]] for i in range(len(phonemes))] # translate the phonemes into their indexes
            phonemes_str = ""
            for phon in phonemes:  # transform the list of phonemes into a string
                phonemes_str += str(phon) + " "
            beep[word] = phonemes_str[:-1]
    return beep

def get_phoneme_indexes(phonemes_index = __phonemes_path):
    """ Returns a dict pairing (phoneme, index)

        Args:
            phonemes_index: the path in which the phonemes are indicated    
    """
    phoneme_dict = dict()
    i = 0
    with open(phonemes_index, "r", encoding = "utf-8") as f:
        for line in f:
            phoneme = line.split()[0]
            phoneme_dict[phoneme] = i
            i += 1
    return phoneme_dict

def decode_phonemes(indexes, phonemes_index = __phonemes_path):
    """ Translates phonemes indexes into their phonetic representation.

        Args:
            indexes: the list of indexes to be translated
            phonemes_index: the path in which the phonemes are indicated    
    """
    phoneme_indexes = get_phoneme_indexes(phonemes_index)
    phoneme_indexes = [(value, key) for (key, value) in phoneme_indexes.items()]
    translation = [phoneme_indexes[i] for i in indexes]
    return translation