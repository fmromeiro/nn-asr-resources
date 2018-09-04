import os
__package_directory = os.path.dirname(os.path.abspath(__file__))
__default_path = os.path.join(__package_directory, "beep/beep-1.0")
__phonemes_path = os.path.join(__package_directory, "beep/phone45.tab")

def get_phoneme_dict(path = __default_path, phonemes_index = __phonemes_path):
    """ Makes a word to phoneme dict

        Returns:
            A dict mapping words to its corresponding phonetic translation in
            index format. For example:

            {'SOMEWORD': [1, 2, 3, 4, 5],
             [...]}
        Args:
            path: Optional. Indicates which phonetic/pronunciation dictionary to use
            phonemes_index: Optional, but recommended if path is non-default. Indicates which are the phonemes in translation
    """
    beep = dict()
    phoneme_indexes = get_phoneme_indexes(phonemes_index)
    with open(path,  "r", encoding = "utf-8") as f:
        for line in f:
            splitLine = line.split( )  # Each line in the file has a single word and it's respective phoneme translation
            word = splitLine[0]  # The first word is always the one being translated  
            if word == '#':
                continue
            word = word.replace("_", " ")  # We use replace because the dictionary file substitutes spaces for underlines
            word = word.strip()  # Some word may have have spaces at their end, so we need to remove those
            phonemes = splitLine[1:]  # After the word we have the phoneme translation, separated by spaces
            phonemes = [phoneme_indexes[phoneme] for phoneme in phonemes] # translate the phonemes into their indexes
            phonemes_list = list()
            for phon in phonemes:  # transform the list of phonemes into a string
                phonemes_list.append(phon)
            beep[word] = phonemes_list
    return beep

def get_phoneme_indexes(phonemes_index = __phonemes_path):
    """ Returns a dict pairing (phoneme, index)

        Args:
            phonemes_index: the path in which the phonemes are indicated    
    """
    phoneme_dict = dict()
    i = -1 # sil is not considered a phoneme
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