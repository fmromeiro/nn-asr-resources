def get_phoneme_dict(path = "beep/beep-1.0"):
    beep = dict()
    with open(path,  "r", encoding = "utf-8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            phonemes = splitLine[1:]
            phonemes_str = ""
            for phon in phonemes:
                # print(phon)
                phonemes_str += phon + " "
                #print(phonemes_str)
            beep[word] = phonemes_str[:-1]
    return beep