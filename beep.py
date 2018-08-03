def get_phoneme_dict(path = "beep/beep-1.0"):
    beep = dict()
    with open(path,  "r", encoding = "utf-8") as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            phonemes = splitLine[1:]
            beep[word] = phonemes