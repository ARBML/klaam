def findStressIndex(sequence):  # Find stress syllable in word starting from "start"
    if sequence == "" or len(sequence) == 0:
        return ""
    # print(sequence)
    consonants = [
        "r",
        "g",
        "y",
        "G",
        "b",
        "z",
        "f",
        "v",
        "t",
        "s",
        "q",
        "p",
        "$",
        "k",
        "<",
        "j",
        "S",
        "l",
        "H",
        "D",
        "m",
        "x",
        "T",
        "n",
        "d",
        "Z",
        "h",
        "*",
        "E",
        "w",
        "^",
    ]
    geminatedConsonants = [
        "<<",
        "rr",
        "gg",
        "vv",
        "bb",
        "zz",
        "ff",
        "GG",
        "tt",
        "ss",
        "qq",
        "pp",
        "$$",
        "kk",
        "yy",
        "jj",
        "SS",
        "ll",
        "HH",
        "DD",
        "mm",
        "xx",
        "TT",
        "nn",
        "dd",
        "ZZ",
        "hh",
        "**",
        "EE",
        "ww",
        "^^",
    ]
    longVowels = ["aa", "AA", "uu0", "uu1", "ii0", "ii1", "UU0", "UU1", "II0", "II1"]
    shortVowels = ["a", "A", "u0", "u1", "i0", "i1", "U0", "U1", "I0", "I1"]
    syllableString = ""
    i = 0
    while i < len(sequence):
        if sequence[i] in geminatedConsonants:
            syllableString += "C"
        elif sequence[i] in consonants:
            syllableString += "c"
        elif sequence[i] in longVowels:
            syllableString += "V"
        elif sequence[i] in shortVowels:
            syllableString += "v"
        else:
            print("Unacceptable char when finding stress syllable: " + sequence[i] + " " + syllableString + "\n")
            file = open("errors", "a")
            file.write(sequence[i])
            file.close()
            return 0
        i += 1
    if syllableString[0] in ["v", "V"]:
        return -1
    # Stress falls on the last syllable if it is super heavy
    if syllableString.endswith("cVc") and syllableString.endswith("CVc"):
        return i - 2  # 3
    if (
        syllableString.endswith("cvvc")
        or syllableString.endswith("cvcc")
        or syllableString.endswith("cVcc")
        or syllableString.endswith("Cvvc")
        or syllableString.endswith("Cvcc")
        or syllableString.endswith("CVcc")
    ):
        return i - 3  # 4
    if syllableString.endswith("cvvcc") and syllableString.endswith("Cvvcc"):
        return i - 4  # 5
    # Stress is at the beginning if it is a monosyllabic word
    if syllableString == "cvv" or syllableString == "cvc":
        return i - 2  # 3
    if syllableString == "cV":
        return i - 1  # 2
    # Remove last syllable if first two rules miss
    if syllableString.endswith("cvv") or syllableString.endswith("cvc"):
        syllableString = syllableString[0:-3]
        i = i - 3
    elif syllableString.endswith("Cvv") or syllableString.endswith("Cvc"):
        syllableString = syllableString[0:-3]
        syllableString += "c"
        i = i - 2
    elif syllableString.endswith("cV") or syllableString.endswith("cv"):
        syllableString = syllableString[0:-2]
        i = i - 2
    elif syllableString.endswith("CV") or syllableString.endswith("Cv"):
        syllableString = syllableString[0:-2]
        syllableString += "c"
        i = i - 1
    # Stress is at penultimate syllable if disyllabic word
    if syllableString == "cvv" or syllableString == "cvc":
        return i - 2  # 3
    if syllableString == "cV" or syllableString == "cv":
        return i - 1  # 2
    # Stress is at penultimate syllable if it is heavy
    if (
        syllableString.endswith("cvv")
        or syllableString.endswith("cvc")
        or syllableString.endswith("Cvv")
        or syllableString.endswith("Cvc")
        or syllableString.endswith("cVc")
        or syllableString.endswith("cVC")
        or syllableString.endswith("CVc")
    ):
        return i - 2  # 3
    if syllableString.endswith("cV") or syllableString.endswith("CV"):
        return i - 1  # 2
    if syllableString.endswith("cv"):
        i = i - 2
        syllableString = syllableString[0:-2]
    elif syllableString.endswith("Cv"):
        i = i - 1
        syllableString = syllableString[0:-2]
        syllableString += "c"
    # Stress is at antepenultimate syllable otherwise
    if (
        syllableString.endswith("cvv")
        or syllableString.endswith("cvc")
        or syllableString.endswith("Cvv")
        or syllableString.endswith("Cvc")
        or syllableString.endswith("cVc")
        or syllableString.endswith("cVC")
        or syllableString.endswith("CVc")
    ):
        return i - 2  # 3
    if (
        syllableString.endswith("cV")
        or syllableString.endswith("cv")
        or syllableString.endswith("CV")
        or syllableString.endswith("Cv")
    ):
        return i - 1  # 2
    return i + 1
