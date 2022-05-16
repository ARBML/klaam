# https://github.com/nawarhalabi/Arabic-Phonetiser
import codecs
import os
import re
import sys

from .find_stress import *

buckwalter = {  # mapping from Arabic script to Buckwalter
    "\u0628": "b",
    "\u0630": "*",
    "\u0637": "T",
    "\u0645": "m",
    "\u062a": "t",
    "\u0631": "r",
    "\u0638": "Z",
    "\u0646": "n",
    "\u062b": "^",
    "\u0632": "z",
    "\u0639": "E",
    "\u0647": "h",
    "\u062c": "j",
    "\u0633": "s",
    "\u063a": "g",
    "\u062d": "H",
    "\u0642": "q",
    "\u0641": "f",
    "\u062e": "x",
    "\u0635": "S",
    "\u0634": "$",
    "\u062f": "d",
    "\u0636": "D",
    "\u0643": "k",
    "\u0623": ">",
    "\u0621": "'",
    "\u0626": "}",
    "\u0624": "&",
    "\u0625": "<",
    "\u0622": "|",
    "\u0627": "A",
    "\u0649": "Y",
    "\u0629": "p",
    "\u064a": "y",
    "\u0644": "l",
    "\u0648": "w",
    "\u064b": "F",
    "\u064c": "N",
    "\u064d": "K",
    "\u064e": "a",
    "\u064f": "u",
    "\u0650": "i",
    "\u0651": "~",
    "\u0652": "o",
}

ArabicScript = {  # mapping from Buckwalter to Arabic script
    "b": "\u0628",
    "*": "\u0630",
    "T": "\u0637",
    "m": "\u0645",
    "t": "\u062a",
    "r": "\u0631",
    "Z": "\u0638",
    "n": "\u0646",
    "^": "\u062b",
    "z": "\u0632",
    "E": "\u0639",
    "h": "\u0647",
    "j": "\u062c",
    "s": "\u0633",
    "g": "\u063a",
    "H": "\u062d",
    "q": "\u0642",
    "f": "\u0641",
    "x": "\u062e",
    "S": "\u0635",
    "$": "\u0634",
    "d": "\u062f",
    "D": "\u0636",
    "k": "\u0643",
    ">": "\u0623",
    "'": "\u0621",
    "}": "\u0626",
    "&": "\u0624",
    "<": "\u0625",
    "|": "\u0622",
    "A": "\u0627",
    "Y": "\u0649",
    "p": "\u0629",
    "y": "\u064a",
    "l": "\u0644",
    "w": "\u0648",
    "F": "\u064b",
    "N": "\u064c",
    "K": "\u064d",
    "a": "\u064e",
    "u": "\u064f",
    "i": "\u0650",
    "~": "\u0651",
    "o": "\u0652",
}


def arabicToBuckwalter(word):  # Convert input string to Buckwalter
    res = ""
    for letter in word:
        if letter in buckwalter:
            res += buckwalter[letter]
        else:
            res += letter
    return res


def buckwalterToArabic(word):  # Convert input string to Arabic
    res = ""
    for letter in word:
        if letter in ArabicScript:
            res += ArabicScript[letter]
        else:
            res += letter
    return res


# ----------------------------------------------------------------------------
# Grapheme to Phoneme mappings------------------------------------------------
# ----------------------------------------------------------------------------
unambiguousConsonantMap = {
    "b": "b",
    "*": "*",
    "T": "T",
    "m": "m",
    "t": "t",
    "r": "r",
    "Z": "Z",
    "n": "n",
    "^": "^",
    "z": "z",
    "E": "E",
    "h": "h",
    "j": "j",
    "s": "s",
    "g": "g",
    "H": "H",
    "q": "q",
    "f": "f",
    "x": "x",
    "S": "S",
    "$": "$",
    "d": "d",
    "D": "D",
    "k": "k",
    ">": "<",
    "'": "<",
    "}": "<",
    "&": "<",
    "<": "<",
}

ambiguousConsonantMap = {
    "l": ["l", ""],
    "w": "w",
    "y": "y",
    "p": ["t", ""],  # These consonants are only unambiguous in certain contexts
}

maddaMap = {"|": [["<", "aa"], ["<", "AA"]]}

vowelMap = {
    "A": [["aa", ""], ["AA", ""]],
    "Y": [["aa", ""], ["AA", ""]],
    "w": [["uu0", "uu1"], ["UU0", "UU1"]],
    "y": [["ii0", "ii1"], ["II0", "II1"]],
    "a": ["a", "A"],
    "u": [["u0", "u1"], ["U0", "U1"]],
    "i": [["i0", "i1"], ["I0", "I1"]],
}

nunationMap = {"F": [["a", "n"], ["A", "n"]], "N": [["u1", "n"], ["U1", "n"]], "K": [["i1", "n"], ["I1", "n"]]}

diacritics = ["o", "a", "u", "i", "F", "N", "K", "~"]
diacriticsWithoutShadda = ["o", "a", "u", "i", "F", "N", "K"]
emphatics = ["D", "S", "T", "Z", "g", "x", "q"]
forwardEmphatics = ["g", "x"]
consonants = [
    ">",
    "<",
    "}",
    "&",
    "'",
    "b",
    "t",
    "^",
    "j",
    "H",
    "x",
    "d",
    "*",
    "r",
    "z",
    "s",
    "$",
    "S",
    "D",
    "T",
    "Z",
    "E",
    "g",
    "f",
    "q",
    "k",
    "l",
    "m",
    "n",
    "h",
    "|",
]

# ------------------------------------------------------------------------------------
# Words with fixed irregular pronunciations-------------------------------------------
# ------------------------------------------------------------------------------------
fixedWords = {
    "h*A": [
        "h aa * aa",
        "h aa * a",
    ],
    "bh*A": [
        "b i0 h aa * aa",
        "b i0 h aa * a",
    ],
    "kh*A": [
        "k a h aa * aa",
        "k a h aa * a",
    ],
    "fh*A": [
        "f a h aa * aa",
        "f a h aa * a",
    ],
    "h*h": ["h aa * i0 h i0", "h aa * i1 h"],
    "bh*h": ["b i0 h aa * i0 h i0", "b i0 h aa * i1 h"],
    "kh*h": ["k a h aa * i0 h i0", "k a h aa * i1 h"],
    "fh*h": ["f a h aa * i0 h i0", "f a h aa * i1 h"],
    "h*An": ["h aa * aa n i0", "h aa * aa n"],
    "h&lA'": ["h aa < u0 l aa < i0", "h aa < u0 l aa <"],
    "*lk": ["* aa l i0 k a", "* aa l i0 k"],
    "b*lk": ["b i0 * aa l i0 k a", "b i0 * aa l i0 k"],
    "k*lk": ["k a * aa l i0 k a", "k a * aa l i1 k"],
    "*lkm": "* aa l i0 k u1 m",
    ">wl}k": ["< u0 l aa < i0 k a", "< u0 l aa < i1 k"],
    "Th": "T aa h a",
    "lkn": ["l aa k i0 nn a", "l aa k i1 n"],
    "lknh": "l aa k i0 nn a h u0",
    "lknhm": "l aa k i0 nn a h u1 m",
    "lknk": ["l aa k i0 nn a k a", "l aa k i0 nn a k i0"],
    "lknkm": "l aa k i0 nn a k u1 m",
    "lknkmA": "l aa k i0 nn a k u0 m aa",
    "lknnA": "l aa k i0 nn a n aa",
    "AlrHmn": ["rr a H m aa n i0", "rr a H m aa n"],
    "Allh": ["ll aa h i0", "ll aa h", "ll AA h u0", "ll AA h a", "ll AA h", "ll A"],
    "h*yn": ["h aa * a y n i0", "h aa * a y n"],
    "wh*A": [
        "w a h aa * aa",
        "w a h aa * a",
    ],
    "wbh*A": [
        "w a b i0 h aa * aa",
        "w a b i0 h aa * a",
    ],
    "wkh*A": [
        "w a k a h aa * aa",
        "w a k a h aa * a",
    ],
    "wh*h": ["w a h aa * i0 h i0", "w a h aa * i1 h"],
    "wbh*h": ["w a b i0 h aa * i0 h i0", "w a b i0 h aa * i1 h"],
    "wkh*h": ["w a k a h aa * i0 h i0", "w a k a h aa * i1 h"],
    "wh*An": ["w a h aa * aa n i0", "w a h aa * aa n"],
    "wh&lA'": ["w a h aa < u0 l aa < i0", "w a h aa < u0 l aa <"],
    "w*lk": ["w a * aa l i0 k a", "w a * aa l i0 k"],
    "wb*lk": ["w a b i0 * aa l i0 k a", "w a b i0 * aa l i0 k"],
    "wk*lk": ["w a k a * aa l i0 k a", "w a k a * aa l i1 k"],
    "w*lkm": "w a * aa l i0 k u1 m",
    "w>wl}k": ["w a < u0 l aa < i0 k a", "w a < u0 l aa < i1 k"],
    "wTh": "w a T aa h a",
    "wlkn": ["w a l aa k i0 nn a", "w a l aa k i1 n"],
    "wlknh": "w a l aa k i0 nn a h u0",
    "wlknhm": "w a l aa k i0 nn a h u1 m",
    "wlknk": ["w a l aa k i0 nn a k a", "w a l aa k i0 nn a k i0"],
    "wlknkm": "w a l aa k i0 nn a k u1 m",
    "wlknkmA": "w a l aa k i0 nn a k u0 m aa",
    "wlknnA": "w a l aa k i0 nn a n aa",
    "wAlrHmn": ["w a rr a H m aa n i0", "w a rr a H m aa n"],
    "wAllh": ["w a ll aa h i0", "w a ll aa h", "w a ll AA h u0", "w a ll AA h a", "w a ll AA h", "w a ll A"],
    "wh*yn": ["w a h aa * a y n i0", "w a h aa * a y n"],
    "w": ["w a"],
    "Aw": ["< a w"],
    ">w": ["< a w"],
    "Alf": ["< a l f"],
    ">lf": ["< a l f"],
    "b>lf": ["b i0 < a l f"],
    "f>lf": ["f a < a l f"],
    "wAlf": ["w a < a l f"],
    "w>lf": ["w a < a l f"],
    "wb>lf": ["w a b i0 < a l f"],
    "nt": "n i1 t",
    "fydyw": "v i0 d y uu1",
    "lndn": "l A n d u1 n",
}


def isFixedWord(word, results, orthography, pronunciations):
    lastLetter = ""
    if len(word) > 0:
        lastLetter = word[-1]
    if lastLetter == "a":
        lastLetter = ["a", "A"]
    elif lastLetter == "A":
        lastLetter = ["aa"]
    elif lastLetter == "u":
        lastLetter = ["u0"]
    elif lastLetter == "i":
        lastLetter = ["i0"]
    elif lastLetter in unambiguousConsonantMap:
        lastLetter = [unambiguousConsonantMap[lastLetter]]
    wordConsonants = re.sub("[^h*Ahn'>wl}kmyTtfdb]", "", word)  # Remove all dacritics from word
    if wordConsonants in fixedWords:  # check if word is in the fixed word lookup table
        if isinstance(fixedWords[wordConsonants], list):
            done = False
            for pronunciation in fixedWords[wordConsonants]:
                if pronunciation.split(" ")[-1] in lastLetter:
                    results += (
                        word + " " + pronunciation + "\n"
                    )  # add each pronunciation to the pronunciation dictionary
                    pronunciations.append(pronunciation.split(" "))
                    done = True
            if not done:
                results += (
                    word + " " + fixedWords[wordConsonants][0] + "\n"
                )  # add each pronunciation to the pronunciation dictionary
                pronunciations.append(fixedWords[wordConsonants][0].split(" "))
        else:
            results += (
                word + " " + fixedWords[wordConsonants] + "\n"
            )  # add pronunciation to the pronunciation dictionary
            pronunciations.append(fixedWords[wordConsonants].split(" "))
    return results


def phonetise(text):
    utterances = text.splitlines()
    result = ""  # Pronunciations Dictionary
    utterancesPronuncations = []  # Most likely pronunciation for all utterances
    utterancesPronuncationsWithBoundaries = []  # Most likely pronunciation for all utterances

    # -----------------------------------------------------------------------------------------------------
    # Loop through utterances------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    utteranceNumber = 1
    for utterance in utterances:
        utteranceNumber += 1
        utterancesPronuncations.append("")  # Add empty entry that will hold this utterance's pronuncation
        utterancesPronuncationsWithBoundaries.append("")  # Add empty entry that will hold this utterance's pronuncation

        utterance = arabicToBuckwalter(utterance)
        # print(u"phoetising utterance")
        # print(utterance)
        # Do some normalisation work and split utterance to words
        utterance = utterance.replace("AF", "F")
        utterance = utterance.replace("\u0640", "")
        utterance = utterance.replace("o", "")
        utterance = utterance.replace("aA", "A")
        utterance = utterance.replace("aY", "Y")
        utterance = re.sub("([^\\-]) A", "\\1 ", utterance)
        utterance = utterance.replace("F", "an")
        utterance = utterance.replace("N", "un")
        utterance = utterance.replace("K", "in")
        utterance = utterance.replace("|", ">A")

        # Deal with Hamza types that when not followed by a short vowel letter,
        # this short vowel is added automatically
        utterance = re.sub("^Ai", "<i", utterance)
        utterance = re.sub("^Aa", ">a", utterance)
        utterance = re.sub("^Au", ">u", utterance)
        utterance = re.sub("Ai", "<i", utterance)
        utterance = re.sub("Aa", ">a", utterance)
        utterance = re.sub("Au", ">u", utterance)
        utterance = re.sub("^Al", ">al", utterance)
        utterance = re.sub(" - Al", " - >al", utterance)
        utterance = re.sub("^- Al", "- >al", utterance)
        utterance = re.sub("^>([^auAw])", ">a\\1", utterance)
        utterance = re.sub(" >([^auAw ])", " >a\\1", utterance)
        utterance = re.sub("<([^i])", "<i\\1", utterance)
        utterance = re.sub(" A([^aui])", " \\1", utterance)
        utterance = re.sub("^A([^aui])", "\\1", utterance)

        utterance = utterance.split(" ")
        # ---------------------------
        wordIndex = -1

        # Loop through words
        for word in utterance:
            wordIndex += 1
            if not word in ["-", "sil"]:
                pronunciations = []  # Start with empty set of possible pronunciations of current word
                result = isFixedWord(
                    word, result, word, pronunciations
                )  # Add fixed irregular pronunciations if possible

                emphaticContext = (
                    False  # Indicates whether current character is in an emphatic context or not. Starts with False
                )
                word = "bb" + word + "ee"  # This is the end/beginning of word symbol. just for convenience

                phones = []  # Empty list which will hold individual possible word's pronunciation

                # -----------------------------------------------------------------------------------
                # MAIN LOOP: here is where the Modern Standard Arabic phonetisation rule-set starts--
                # -----------------------------------------------------------------------------------
                for index in range(2, len(word) - 2):
                    letter = word[index]  # Current Character
                    letter1 = word[index + 1]  # Next Character
                    letter2 = word[index + 2]  # Next-Next Character
                    letter_1 = word[index - 1]  # Previous Character
                    letter_2 = word[index - 2]  # Before Previous Character
                    # ----------------------------------------------------------------------------------------------------------------
                    if letter in consonants + ["w", "y"] and not letter in emphatics + [
                        "r" """, u'l'"""
                    ]:  # non-emphatic consonants (except for Lam and Ra) change emphasis back to False
                        emphaticContext = False
                    if letter in emphatics:  # Emphatic consonants change emphasis context to True
                        emphaticContext = True
                    if (
                        letter1 in emphatics and not letter1 in forwardEmphatics
                    ):  # If following letter is backward emphatic, emphasis state is set to True
                        emphaticContext = True
                    # ----------------------------------------------------------------------------------------------------------------
                    # ----------------------------------------------------------------------------------------------------------------
                    if (
                        letter in unambiguousConsonantMap
                    ):  # Unambiguous consonant phones. These map to a predetermined phoneme
                        phones += [unambiguousConsonantMap[letter]]
                    # ----------------------------------------------------------------------------------------------------------------
                    if letter == "l":  # Lam is a consonant which requires special treatment
                        if (
                            (not letter1 in diacritics and not letter1 in vowelMap)
                            and letter2 in ["~"]
                            and (
                                (letter_1 in ["A", "l", "b"])
                                or (letter_1 in diacritics and letter_2 in ["A", "l", "b"])
                            )
                        ):  # Lam could be omitted in definite article (sun letters)
                            phones += [ambiguousConsonantMap["l"][1]]  # omit
                        else:
                            phones += [ambiguousConsonantMap["l"][0]]  # do not omit
                    # ----------------------------------------------------------------------------------------------------------------
                    if (
                        letter == "~" and not letter_1 in ["w", "y"] and len(phones) > 0
                    ):  # shadda just doubles the letter before it
                        phones[-1] += phones[-1]
                    # ----------------------------------------------------------------------------------------------------------------
                    if letter == "|":  # Madda only changes based in emphaticness
                        if emphaticContext:
                            phones += [maddaMap["|"][1]]
                        else:
                            phones += [maddaMap["|"][0]]
                    # ----------------------------------------------------------------------------------------------------------------
                    if letter == "p":  # Ta' marboota is determined by the following if it is a diacritic or not
                        if letter1 in diacritics:
                            phones += [ambiguousConsonantMap["p"][0]]
                        else:
                            phones += [ambiguousConsonantMap["p"][1]]
                    # ----------------------------------------------------------------------------------------------------------------
                    if letter in vowelMap:
                        if letter in [
                            "w",
                            "y",
                        ]:  # Waw and Ya are complex they could be consonants or vowels and their gemination is complex as it could be a combination of a vowel and consonants
                            if (
                                letter1 in diacriticsWithoutShadda + ["A", "Y"]
                                or (letter1 in ["w", "y"] and not letter2 in diacritics + ["A", "w", "y"])
                                or (letter_1 in diacriticsWithoutShadda and letter1 in consonants + ["e"])
                            ):
                                if (letter in ["w"] and letter_1 in ["u"] and not letter1 in ["a", "i", "A", "Y"]) or (
                                    letter in ["y"] and letter_1 in ["i"] and not letter1 in ["a", "u", "A", "Y"]
                                ):
                                    if emphaticContext:
                                        phones += [vowelMap[letter][1][0]]
                                    else:
                                        phones += [vowelMap[letter][0][0]]
                                else:
                                    if letter1 in ["A"] and letter in ["w"] and letter2 in ["e"]:
                                        phones += [[vowelMap[letter][0][0], ambiguousConsonantMap[letter]]]
                                    else:
                                        phones += [ambiguousConsonantMap[letter]]
                            elif letter1 in ["~"]:
                                if (
                                    letter_1 in ["a"]
                                    or (letter in ["w"] and letter_1 in ["i", "y"])
                                    or (letter in ["y"] and letter_1 in ["w", "u"])
                                ):
                                    phones += [ambiguousConsonantMap[letter], ambiguousConsonantMap[letter]]
                                else:
                                    phones += [vowelMap[letter][0][0], ambiguousConsonantMap[letter]]
                            else:  # Waws and Ya's at the end of the word could be shortened
                                if emphaticContext:
                                    if letter_1 in consonants + ["u", "i"] and letter1 in ["e"]:
                                        phones += [[vowelMap[letter][1][0], vowelMap[letter][1][0][1:]]]
                                    else:
                                        phones += [vowelMap[letter][1][0]]
                                else:
                                    if letter_1 in consonants + ["u", "i"] and letter1 in ["e"]:
                                        phones += [[vowelMap[letter][0][0], vowelMap[letter][0][0][1:]]]
                                    else:
                                        phones += [vowelMap[letter][0][0]]
                        if letter in ["u", "i"]:  # Kasra and Damma could be mildened if before a final silent consonant
                            if emphaticContext:
                                if (
                                    (letter1 in unambiguousConsonantMap or letter1 == "l")
                                    and letter2 == "e"
                                    and len(word) > 7
                                ):
                                    phones += [vowelMap[letter][1][1]]
                                else:
                                    phones += [vowelMap[letter][1][0]]
                            else:
                                if (
                                    (letter1 in unambiguousConsonantMap or letter1 == "l")
                                    and letter2 == "e"
                                    and len(word) > 7
                                ):
                                    phones += [vowelMap[letter][0][1]]
                                else:
                                    phones += [vowelMap[letter][0][0]]
                        if letter in [
                            "a",
                            "A",
                            "Y",
                        ]:  # Alif could be ommited in definite article and beginning of some words
                            if letter in ["A"] and letter_1 in ["w", "k"] and letter_2 == "b" and letter1 in ["l"]:
                                phones += [["a", vowelMap[letter][0][0]]]
                            elif letter in ["A"] and letter_1 in ["u", "i"]:
                                temp = True  # do nothing
                            elif (
                                letter in ["A"] and letter_1 in ["w"] and letter1 in ["e"]
                            ):  # Waw al jama3a: The Alif after is optional
                                phones += [[vowelMap[letter][0][1], vowelMap[letter][0][0]]]
                            elif letter in ["A", "Y"] and letter1 in ["e"]:
                                if emphaticContext:
                                    phones += [[vowelMap[letter][1][0], vowelMap["a"][1]]]
                                else:
                                    phones += [[vowelMap[letter][0][0], vowelMap["a"][0]]]
                            else:
                                if emphaticContext:
                                    phones += [vowelMap[letter][1][0]]
                                else:
                                    phones += [vowelMap[letter][0][0]]
                # -------------------------------------------------------------------------------------------------------------------------
                # End of main loop---------------------------------------------------------------------------------------------------------
                # -------------------------------------------------------------------------------------------------------------------------
                possibilities = 1  # Holds the number of possible pronunciations of a word

                # count the number of possible pronunciations
                for letter in phones:
                    if isinstance(letter, list):
                        possibilities = possibilities * len(letter)

                # Generate all possible pronunciations
                for i in range(0, possibilities):
                    pronunciations.append([])
                    iterations = 1
                    for index, letter in enumerate(phones):
                        if isinstance(letter, list):
                            curIndex = int((i / iterations) % len(letter))
                            if letter[curIndex] != "":
                                pronunciations[-1].append(letter[curIndex])
                            iterations = iterations * len(letter)
                        else:
                            if letter != "":
                                pronunciations[-1].append(letter)

                # Iterate through each pronunciation to perform some house keeping. And append pronunciation to dictionary
                # 1- Remove duplicate vowels
                # 2- Remove duplicate y and w
                for pronunciation in pronunciations:
                    prevLetter = ""
                    toDelete = []
                    for i in range(0, len(pronunciation)):
                        letter = pronunciation[i]
                        if (
                            letter in ["aa", "uu0", "ii0", "AA", "UU0", "II0"]
                            and prevLetter.lower() == letter[1:].lower()
                        ):  # Delete duplicate consecutive vowels
                            toDelete.append(i - 1)
                            pronunciation[i] = pronunciation[i - 1][0] + pronunciation[i - 1]
                        if letter in ["u0", "i0"] and prevLetter.lower() == letter.lower():  # Delete duplicates
                            toDelete.append(i - 1)
                            pronunciation[i] = pronunciation[i - 1]
                        if letter in ["y", "w"] and prevLetter == letter:  # delete duplicate
                            pronunciation[i - 1] += pronunciation[i - 1]
                            toDelete.append(i)
                        if letter in ["a"] and prevLetter == letter:  # delete duplicate
                            toDelete.append(i)

                        prevLetter = letter
                    for i in reversed(range(0, len(toDelete))):
                        del pronunciation[toDelete[i]]
                    result += word[2:-2] + " " + " ".join(pronunciation) + "\n"

                # Append utterance pronunciation to utterancesPronunciations
                utterancesPronuncations[-1] += " " + " ".join(pronunciations[0])

                # Add Stress to each pronunciation
                pIndex = 0
                for pronunciation in pronunciations:
                    stressIndex = findStressIndex(pronunciation)
                    if stressIndex < len(pronunciation) and stressIndex != -1:
                        pronunciation[stressIndex] += "'"
                    else:
                        if pIndex == 0:
                            print("skipped")
                            print(pronunciation)
                    pIndex += 1
                # Append utterance pronunciation to utterancesPronunciations
                utterancesPronuncationsWithBoundaries[-1] += " " + "".join(pronunciations[0])
            else:
                utterancesPronuncations[-1] += " sil"
                utterancesPronuncationsWithBoundaries[-1] += " sil"

        # Add sound file name back
        utterancesPronuncations[-1] = utterancesPronuncations[-1].strip() + " sil"
        utterancesPronuncationsWithBoundaries[-1] = utterancesPronuncationsWithBoundaries[-1].strip() + " sil"

    return utterancesPronuncations


# -----------------------------------------------------------------------------------------------------
# Read input file--------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        inputFileName = sys.argv[1]
    except:
        print("No input file provided")
        sys.exit()

    inputFile = codecs.open(inputFileName, "r", "utf-8")
    (utterancesPronuncationsWithBoundaries, utterancesPronuncations, dict) = phonetise(inputFile.read())
    inputFile.close()

    # ----------------------------------------------------------------------------
    # Save output-----------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # Save Utterances pronunciations
    outFile = codecs.open("utterance-pronunciations.txt", "w", "utf-8")
    outFile.write("\n".join(utterancesPronuncations))
    outFile.close()
    # Save Utterances pronunciations (with wordboundaries)
    outFile = codecs.open("utterance-pronunciations-with-boundaries.txt", "w", "utf-8")
    outFile.write("\n".join(utterancesPronuncationsWithBoundaries))
    outFile.close()

    # Save Pronunciation Dictionary
    outFile = codecs.open("dict", "w", "utf-8")
    outFile.write(dict.rstrip())
    outFile.close()

    # Sort Dictionary
    os.system("sortandfilter.py dict")
