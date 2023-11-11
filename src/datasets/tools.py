import os
import string
import numpy as np


def condense_duplicates(lst):
    changed_positions = np.append(0, np.nonzero(np.array(lst[:-1]) != np.array(lst[1:]))[0] + 1)
    return np.array(lst)[changed_positions]


def parse_info_name(path):
    name = os.path.splitext(os.path.split(path)[-1])[0]
    info = {}
    current_letter = None
    for letter in name:
        if letter in string.ascii_letters:
            info[letter] = []
            current_letter = letter
        else:
            info[current_letter].append(letter)
    for key in info.keys():
        info[key] = "".join(info[key])
    return info
