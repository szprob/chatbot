import os
import pickle


def split_list_by2(lst):
    result = []
    for i in range(2, len(lst) + 1, 2):
        result.append(lst[:i])
    return result


def save_pkl(path, file):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file
