import numpy as np
from params import *
from lib import umbrella_get_letters, read_and_extract_threshold
from tensorflow import keras
import pandas as pd
# import json
import os
import sys
import pickle
from tqdm import tqdm

sys.setrecursionlimit(10000)

with open('classes.pickle', 'rb') as fp:
    classes = pickle.load(fp)

model = keras.models.load_model(MODEL_PATH)


def decaptcha(file_paths):
    all_letters = []

    # preprocessing steps:
    lower_threshold_letter, upper_threshold_letter = read_and_extract_threshold(
        file_paths[0])

    if verbose:
        sys.stderr.write("Starting preprocessing steps...\n")
    if verbose:
        for path_ in tqdm(file_paths):
            # will receive 3 letters preprocessed
            letters = umbrella_get_letters(
                path_, lower_threshold_letter, upper_threshold_letter)
            all_letters.extend(letters)
    else:
        for path_ in file_paths:
            # will receive 3 letters preprocessed
            letters = umbrella_get_letters(path_)
            all_letters.extend(letters)
    all_letters = np.array(all_letters)
    if verbose:
        sys.stderr.write("...Done with preprocessing\n")

    all_letters = np.expand_dims(all_letters, -1)  # for channel dimension
    logits = model.predict(all_letters, verbose=verbose)
    indices = np.argmax(logits, axis=1)
    answers = [classes[index] for index in indices]
    final_answers = []
    for idx in range(0, len(answers), 3):
        final_answers.append(
            ','.join([answers[idx], answers[idx+1], answers[idx+2]]))
    return final_answers


if __name__ == '__main__':
    files = os.listdir('./test/')
    random_numbers = np.random.randint(0, len(files)-1, num_files)
    paths = [DATA_PATH + str(random_number) +
             '.png' for random_number in random_numbers]
    result = decaptcha(paths)

    # uncomment for seeing results:
    df = pd.read_csv(LABELS_PATH, header=None)
    df['file'] = df.index
    df['file'] = df['file'].apply(lambda x: str(x)+'.png')
    df['labels'] = df[[0, 1, 2]].apply(lambda row: ','.join(row), axis=1)
    counter = 0
    for idx in range(len(result)):
        print(paths[idx], ":", result[idx])
        if df.iloc[random_numbers[idx]]['labels'] != result[idx]:
            counter += 1
    print("Accuracy:", (len(result) - counter)/len(result))
