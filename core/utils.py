"""
This file implements general utilities.
"""

import os
import shutil
import glob
import csv
import librosa
import time
import numpy as np
import zipfile
import soundfile as sf




def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_recursive_wavfile_list(folder, ext='wav'):
    files = []
    for filename in sorted(glob.iglob(folder + '/**/*.' + ext, recursive=True)):
        files.append(filename)
    return files


def zipdir(path, ziph_name):
    ziph = zipfile.ZipFile(ziph_name, 'w', zipfile.ZIP_DEFLATED)
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
    ziph.close()


class ApplyEstimatorToSignal:

    def __init__(self, signal,
                 sample_rate,
                 feature_extractor,
                 keras_model,
                 hop_size_ms,
                 zero_pad=True,
                 synthetic_noise_floor_dB=-80):
        self.sample_rate = sample_rate
        self.feature_extractor = feature_extractor
        self.frame_size = self.feature_extractor.get_input_shape()[0]
        self.hop_size = int(hop_size_ms*self.feature_extractor.get_sample_rate())
        if zero_pad:
            silence = 0*np.random.normal(loc=0.0, scale=10 ** (synthetic_noise_floor_dB / 20), size=(self.frame_size,))
            self.signal = np.concatenate((silence, signal))
        else:
            self.signal = signal
        self.model = keras_model
        self.time_sec_per_frame = 'Unknown'

    def run(self):
        chunks = librosa.util.frame(self.signal, frame_length=self.frame_size, hop_length=self.hop_size)
        output = np.zeros(chunks.shape[1])
        ave_time = 0
        for i in range(chunks.shape[1]):
            a = time.time()
            features = self.feature_extractor.extract(np.expand_dims(chunks[:, i], axis=1))
            output[i] = self.model.predict(np.expand_dims(features, axis=0))
            b = time.time()
            ave_time = ave_time + b - a
        self.time_sec_per_frame = ave_time / chunks.shape[1]

        return output

    def get_time_seconds_per_frame(self):
        return self.time_sec_per_frame



def measure_length(all_files):

    min_len_sec = 4.0
    count = 0
    ave_sec = 0
    for file in all_files:
        x, orig_sr = sf.read(file)
        len_sec = len(x)/orig_sr
        if len_sec > min_len_sec:
            print(file, len_sec)
            ave_sec += len_sec
            count = count + 1

    cum_sec = ave_sec
    ave_sec /= count
    print(cum_sec, ave_sec)


def split_each_list(list_list):
    stuff = [[], [], []]
    for sub_list in list_list:
        train_, val_, test_ = split_list(sub_list)
        stuff[0] = stuff[0] + train_
        stuff[1] = stuff[1] + val_
        stuff[2] = stuff[2] + test_

    np.random.shuffle(stuff[0])
    np.random.shuffle(stuff[1])
    np.random.shuffle(stuff[2])
    return stuff

def split_list(my_list, split=.8):
    x = my_list.copy()
    np.random.shuffle(x)

    # Partition
    train_val_end = np.round(len(x)*split).astype('int')
    train_end = np.round(train_val_end*split).astype('int')
    train_slice = slice(0, train_end)
    val_slice = slice(train_end, train_val_end)
    test_slice = slice(train_val_end, len(x))

    return x[train_slice], x[val_slice], x[test_slice]

def get_mix50_csv_info(csv_filepath):
    label_map = {}
    with open(csv_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                label_map[row[0]] = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    return label_map

# Utility method to wrangle dimensions of mean and std
def fix_dims(x):

    if len(x.shape) == 1:
        x = np.expand_dims(np.expand_dims(x, axis=1), axis=2)
    elif len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    else:
        pass

    return x
