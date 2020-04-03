import os
from tensorflow import keras
import numpy as np
import scipy as sp
import soundfile as sf
import glob
import csv
import h5py
import core.features as va_features
import core.utils as va_utils


from enum import Enum

"""Basic data generator that assumes files are pre-mixed and have a CSV file of labels"""
class PreMixedAcousticScene(keras.utils.Sequence):

    def __init__(self, dataset_path, feature_extractor, stats_name="stats.csv", label='t60', fs=16000, batch_size=32):
        self.dataset_path = dataset_path
        self.fs = fs
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.files = va_utils.get_recursive_wavfile_list(self.dataset_path)
        self.instance_len_sec = feature_extractor.get_input_shape()[0]/feature_extractor.get_sample_rate()

        # read in the csv file
        if label == 'sti':
            index = 1
        elif label == 't60':
            index = 2
        elif label == 'drr':
            index = 3
        elif label == 'snr':
            index = 4
        else:
            assert 0, 'Invalid label'

        csv_filepath = os.path.join(self.dataset_path, stats_name)
        label_map = {}
        with open(csv_filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    line_count += 1
                    label_map[row[0]] = float(row[index])
        self.label_map = label_map

    def __len__(self):
        """ Returns the number of batches per epoch """
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """

        # Randomly permute the file order
        self.files = np.random.permutation(self.files)

    def find_shorter_sample(self, mix):

        if int(self.instance_len_sec*self.fs) == len(mix):
            return mix[:int(self.instance_len_sec*self.fs)]

        threshold = -10
        full_energy = 10*np.log10(np.mean(np.square(mix)))

        sample_len = int(self.instance_len_sec * self.fs)
        full_len = len(mix)

        loop_count = 0
        loop_count_max = 100
        while True:
            start_ind = np.random.randint(low=0, high=full_len - sample_len)
            end_ind = start_ind + sample_len
            chunk = mix[start_ind:end_ind]


            chunk_energy = 10 * np.log10(np.mean(np.square(chunk)))
            if chunk_energy > full_energy + threshold:
                return chunk
            loop_count += 1

            # Give up
            if loop_count > loop_count_max:
                print('Warning: low level speech samples')
                return chunk

    def generate_samples(self, num_samples):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)
        # TODO: eliminate the copied code between this and the __get_item__ method
        self.files = np.random.permutation(self.files)
        if num_samples < 0:
            num_samples = len(self.files)
        inds = list(range(num_samples))
        batch_x = np.zeros((num_samples,) + self.feature_extractor.get_output_shape())
        batch_y = np.zeros(num_samples)
        for _i in range(num_samples):
            # Read in the mixture file
            mix, fs_in = sf.read(self.files[inds[_i]])
            assert fs_in == self.fs

            mix = self.find_shorter_sample(mix)

            # Extract features
            batch_x[_i] = np.expand_dims(self.feature_extractor.extract(np.expand_dims(mix, axis=1)), axis=0)

            # Extract label
            batch_y[_i] = self.label_map[os.path.basename(self.files[inds[_i]])]

        return batch_x, batch_y

    def __getitem__(self, index):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)

        inds = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        batch_x = np.zeros((self.batch_size,) + self.feature_extractor.get_output_shape())
        batch_y = np.zeros(self.batch_size)
        for _i in range(self.batch_size):

            # Read in the mixture file
            mix, fs_in = sf.read(self.files[inds[_i]])
            assert fs_in == self.fs

            mix = self.find_shorter_sample(mix)

            # Extract features
            batch_x[_i] = np.expand_dims(self.feature_extractor.extract(np.expand_dims(mix, axis=1)), axis=0)

            # Extract label
            batch_y[_i] = self.label_map[os.path.basename(self.files[inds[_i]])]

        return batch_x, batch_y


"""Basic data generator that assumes files are pre-mixed and have a CSV file of labels"""
class PreMixedAcousticSceneWASPAAH5(keras.utils.Sequence):

    def __init__(self, dataset_h5_path, partition, feature_extractor, label='t60', batch_size=32):
        """Constructor"""
        self.batch_size = batch_size
        self.extractor = feature_extractor
        self.instance_len_sec = feature_extractor.get_input_shape()[0]/feature_extractor.get_sample_rate()
        self.h5_dataset = h5py.File(dataset_h5_path, 'r')
        self.datah5 = self.h5_dataset[partition]
        self.all_inds = list(range(self.h5_dataset[partition]['audio'].shape[0]))
        self.label = label
        self.precompute_features = False

        if self.precompute_features:
            # convert H5 to numpy array
            raw_audio = self.datah5['audio'].value
            samples = raw_audio.shape[0]

            self.feature_audio = np.zeros((samples,) + self.extractor.get_output_shape())
            self.labels = self.datah5[label].value

            # Precompute features
            print("Pre-computing features")
            for _i in self.all_inds:
                self.feature_audio[_i] = np.expand_dims(self.extractor.extract(np.expand_dims(raw_audio[_i, :].T, axis=1)), axis=0)
            print("Done: Pre-computing features")

    def __del__(self):
        """Destructor"""
        # self.h5_dataset.close()

    def __len__(self):
        """ Returns the number of batches per epoch """
        return int(np.floor(len(self.all_inds) / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.all_inds = np.random.permutation(self.all_inds)

    def _collect_samples(self, inds):

        if self.precompute_features:
            batch_x = self.feature_audio[inds,]
            batch_y = self.labels[inds]
        else:
            num_samples = len(inds)
            batch_x = np.zeros((num_samples,) + self.extractor.get_output_shape())
            batch_y = np.zeros(num_samples)
            for ind, _i in zip(inds, range(self.batch_size)):

                mix = self.datah5['audio'][ind]

                # Extract features & label
                batch_x[_i] = np.expand_dims(self.extractor.extract(np.expand_dims(mix, axis=1)), axis=0)
                batch_y[_i] = self.datah5[self.label][ind]


        return batch_x, batch_y

    def generate_samples(self, num_samples):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)
        self.all_inds = np.random.permutation(self.all_inds)
        if num_samples < 0:
            num_samples = len(self.all_inds)
        inds = list(range(num_samples))
        return self._collect_samples(inds)

    def __getitem__(self, index):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)
        num_samples = self.batch_size
        inds = list(range(index * num_samples, (index + 1) * num_samples))
        return self._collect_samples(inds)

"""Basic data generator that assumes files are pre-mixed and have a CSV file of labels"""
class PreMixedAcousticSceneWASPAA(keras.utils.Sequence):

    def __init__(self, dataset_path, feature_extractor, label='t60', fs=16000, batch_size=32, bands=None, stats_name="stats.csv"):
        self.dataset_path = dataset_path
        self.fs = fs
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        # self.files = va_utils.get_recursive_wavfile_list(self.dataset_path)
        self.files = []
        self.instance_len_sec = feature_extractor.get_input_shape()[0]/feature_extractor.get_sample_rate()

        # read in the csv file
        if label == 't60':
            self.index = 1
        elif label == 'drr':
            self.index = 2
        elif label == 'subbands':
            self.index = bands
        else:
            assert 0, 'Invalid label'

        csv_filepath = os.path.join(self.dataset_path, stats_name)
        label_map = {}
        if os.path.exists(csv_filepath):
            with open(csv_filepath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        line_count += 1
                        self.files.append(os.path.join(self.dataset_path, row[0]))
                        label_map[row[0]] = [float(row[ind]) for ind in np.atleast_1d(self.index)]
        else:
            for r, d, f in os.walk(self.dataset_path):
                for file in f:
                    if file.endswith('wav'):
                        self.files.append(os.path.join(r, file))

        self.label_map = label_map

    def __len__(self):
        """ Returns the number of batches per epoch """
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """

        # Randomly permute the file order
        self.files = np.random.permutation(self.files)

    def find_shorter_sample(self, mix):

        if int(self.instance_len_sec*self.fs) == len(mix):
            return mix[:int(self.instance_len_sec*self.fs)]

        threshold = -10
        full_energy = 10*np.log10(np.mean(np.square(mix)))

        sample_len = int(self.instance_len_sec * self.fs)
        full_len = len(mix)

        loop_count = 0
        loop_count_max = 100
        while True:
            start_ind = np.random.randint(low=0, high=full_len - sample_len)
            end_ind = start_ind + sample_len
            chunk = mix[start_ind:end_ind]


            chunk_energy = 10 * np.log10(np.mean(np.square(chunk)))
            if chunk_energy > full_energy + threshold:
                return chunk
            loop_count += 1

            # Give up
            if loop_count > loop_count_max:
                print('Warning: low level speech samples')
                return chunk

    def generate_samples(self, num_samples):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)

        self.files = np.random.permutation(self.files)

        if num_samples < 0:
            num_samples = len(self.files)
        if num_samples > len(self.files):
            num_samples = len(self.files)
        inds = list(range(num_samples))
        batch_x = np.zeros((num_samples,) + self.feature_extractor.get_output_shape())
        batch_y = np.zeros((num_samples, len(np.atleast_1d(self.index))))
        for _i in range(num_samples):
            # Read in the mixture file
            mix, fs_in = sf.read(self.files[inds[_i]])
            assert fs_in == self.fs

            mix = self.find_shorter_sample(mix)

            # Extract features
            batch_x[_i] = np.expand_dims(self.feature_extractor.extract(np.expand_dims(mix, axis=1)), axis=0)

            # Extract label
            batch_y[_i] = self.label_map.get(os.path.basename(self.files[inds[_i]]))

        return batch_x, batch_y

    def __getitem__(self, index):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)

        inds = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        batch_x = np.zeros((self.batch_size,) + self.feature_extractor.get_output_shape())
        batch_y = np.zeros((self.batch_size, len(np.atleast_1d(self.index))))
        for _i in range(self.batch_size):

            # Read in the mixture file
            mix, fs_in = sf.read(self.files[inds[_i]])
            assert fs_in == self.fs

            mix = self.find_shorter_sample(mix)

            # Extract features
            batch_x[_i] = np.expand_dims(self.feature_extractor.extract(np.expand_dims(mix, axis=1)), axis=0)

            # Extract label
            batch_y[_i] = self.label_map.get(os.path.basename(self.files[inds[_i]]))

        return batch_x, batch_y


"""Basic data generator that generates features and labels from a pre-computed h5py file"""
class PreFeatureGenerator(keras.utils.Sequence):

    def __init__(self, dataset, partition, label='modulation', normalize_band=None, batch_size=32, bands=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = bands
        self.partition = partition
        self.label = label
        self.norm_band = normalize_band
        # self.shape = self.dataset[partition]['audio'][0].shape

    def __len__(self):
        """ Returns the number of batches per epoch """
        return int(np.floor(len(self.dataset[self.partition][self.label]) / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """

    def generate_samples(self, num_samples):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)
        if num_samples < 0:
            num_samples = len(self.dataset[self.partition][self.label])
        else:
            num_samples = min(num_samples, len(self.dataset[self.partition][self.label]))
        batch_x = self.dataset[self.partition]['audio'][:num_samples]
        if self.norm_band:
            batch_y = self.dataset[self.partition][self.label][:num_samples][:, self.index] - \
                      np.expand_dims(self.dataset[self.partition][self.label][:num_samples][:, self.norm_band], axis=1)
        else:
            batch_y = self.dataset[self.partition][self.label][:num_samples][:, self.index]
        speaker = self.dataset[self.partition]['speaker'][:num_samples]
        inputfile = self.dataset[self.partition]['filename'][:num_samples]

        return batch_x, batch_y, speaker, inputfile

    def __getitem__(self, index):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)

        inds = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        batch_x = self.dataset[self.partition]['audio'][inds]
        if self.norm_band:
            batch_y = self.dataset[self.partition][self.label][inds][:, self.index] - \
                      np.expand_dims(self.dataset[self.partition][self.label][inds][:, self.norm_band], axis=1)
        else:
            batch_y = self.dataset[self.partition][self.label][inds][:, self.index]

        return batch_x, batch_y


class AcousticSceneGenerator(keras.utils.Sequence):

    class FeatureMode(Enum):
        kSNR = 0  # TODO: make as mix feature
        kSTI = 1  # TODO: make as mix feature
        kVAD = 2  # TODO: make a speech feature
        kSpeechFeature = 3
        kNoiseFeature = 4
        kRIRFeature = 5

    def __init__(self,  dataset_path,
                        feature_callback=va_features.TimeDomainExtractor(16000, 4.0),
                        batch_size=32,
                        instance_len_sec=4.0,
                        snr_limits=(-5, 40),
                        gain_db_limits=(0, 0),
                        fs=16000,
                        rir_probability=1.0,
                        speech_probability=1.0,
                        feature_mode=FeatureMode.kSTI,
                        feature_map={},
                        dataset_speech_normalization_level=-30,
                        dataset_noise_normalization_level=-30,
                        min_asl_percent=.5,
                        random_seed=0
                        ):

        assert fs == 16000, "Sampling rate must be 16kHz."
        np.random.seed(random_seed)

        # Get file paths
        speech_folder = os.path.join(dataset_path, 'speech')
        rir_folder = os.path.join(dataset_path, 'rir')
        noise_folder = os.path.join(dataset_path, 'noise')
        self.speech_files = self.get_wav_file_list(speech_folder)
        self.rir_files = self.get_wav_file_list(rir_folder)
        self.noise_files = self.get_wav_file_list(noise_folder)
        self.feature_extractor = feature_callback
        self.feature_map = feature_map
        self.feature_mode = feature_mode

        # Store internal params
        self.batch_size = batch_size
        self.instance_len_sec = instance_len_sec
        self.dataset_speech_level = dataset_speech_normalization_level
        self.fixed_noise_level_db = dataset_noise_normalization_level
        self.snr_limits = snr_limits
        self.gain_db_limits = gain_db_limits

        self.fs = 16000
        self.max_length_sec = 4.0
        self.max_rir_length_sec = 3.0
        self.min_asl_percent = min_asl_percent
        self.vad_frame_size = .01
        assert instance_len_sec <= self.max_length_sec

        self.num_speech = len(self.speech_files)
        self.num_noise = len(self.noise_files)
        self.num_rir = len(self.rir_files)

        self.speech_probability = speech_probability
        self.rir_probability = rir_probability
        self.on_epoch_end()

    def __len__(self):
        """ Returns the number of batches per epoch """
        return 100 #int(np.floor(self.num_rir / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        pass

    def get_sampling_rate(self):
        return self.fs

    @staticmethod
    def get_wav_file_list(folder):
        file_list = []
        for filename in glob.iglob(folder + '/**/*.wav', recursive=True):
            file_list.append(filename)
        return file_list


    def generate_samples(self, num_samples):
        x, y = self.__generate_num_samples(num_samples)
        return x, y


    def __get_noise(self, file, segment_len_samples):

        x, fs = sf.read(file)
        assert fs == self.fs, "Must be 16kHz"
        assert len(x) >= segment_len_samples
        x = x[0:segment_len_samples]
        return x

    ## TDOD: have alternative version
    def __generate_speech_with_minimum_threshold(self, trim_len_samples, min_asl_percent=.2):

        count = 0
        while True:
            speech_id = np.random.choice(self.num_speech, replace=False, size=1)[0]

            file = self.speech_files[speech_id]

            uuid = os.path.basename(file)
            folder = os.path.dirname(file)
            csv_file_path = os.path.join(folder, uuid[0:-4] + '.csv')

            vad = []
            with open(csv_file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        vad = np.array(row).astype('float')
            assert len(vad) > 0, "ERROR reading VAD file"

            vad = vad[0:(int(self.instance_len_sec/self.vad_frame_size))]

            if np.mean(vad) >= min_asl_percent:
                x, fs = sf.read(file)
                x = x[0:trim_len_samples]
                assert fs == self.fs, "Must be 16kHz"
                break

            count = count + 1
            if count > 10000:
                assert 0, "ERROR finding active speech regions"


        return x, file, vad

    def __generate_num_samples(self, num_samples):

        batch_x = np.zeros((num_samples,) + self.feature_extractor.get_output_shape())
        batch_y = np.zeros(num_samples)
        if num_samples > self.num_rir:
            rir_inds = np.random.choice(self.num_rir, replace=True, size=num_samples)
        else:
            rir_inds = np.random.choice(self.num_rir, replace=False, size=num_samples)

        snr_dbs = np.random.uniform(self.snr_limits[0], self.snr_limits[1], size=num_samples)

        if num_samples > self.num_noise:
            noise_ids = np.random.choice(self.num_noise, replace=True, size=num_samples)
        else:
            noise_ids = np.random.choice(self.num_noise, replace=False, size=num_samples)

        gain_dbs = np.random.uniform(self.gain_db_limits[0], self.gain_db_limits[1], size=num_samples)
        for _i in range(num_samples):

            snr_db = snr_dbs[_i]
            gain_db = gain_dbs[_i]
            desired_speech_asl_db = self.dataset_speech_level + gain_db
            noise_file = self.noise_files[noise_ids[_i]]
            rir_file = self.rir_files[rir_inds[_i]]

            # RIR
            rir, fs_in = sf.read(rir_file)
            assert fs_in == self.fs, "Must be 16kHz"
            if len(rir) > int(self.max_rir_length_sec*self.fs):
                rir = rir[0:int(self.max_rir_length_sec*self.fs)]

            # Speech
            trim_len = int(self.instance_len_sec * self.fs)
            speech, speech_file, vad = self.__generate_speech_with_minimum_threshold(trim_len, self.min_asl_percent)

            # RIR Convolve
            rir_flag = np.random.binomial(1, self.rir_probability)
            if rir_flag:
                wet = sp.signal.fftconvolve(speech, rir)

                # Time-synchronize wet mix with dry (assume direct path is max(abs(rir))
                max_ind = np.argmax(np.abs(rir))
                wet = wet[max_ind:]
                wet = wet[0:trim_len]
            else:
                wet = speech

            # Noise
            noise = self.__get_noise(noise_file, trim_len)

            # Mix the reverberated speech and noise
            output, scaled_wet, scaled_noise, speech_scale, noise_scale = \
                mix_snr.mix_speech_and_noise(wet, noise, snr_db, self.fs,
                                             speech_weighting='rmsa',
                                             noise_weighting='fixed',
                                             fixed_noise_level_db=self.fixed_noise_level_db,
                                             rescale_speech=True,
                                             desired_speech_asl_db=desired_speech_asl_db)

            speech_flag = np.random.binomial(1, self.speech_probability)
            if speech_flag and self.feature_mode == self.FeatureMode.kVAD and np.random.choice(2):
                output = scaled_noise
                vad = vad*0


            #batch_audio[_i, :, :, 0] = np.expand_dims(output, axis=1)
            batch_x[_i] = np.expand_dims(self.feature_extractor.extract(np.expand_dims(output, axis=1)), axis=0)


            # Get value to predict
            if self.feature_mode == self.FeatureMode.kSNR:
                batch_y[_i] = snr_db
            elif self.feature_mode == self.FeatureMode.kSpeechFeature:
                batch_y[_i] = self.feature_map[os.path.basename(speech_file)[0:-4]]
            elif self.feature_mode == self.FeatureMode.kNoiseFeature:
                batch_y[_i] = self.feature_map[os.path.basename(noise_file)[0:-4]]
            elif self.feature_mode == self.FeatureMode.kRIRFeature:
                batch_y[_i] = self.feature_map[os.path.basename(rir_file)[0:-4]]
            elif self.feature_mode == self.FeatureMode.kSTI:
                sti = rir_stats.speech_transmission_index2(rir, scaled_wet, scaled_noise, self.fs)
                batch_y[_i] = sti
            elif self.feature_mode == self.FeatureMode.kVAD:
                batch_y[_i] = vad[-1]
            else:
                assert 0, "Unknown feature type"

        # Feature extract
        # batch_x = self.__feature_extract(batch_audio)

        return batch_x, batch_y

    def __getitem__(self, index):
        """Generates one batch of data"""  # X : (n_samples, *dim, n_channels)

        # #rir_inds = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        # if self.batch_size <= self.num_rir:
        #     rir_inds = np.random.choice(self.num_rir, replace=False, size=self.batch_size)
        # else:
        #     rir_inds = np.random.choice(self.num_rir, replace=True, size=self.batch_size)

        x, y = self.__generate_num_samples(self.batch_size)

        return x, y