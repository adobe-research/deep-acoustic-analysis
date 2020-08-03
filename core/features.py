"""
    A Library of front-end feature extraction classes

    All classes here should be relatively self contained and have no explicit dependencies on the DNN model used.
    These classes are meant to separate out the feature extraction and make it easier to perform inference once a
    model is trained.

"""

import abc
import numpy as np
import librosa

from . import filterbank as va_filterbank




# Base class for feature extractor
class FrontEndFeatureExtraction:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    @abc.abstractmethod
    def get_input_shape(self):
        pass

    @abc.abstractmethod
    def get_output_shape(self):
        pass

    @abc.abstractmethod
    def get_sample_rate(self):
        pass

    @abc.abstractmethod
    def extract(self, x):
        pass

# Using this for WASPAA feature extraction
class T60_Extractor_Mel32_Sec4_Basic(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2

        self.n_fft = 256
        self.hop_length = 128
        self.num_mels = 32
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (int(4.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        return (self.num_mels, int((self.get_input_shape()[0]-self.n_fft)/self.hop_length + 1), 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True

    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.std = npzfile['arr_1']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))

        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)

        assert features.shape == out_shape
        return features

class T60_Extractor_Mel32_Sec2(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False

    def get_input_shape(self):
        return (int(2.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the input shape, num_samples x height x width x depth"""
        return (32, 500, 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True


    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.spectral_median = npzfile['arr_1']
        self.std = npzfile['arr_2']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.spectral_median, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        num_mels = out_shape[0]
        mel_length = out_shape[1]
        fmin = 60
        fmax = 8000
        n_fft = 256
        hop_length = 64

        mel = librosa.feature.melspectrogram(x[:, 0], self.sample_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=num_mels,
                                             fmin=fmin,
                                             fmax=fmax)
        mel = mel[:, -mel_length:]

        mel = librosa.power_to_db(mel)
        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)

        assert features.shape == out_shape
        return features


        return 0


class MelSpectrogramExtractor(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization, mels=32, fft_samples=256, hop_samples=128, input_seconds=4, assert_input_size=True):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2
        self.assert_input_size = assert_input_size

        self.input_samples = int(input_seconds * self.sample_rate)
        self.n_fft = fft_samples
        self.hop_length = hop_samples
        self.num_mels = mels
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (self.input_samples,1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        return (self.num_mels, int((self.get_input_shape()[0]-self.n_fft)/self.hop_length + 1), 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True


    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.std = npzfile['arr_1']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        if self.assert_input_size:
            assert x.shape == required__in_shape

        out_shape = self.get_output_shape()

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))
        mel = np.expand_dims(mel, axis=2)
        if self.normalization_set:
            features = (mel - self.mean) / self.std
        else:
            features = mel
        # features = np.expand_dims(mel, axis=2)

        if self.assert_input_size:
            assert features.shape == out_shape
        return features

class T60_Extractor_Mel32_Sec1(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False

    def get_input_shape(self):
        return (int(1.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the input shape, num_samples x height x width x depth"""
        return (32, 250, 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True


    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.spectral_median = npzfile['arr_1']
        self.std = npzfile['arr_2']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.spectral_median, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        num_mels = out_shape[0]
        mel_length = out_shape[1]
        fmin = 60
        fmax = 8000
        n_fft = 256
        hop_length = 64
        ## TODO: when running live, this function causes a bus error. Unknown reason
        mel = librosa.feature.melspectrogram(x[:, 0], self.sample_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=num_mels,
                                             fmin=fmin,
                                             fmax=fmax)
        mel = mel[:, -mel_length:]

        mel = librosa.power_to_db(mel)
        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)

        assert features.shape == out_shape
        return features

class T60_Extractor_Mel32_Sec4_1D(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2

        self.n_fft = 256
        self.hop_length = 32
        self.num_mels = 32
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (int(4.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        return ( int((self.get_input_shape()[0]-self.n_fft)/self.hop_length + 1), self.num_mels)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        #self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :]
        self.std = X.std(axis=0)[:, :]
        self.normalization_set = True

    def clear_normalization(self):
        self.mean = np.zeros((1, 1))
        self.std = 1+np.zeros((1, 1))
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.std = npzfile['arr_1']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))

        # print(mel.shape, self.mean, self.std)
        if self.normalization_set:
            mel = (mel - self.mean.T) / self.std.T
        # features = np.expand_dims(mel, axis=2)


        features = np.transpose(mel, (1, 0))

        assert features.shape == out_shape
        return features




class T60_Extractor_Mel32_Sec4(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2

        self.n_fft = 256
        self.hop_length = 32
        self.num_mels = 32
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (int(4.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        return (self.num_mels, int((self.get_input_shape()[0]-self.n_fft)/self.hop_length + 1), 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        #self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True

    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.std = npzfile['arr_1']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))


        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)

        assert features.shape == out_shape
        return features

class T60_Extractor_Mel32_Sec4_2ms(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2
        self.n_fft = 256
        self.hop_length = 32
        self.num_mels = self.get_output_shape()[0]
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (int(4.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        return (32, 1993, 1)

        # return (32, 997, 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True


    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.spectral_median = npzfile['arr_1']
        self.std = npzfile['arr_2']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.spectral_median, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        num_mels = out_shape[0]
        mel_length = out_shape[1]

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))

        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)


        assert features.shape == out_shape
        return features


        return 0

class T60_Extractor_Mel32_Sec4_4ms(FrontEndFeatureExtraction):
    """Implements a simplified version of the T60 feature extraction and feature normalization"""
    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.spectral_median = 0
        self.std = 1
        self.normalization_set = False
        self.fmin = 0
        self.fmax = self.sample_rate/2
        self.n_fft = 256
        self.hop_length = 32*2
        self.num_mels = self.get_output_shape()[0]
        self.melfb = librosa.filters.mel(self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def get_input_shape(self):
        return (int(4.0 * self.sample_rate),1)

    def get_output_shape(self):
        """return the output shape, num_samples x height x width x depth"""
        # return (32, 1993, 1)

        return (32, 997, 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization_from_samples(self, X):

        # compute the median over the samples and columns (and expand dims back to match)
        self.spectral_median = np.kron(np.median(X, axis=(0,2)), np.ones(self.get_output_shape()[1],))
        self.mean = X.mean(axis=0)[:, :, 0]
        self.std = X.std(axis=0)[:, :, 0]
        self.normalization_set = True


    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.spectral_median = 0
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.spectral_median = npzfile['arr_1']
        self.std = npzfile['arr_2']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.spectral_median, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        num_mels = out_shape[0]
        mel_length = out_shape[1]

        S = librosa.core.stft(x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, window='hann', center=False)
        mel = librosa.power_to_db(np.dot(self.melfb, np.abs(S) ** 2))

        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)


        assert features.shape == out_shape
        return features


        return 0

class CNN_VAD_Extractor(FrontEndFeatureExtraction):

    def __init__(self, sample_rate, require_normalization):
        super().__init__(sample_rate)
        self.require_normalization = require_normalization
        self.mean = 0
        self.std = 1
        self.normalization_set = False

    def get_input_shape(self):
        return (int(0.4875 * self.sample_rate),1)

    def get_output_shape(self):
        """return the input shape, num_samples x height x width x depth"""
        return (40, 40, 1)

    def get_sample_rate(self):
        return 16000

    def set_normalization(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalization_set = True

    def clear_normalization(self):
        self.mean = 0
        self.std = 1
        self.normalization_set = False

    def read_normalization_from_file(self, filepath):
        npzfile = np.load(filepath)
        self.mean = npzfile['arr_0']
        self.std = npzfile['arr_1']
        self.normalization_set = True

    def write_normalization_to_file(self, filepath):
        np.savez(filepath, self.mean, self.std)

    def extract(self, x):

        if self.require_normalization and not self.normalization_set:
            assert 0, "Must set normalization file"

        required__in_shape = self.get_input_shape()
        assert x.shape == required__in_shape
        out_shape = self.get_output_shape()

        num_mels = out_shape[0]
        fmin = 300
        fmax = 8000
        n_fft = int(.025 * self.sample_rate)
        hop_length = int(.0125 * self.sample_rate)
        mel = librosa.feature.melspectrogram(x[:, 0], self.sample_rate,
                                             n_fft=n_fft,
                                             hop_length=hop_length,
                                             n_mels=num_mels,
                                             fmin=fmin,
                                             fmax=fmax)
        mel = librosa.power_to_db(mel)
        if self.normalization_set:
            mel = (mel - self.mean) / self.std
        features = np.expand_dims(mel, axis=2)

        assert features.shape == out_shape
        return features

class TimeDomainExtractor(FrontEndFeatureExtraction):
    def __init__(self, sample_rate, length_seconds):
        super().__init__(sample_rate)
        self.length_seconds = length_seconds

    def get_input_shape(self):
        return (int(self.length_seconds * self.sample_rate),1)

    def get_output_shape(self):
        """return the input shape, num_samples x height x width x depth"""
        return (int(self.length_seconds * self.sample_rate), 1)

    def get_sample_rate(self):
        return 16000

    def extract(self, x):
        samples = int(self.length_seconds * self.sample_rate)
        return x[0:samples, :]

    def set_normalization_from_samples(self, X):
        pass

    def write_normalization_to_file(self, filepath):
        pass


