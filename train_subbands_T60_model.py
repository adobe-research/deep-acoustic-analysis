import numpy as np
import argparse
from tensorflow import keras
import os
import core.features as va_features
import core.utils as va_utils
import va_data_generators
import librosa
import multiprocessing
from datetime import datetime


class T60_Extractor_Mel32_Sec4_Basic(va_features.FrontEndFeatureExtraction):
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


def build_blind_t60_2018_4sec_subbands(num_mels, mel_length, bandcnt=8, extra_layer=None):
    """Creates a CNN model for blind T60 estimation. """

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation='relu', input_shape=(num_mels, mel_length, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(1, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(.5))
    if extra_layer:
        model.add(keras.layers.Dense(extra_layer))    # added layer for subband fitting
    model.add(keras.layers.Dense(bandcnt))

    return model


def main():
    parser = argparse.ArgumentParser(prog='train_subbands_T60_model',
                                     description="""Script to train subband T60 prediction model""")
    parser.add_argument("--input", "-i", required=True, help="Directory where data and labels are", type=str)
    parser.add_argument("--output", "-o", required=True, help="Directory to write results", type=str)
    parser.add_argument("--statfile", "-s", default="stats.csv", type=str, help="Name of the stats files")
    parser.add_argument("--target", "-t", type=int, default=0, help="Target band to predict (1~8), 0=fullband (default)")
    parser.add_argument("--extra_layer", "-e", type=int, default=0, help="Dimension of an extra layer")

    args = parser.parse_args()

    dataset_path = args.input
    if not os.path.exists(dataset_path):
        print('input folder non-exist, abort!')
        exit(1)

    fs = 16000
    batch_size = 64
    initial_epoch = 0
    num_epochs = 500
    test_name = 'T60-B-extra{}-band{}'.format(args.extra_layer, args.target)


    # Specify the front-end feature extractor
    feature_extractor = T60_Extractor_Mel32_Sec4_Basic(fs, False)
    model_function = build_blind_t60_2018_4sec_subbands
    predict_name = 'subbands'

    feature_normalization = True
    num_normalization_samples = 10000
    stats_filepath = ''
    model_filepath = ''

    data_generator = va_data_generators.PreMixedAcousticSceneWASPAA

    # Create test folder to store all test products
    test_name_time = test_name + "-{:%m%d%y-%H%M%S}".format(datetime.now())
    products_dir = args.output
    va_utils.make_folder(products_dir)

    test_folder = os.path.join(products_dir, test_name_time)
    va_utils.make_folder(test_folder)

    # Zip the contents of the current code directory to make it easier to recreate the experiment
    va_utils.zipdir(os.getcwd(), os.path.join(test_folder, 'code.zip'))

    # Create the model
    shape = feature_extractor.get_output_shape()

    if model_filepath == '':
        use_fullband = (args.target == 0)
        model = model_function(shape[0], shape[1], use_fullband, args.extra_layer)
    else:
        model = keras.models.load_model(model_filepath)

    model.summary()

    if args.target == 0:
        bands = [x for x in range(1, 9)]
    else:
        bands = args.target

    train_generator = data_generator(os.path.join(dataset_path, 'train'), feature_extractor, label=predict_name,
                                     batch_size=batch_size, bands=bands, stats_name=args.statfile)
    val_generator = data_generator(os.path.join(dataset_path, 'validation'), feature_extractor, label=predict_name,
                                   batch_size=batch_size, bands=bands, stats_name=args.statfile)
    test_generator = data_generator(os.path.join(dataset_path, 'test'), feature_extractor, label=predict_name,
                                    batch_size=batch_size, bands=bands, stats_name=args.statfile)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    if feature_normalization:

        if os.path.exists(stats_filepath):
            print('Reading mean and std from file...')
            feature_extractor.read_normalization_from_file(stats_filepath)

        else:
            print('Creating mean and std...')
            stats_filepath = os.path.join(test_folder, 'feature_mean_std.npz')

            # Generate a bunch of samples and fit the mean and std
            XX, yy = train_generator.generate_samples(num_normalization_samples)

            # Turn on the normalization from the feature extractor
            feature_extractor.set_normalization_from_samples(XX)
            feature_extractor.write_normalization_to_file(stats_filepath)

    # Create tensorboard and checkpoint callback
    log_dir = os.path.join(test_folder, 'logs')
    va_utils.make_folder(log_dir)

    # Create a tensorboard callback
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size)
    checkpoints_dir = os.path.join(test_folder, 'checkpoints')
    va_utils.make_folder(checkpoints_dir)

    # Early Stopping
    # es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Create the model checkpoint callback
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(os.path.join(checkpoints_dir,
                                                                             'model.{epoch:04d}-{val_loss:.3f}.hdf5'),
                                                                monitor='val_loss',
                                                                verbose=0,
                                                                save_best_only=True,
                                                                save_weights_only=False,
                                                                mode='auto',
                                                                period=1)

    # Fit model
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.__len__(),
                                  epochs=num_epochs,
                                  initial_epoch=initial_epoch,
                                  validation_data=val_generator,
                                  validation_steps=val_generator.__len__(),
                                  callbacks=[tensorboard, model_checkpoint_callback],
                                  workers=multiprocessing.cpu_count(),
                                  use_multiprocessing=True,
                                  max_queue_size=10)


if __name__ == "__main__":
    main()
