import numpy as np
import argparse
from tensorflow import keras
import os
import core.utils as va_utils
import va_data_generators
from datetime import datetime
import h5py
import shutil

from train_subbands_T60_model import build_blind_t60_2018_4sec_subbands


def main():
    parser = argparse.ArgumentParser(prog='train_subbands_T60_model_pre-feature',
                                     description="""Script to train subband T60 prediction model using precomptued features""")
    parser.add_argument("--input", "-i", type=str, default="data/pre_features.h5", help="Path to the precomputed feature and label file")
    parser.add_argument("--output", "-o", type=str, default="trained_models", help="Directory to write results")
    parser.add_argument("--target", "-t", type=int, default=0, help="Target band to predict (1~8), 0=fullband (default), 9=selected bands")
    parser.add_argument("--label", "-l", type=str, default="t60", help="Which label to use for training")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to an existing model, or None")

    args = parser.parse_args()

    dataset_path = args.input
    assert os.path.exists(dataset_path)
    dataset = h5py.File(dataset_path, 'r')

    batch_size = 64
    initial_epoch = 0
    num_epochs = 500
    label = args.label

    # Specify the sub-bands for prediction
    norm_band = None
    if args.target == 0:
        bands = [x for x in range(0, 8)]    # pay attention to the h5 structure
    elif args.target == 9:
        if label == 'modulation':
            bands = [0, 1, 2, 3, 5, 6]   # use the 1000Hz band as reference
            norm_band = 4
        elif label == 't60':
            bands = [1, 2, 3, 4, 5, 6, 7]  # ignore the problematic low freq bands
    else:
        bands = args.target - 1

    # Specify the data generator
    model_function = build_blind_t60_2018_4sec_subbands
    model_filepath = args.model

    data_generator = va_data_generators.PreFeatureGenerator
    train_generator = data_generator(dataset, 'train', label=label, batch_size=32, bands=bands, normalize_band=norm_band)
    val_generator = data_generator(dataset, 'validation', label=label, batch_size=32, bands=bands, normalize_band=norm_band)

    # Create the model
    shape = dataset['train']['audio'][0].shape
    if model_filepath:
        model = keras.models.load_model(model_filepath)
        basename = os.path.splitext(os.path.basename(model_filepath))[0]
        test_name = '{}-{}'.format(label, basename)
    else:
        model = model_function(shape[0], shape[1], len(np.atleast_1d(bands)))
        test_name = '{}-band{}'.format(label, args.target)

    model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Create test folder to store all test products
    test_name_time = test_name + "-{:%m%d%y-%H%M%S}".format(datetime.now())
    products_dir = args.output
    va_utils.make_folder(products_dir)

    test_folder = os.path.join(products_dir, test_name_time)
    va_utils.make_folder(test_folder)
    try:
        shutil.copy(os.path.join(os.path.dirname(dataset_path), 'feature_mean_std.npz'), test_folder)
    except shutil.SameFileError:
        print('Stats file already in this path, skip copying')

    # Create tensorboard and checkpoint callback
    log_dir = os.path.join(test_folder, 'logs')
    va_utils.make_folder(log_dir)

    # Create a tensorboard callback
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size)
    checkpoints_dir = os.path.join(test_folder, 'checkpoints')
    va_utils.make_folder(checkpoints_dir)

    # Early Stopping
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=30)

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
                                  callbacks=[tensorboard, model_checkpoint_callback, es_callback],
                                  # workers=multiprocessing.cpu_count(),
                                  workers=1,
                                  use_multiprocessing=False,
                                  max_queue_size=10)

    dataset.close()


if __name__ == "__main__":
    main()
