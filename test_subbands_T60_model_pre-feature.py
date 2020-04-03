import argparse
from tensorflow import keras
import os
import csv
import va_data_generators
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(prog='test_subbands_T60_model_pre-feature',
                                     description="""Script to test subband T60 prediction model""")
    parser.add_argument("--input", "-i", type=str, default="data/pre_features.h5", help="Path to the precomputed feature and label file")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--target", "-t", type=int, default=0, help="Target band to predict (1~8), 0=fullband (default), 9=selected bands")
    parser.add_argument("--label", "-l", type=str, required=True, choices=['modulation', 't60'], help="Which label to use for training")

    args = parser.parse_args()

    dataset_path = args.input
    assert os.path.exists(dataset_path)
    dataset = h5py.File(dataset_path, 'r')

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
    partition = 'test'
    model_filepath = args.model

    data_generator = va_data_generators.PreFeatureGenerator
    test_generator = data_generator(dataset, partition=partition, label=label, batch_size=512, bands=bands, normalize_band=norm_band)

    # Load the trained model
    model = keras.models.load_model(model_filepath)
    model.summary()

    # Test model
    # score = model.evaluate_generator(test_generator)
    # print('{}: {}'.format(model.metrics_names[1], score[1]))

    pred = model.predict_generator(test_generator)
    truth, speakers, inputfile = test_generator.generate_samples(pred.shape[0])[1:4]
    mae = np.mean(abs(pred - truth), axis=1)
    print('{}: {}'.format(model.metrics_names[1], mae.mean()))

    # Exporting csv
    output_csv = os.path.join(os.path.dirname(model_filepath), 'test_result_{}_{}.csv'.format(partition, label))
    with open(output_csv, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Input file'] + ["{}_truth_{}".format(label, f) for f in bands] + ["{}_prediction_{}".format(label, f) for f in bands] + ['mae'])
        # for (mixture_file, t60_est, drr_est) in data:
        #     writer.writerow([mixture_file, t60_est, drr_est])
        for i in range(len(pred)):
            writer.writerow([speakers[i].strip()] + [s for s in truth[i]] + [t for t in pred[i]] + [mae[i]])

    dataset.close()


if __name__ == "__main__":
    main()
