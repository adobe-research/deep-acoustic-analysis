# Deep Acoustic Repo

This repo contains python code for our TVCG paper ["Scene-Aware Audio Rendering via Deep Acoustic Analysis"](https://arxiv.org/abs/1911.06245). We include the training and testing code for reverberation (T60) and equalization (EQ) analysis.


## Preparation

First step is to download data. `cd` into the `data` folder under this repo and run `./download_data.sh`. The dataset takes 5.5GB disk space. The code uses extracted features instead of raw audio files. To reproduce the dataset, please refer to our paper and retrieve impulse responses from the [ACE Challenge](http://www.ee.ic.ac.uk/naylor/ACEweb/), [MIT IR Survey](http://mcdermottlab.mit.edu/Reverb/IR_Survey.html), and human speech from the [DAPS dataset](https://ccrma.stanford.edu/~gautham/Site/daps.html), as described in our paper. We do not re-distribute these data here.

Then you need to create the python environment. We recommend using `conda`:
```
$ conda env create -f environment.yml -n deepacoustic
$ conda activate deepacoustic
```

## Usage
We provide two trained models in the `models` folder. You can use them for inference already, or train from scratch by not providing the `-m` (model) argument to the training script. You can provide `-h` argument to see help messages.
```
usage: train_subbands_T60_model_pre-feature [-h] [--input INPUT]
                                            [--output OUTPUT]
                                            [--target TARGET] [--label LABEL]
                                            [--model MODEL]

Script to train subband T60 prediction model using precomptued features

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the precomputed feature and label file
  --output OUTPUT, -o OUTPUT
                        Directory to write results
  --target TARGET, -t TARGET
                        Target band to predict (1~8), 0=fullband (default),
                        9=selected bands
  --label LABEL, -l LABEL
                        Which label to use for training
  --model MODEL, -m MODEL
                        Path to an existing model, or None
```
example:
```
python train_subbands_T60_model_pre-feature.py -i data/pre_features.h5 -t 0 -l t60
```
and the test/inference function:
```
usage: test_subbands_T60_model_pre-feature [-h] [--input INPUT] --model MODEL
                                           [--target TARGET] --label
                                           {modulation,t60}

Script to test subband T60 prediction model

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the precomputed feature and label file
  --model MODEL, -m MODEL
                        Path to the trained model
  --target TARGET, -t TARGET
                        Target band to predict (1~8), 0=fullband (default),
                        9=selected bands
  --label {modulation,t60}, -l {modulation,t60}
                        Which label to use for training
```
example:
```
python test_subbands_T60_model_pre-feature.py -i data/pre_features.h5 -m models/t60.hdf5 -l t60 -t 0
```
After running this, a result `csv` file will be written in the same folder as the model.

## Citation
If you use our code or models, please consider citing:
```
@article{tang2020scene,
  title={Scene-Aware Audio Rendering via Deep Acoustic Analysis},
  author={Tang, Zhenyu and Bryan, Nicholas J and Li, Dingzeyu and Langlois, Timothy R and Manocha, Dinesh},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2020},
  publisher={IEEE}
}
```

### Contributing

Contributions are welcomed! Read the [Contributing Guide](./.github/CONTRIBUTING.md) for more information.

### Licensing

This project is licensed under the Apache V2 License. See [LICENSE](LICENSE) for more information.
