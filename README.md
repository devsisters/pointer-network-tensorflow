# Pointer Networks in Tensorflow

TensorFlow implementation of [Pointer Networks](https://arxiv.org/abs/1506.03134).

![model](./assets/model.png)

(in progress)


## Requirements

- Python 2.7
- [tqdm](httsp://github.com/tqdm/tqdm)
- [TensorFlow 0.12.1](httsp://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

To train a model:

    $ python main.py --task=tsp --max_data_length=20 # download dataset from the paper
    $ python main.py --task=tsp --max_data_length=10 # generate dataset itself

To train a model:

    $ python main.py
    $ tensorboard --logdir=logs --host=0.0.0.0

To test a model:

    $ python main.py --is_train=False

## Results

Train/Test loss of `max_data_length=10` after 40,000 steps:

    $ python main.py --reg_scale=1.0 --optimizer=sgd

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
