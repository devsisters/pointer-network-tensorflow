# Pointer Networks in Tensorflow

TensorFlow implementation of [Pointer Networks](https://arxiv.org/abs/1506.03134).

![model](./assets/model.png)

*Support multithreaded data pipelines to reduce I/O latency.*


## Requirements

- Python 2.7
- [tqdm](httsp://github.com/tqdm/tqdm)
- [TensorFlow 0.12.1](httsp://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

To train a model:

    $ python main.py --task=tsp --max_data_length=20 --hidden_dim=512 # download dataset used in the paper
    $ python main.py --task=tsp --max_data_length=10 --hidden_dim=128 # generate dataset itself

To train a model:

    $ python main.py
    $ tensorboard --logdir=logs --host=0.0.0.0

To test a model:

    $ python main.py --is_train=False

## Results

Train/Test loss of `max_data_length=10` after 24,000 steps:

    $ python main.py --task=tsp --max_data_length=10 --hidden_dim=128
    $ ...
    $ 2017-01-18 17:17:48,054:INFO::
    $ 2017-01-18 17:17:48,054:INFO::test loss: 0.00149279157631
    $ 2017-01-18 17:17:48,055:INFO::test x: [1 6 2 5 4 3 0 0 0 0 0]
    $ 2017-01-18 17:17:48,057:INFO::test y: [1 6 2 5 4 3 0 0 0 0 0] (True)
    $ 2017-01-18 17:17:48,059:INFO::test x: [1 3 8 4 7 5 2 6 0 0 0]
    $ 2017-01-18 17:17:48,059:INFO::test y: [1 3 8 4 7 5 2 6 0 0 0] (True)
    $ 2017-01-18 17:17:48,058:INFO::test x: [ 1  8  3  7  9  5  6  4  2 10  0]
    $ 2017-01-18 17:17:48,058:INFO::test y: [ 1  8  3  7  9  5  6  4  2 10  0] (True)

![model](./assets/max_data_length=10_step=24000.png)

Train/Test loss of `max_data_length=20` after 40,000 steps:

    $ python main.py --task=tsp --max_data_length=20 --hidden_dim=512

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
