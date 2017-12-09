import pandas as pd
import numpy as np
# Somente testes!!

from minisom import MiniSom

from utils import progress


def clean_dataset(filename):
    dataset = pd.read_csv(filename, usecols=range(1, 7), parse_dates=[0])
    dataset['date'] = dataset['date'].apply(lambda x: float(x.value))
    return np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, dataset)


def get_labels(filename):
    return np.genfromtxt(filename, delimiter=',', usecols=(7,), dtype=int, skip_header=True)


def calc_precision(dataset_filename):
    label_map = np.zeros(shape=(DIM_X, DIM_Y))
    label_map *= -1

    for i, d in enumerate(data):
        x, y = som.winner(d)
        l = labels[i]
        label_map[x][y] = l

    data_test = clean_dataset(dataset_filename)
    labels_test = get_labels(dataset_filename)

    hit_1 = 0
    hit_0 = 0
    total = len(data_test)

    for i, t in enumerate(zip(data_test, labels_test)):
        d, l = t
        x, y = som.winner(d)

        if l == 1 and l == label_map[x][y]:
            hit_1 += 1
        elif l == 0 and l == label_map[x][y]:
            hit_0 += 1

        progress(i, total, 'Neuron {} -> {}'.format((x, y), l))

    precision_1 = round((hit_1 * 100) / total, 4)
    precision_0 = round((hit_0 * 100) / total, 4)
    error_1 = 100.0 - precision_1
    error_0 = 100.0 - precision_0
    print('')

    print('Taxa de acerto da classe 1 \t-> {}%'.format(precision_1))
    print('Taxa de acerto da classe 0 \t-> {}%'.format(precision_0))
    print('Total: {}%'.format(precision_1+precision_0))

    print('')
    print('Taxa de erro da classe 1 \t-> {}%'.format(error_1))
    print('Taxa de erro da classe 0 \t-> {}%'.format(error_0))


if __name__ == '__main__':
    datasetfull_filename = 'data/dataset_full.txt'
    datatest_filename = 'data/datatest.txt'
    datatest2_filename = 'data/datatest2.txt'
    datatraining_filename = 'data/datatraining.txt'
    DIM_X, DIM_Y = 6, 6

    data = clean_dataset(datatraining_filename)
    labels = get_labels(datatraining_filename)

    som = MiniSom(DIM_X, DIM_Y, 6, sigma=0.2, learning_rate=0.3)
    som.random_weights_init(data)  # inicia com pesos aleatorios

    progress(0, 100, 'Training...')
    # som.train_random(data, 2000)
    som.train_batch(data, num_iteration=2)

    calc_precision(datatest_filename)

    print("\nDone!!!")

