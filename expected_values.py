import pickle

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from pareto import Pareto
import seaborn as sns


class ExpectedValues:

    def __call__(self, alpha, x_m=5.):
        self.sizes = [1, 100, 10_000, 1_000_000, 100_000_000]
        points_fig, points_ax = plt.subplots()
        _, boxes_ax = plt.subplots()
        points_ax.set_xscale('log')
        means = self.get_means(alpha, x_m)
        points_ax.scatter(means[:, 0], means[:, 1], color='blue')
        expected_values_means = [np.mean(means[i:(i + 1) * 100, 1]) for i in range(len(self.sizes))]
        points_ax.plot(self.sizes, expected_values_means, color='red')
        sns.boxplot(x=means[:, 0], y=means[:, 1], ax=boxes_ax)
        plt.show()
        return expected_values_means[-1]

    def get_means(self, alpha, x_m):
        try:
            with open(f'pickles/{alpha}_means.pickle', 'rb') as file:
                means = pickle.load(file)
        except Exception:
            pareto = Pareto(alpha, x_m)
            means = np.array([(size, np.mean(pareto.sample(size))) for _ in tqdm(range(100)) for size in self.sizes])
            with open(f'pickles/{alpha}_means.pickle', 'wb') as file:
                pickle.dump(means, file)
        return means
