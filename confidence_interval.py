import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

from pareto import Pareto
from alpha_mle import AlphaMLE


class ConfidenceInterval:

    def __init__(self):
        self.pareto = Pareto(7., 5.)
        self.mle = AlphaMLE()

    def __call__(self, significance_level, histogram_sample=True):
        confidence_level = 1 - significance_level
        x = np.array([self.mle(self.pareto.sample(1000), False) for _ in range(10_000)])
        if histogram_sample:
            self.histogram_sample(x)
        mean = np.mean(x)
        n = len(x)
        t = t_dist.ppf(confidence_level / 2, n - 1)
        return ((mean + (t * self._s(n, x, mean) / np.sqrt(n)), mean - (t * self._s(n, x, mean) / np.sqrt(n))),
                (mean - (t * self._s(n, x, mean) / np.sqrt(n))) - (mean + (t * self._s(n, x, mean) / np.sqrt(n))))

    def histogram_sample(self, sample):
        ax = plt.gca()
        plt.title(fr'$\alpha\quad  Histogram$')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\alpha\quad count$')
        count, bins, _ = ax.hist(sample, 100, density=True)
        ax.legend()
        plt.show()

    @staticmethod
    def _s(n, x, mean):
        return np.sqrt((1 / (n - 1)) * np.sum((x - mean) ** 2))

    def plot_confidence_intervals(self):
        cis = self._get_cis()
        for ci in cis:
            plt.plot(ci[:, 0], ci[:, 1], color='blue')
        plt.plot([0, 100], [np.mean(cis[:, 0, 1])] * 2, color='red')
        plt.plot([0, 100], [np.mean(cis[:, 1, 1])] * 2, color='green')
        plt.show()

    def _get_cis(self):
        try:
            with open('pickles/cis.pickle', 'rb') as file:
                cis = pickle.load(file)
        except Exception:
            cis = np.array([[(i, i), self(0.95, False)] for i in tqdm(range(100))]).transpose(0, 2, 1)
            with open('pickles/cis.pickle', 'wb') as file:
                pickle.dump(cis, file)
        return cis
