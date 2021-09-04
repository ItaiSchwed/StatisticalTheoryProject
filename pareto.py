import numpy as np
from matplotlib import pyplot as plt


class Pareto:
    def __init__(self, alpha, x_m):
        self.alpha = alpha
        self.x_m = x_m

    def sample(self, count):
        return (np.random.pareto(self.alpha, count) + 1) * self.x_m

    def histogram(self, sample):
        ax = plt.gca()
        plt.title('Pareto Histogram')
        plt.xlabel('x')
        plt.ylabel(r'$f_{X\sim pareto}(x)$')
        count, bins, _ = ax.hist(sample, 100, density=True)
        fit = (self.alpha * (self.x_m ** self.alpha)) / (bins ** (self.alpha + 1))
        line, = ax.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
        line.set_label('estimated function')
        ax.legend()
        plt.show()
