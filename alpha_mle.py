import math

import matplotlib.pyplot as plt
import numpy as np


class AlphaMLE:
    def __init__(self,
                 x_m=5.,
                 h=0.001,
                 alpha_0=5,
                 convergence_boundary=0.01):
        self.x_m = x_m
        self.h = h
        self.alpha_0 = alpha_0
        self.convergence_boundary = convergence_boundary
        self.alphas = [[self.alpha_0]]
        self.negative_alphas_count = 0

    def __call__(self, sample, plot=True):
        self.sample = sample

        alpha = self.numeric_max(self.alpha_0)

        if plot:
            plt.title(r'$Log\quad Likelihood\quad Of\quad \alpha\quad Numeric\quad Maximum\quad Process$')
            plt.ylabel(r'$\alpha$')
            for index in range(len(self.alphas)):
                start = int(np.sum([len(line) - 1 for line in self.alphas[:index]]))
                indices = range(start, start + len(self.alphas[index]))
                plt.scatter(indices, self.alphas[index], color='blue')
                plt.plot(indices, self.alphas[index], color='red')
            plt.xticks(range(np.sum([len(line) - 1 for line in self.alphas]) + 1))
            plt.yticks(range(15))
            plt.show()

        return alpha

    def numeric_max(self, alpha_n):
        alpha_n_1 = alpha_n - (self.first_derivative(alpha_n) / self.second_derivative(alpha_n))
        self.alphas[self.negative_alphas_count].append(alpha_n_1)
        if alpha_n_1 < 0:
            self.negative_alphas_count += 1
            self.alphas.append([abs(alpha_n_1)])
        if math.fabs(alpha_n_1 - alpha_n) < self.convergence_boundary:
            return alpha_n_1
        # alpha should be positive therefore we are applying abs
        return self.numeric_max(abs(alpha_n_1))

    def likelihood(self, alpha, x):
        return np.where(x >= self.x_m, (alpha * (self.x_m ** alpha)) / (x ** (alpha + 1)), 0)

    def log_likelihood(self, alpha, sample):
        return np.add.reduce(np.log(self.likelihood(alpha, sample)))

    def first_derivative(self, alpha):
        return ((self.log_likelihood(alpha + self.h, self.sample) - self.log_likelihood(alpha - self.h, self.sample)) /
                (2 * self.h))

    def second_derivative(self, alpha):
        return ((self.log_likelihood(alpha + self.h, self.sample) -
                 2 * self.log_likelihood(alpha, self.sample) + self.log_likelihood(alpha - self.h, self.sample)) /
                (self.h ** 2))

    # for verification
    # #######################################################################################
    # def analytic_first_derivative(self, alpha):
    #     n = len(self.sample)
    #     return (n / alpha) - (np.add.reduce(np.log(self.sample))) + (n * np.log(self.x_m))
    #
    # def analytic_second_derivative(self, alpha):
    #     return -len(self.sample) / (alpha ** 2)
    # #######################################################################################

    def plot_random_alphas_log_likelihoods(self, sample, min, max):
        alphas = np.random.uniform(min, max, 100)
        ax = plt.gca()
        plt.title(r'$Random\quad \alpha -s\quad And\quad Their\quad Log-Likelihoods$')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$log-likelihood\quad of\quad \alpha$')

        alphas = np.sort(alphas)
        log_likelihoods = [self.log_likelihood(alpha, sample) for alpha in alphas]
        ax.scatter(alphas, log_likelihoods, color='blue')
        max_point = ax.scatter(alphas[np.argmax(log_likelihoods)], np.max(log_likelihoods), color='red')

        max_point.set_label(r'$the\quad \alpha\quad with\quad the\quad maximum\quad log-likelihood$')
        ax.legend()
        plt.show()
