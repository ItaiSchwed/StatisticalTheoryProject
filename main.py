import numpy as np

from pareto import Pareto
from alpha_mle import AlphaMLE
from expected_values import ExpectedValues
from confidence_interval import ConfidenceInterval

print('4.1')
pareto = Pareto(7., 5.)
sample = pareto.sample(1000)
pareto.histogram(sample)

print('4.2')
mle = AlphaMLE()
mle.plot_random_alphas_log_likelihoods(sample, 3, 11)

print('4.3')
print(f'the mle of the sample is {mle(sample)}')

print('4.4')
pareto = Pareto(7., 5.)
mle = AlphaMLE()
confidence_interval = ConfidenceInterval()
print(f'the confidence interval of 0.95 is {confidence_interval(0.95)}')
print(f'the confidence interval of 0.99 is {confidence_interval(0.99)}')
confidence_interval.plot_confidence_intervals()

print('4.5')
ev = ExpectedValues()
print(f'expected value for alpha=0.9 is {ev(0.9)}')
print(f'expected value for alpha=1.1 is {ev(1.1)}')
