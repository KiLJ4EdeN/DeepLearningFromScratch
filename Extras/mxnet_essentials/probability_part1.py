# -*- coding: utf-8 -*-
# !pip install mxnet gluoncv

import mxnet as mx
from mxnet import nd
from matplotlib import pyplot as plt

# simple die probabilities
# we will try to draw samples to assign probs to each side of the dice.
# we assume that each has and equal 1/6.
probabilities = nd.ones(6) / 6
print(probabilities)
# draw samples from the distribution.
nd.sample_multinomial(probabilities)

# multiple draws at one time
print(nd.sample_multinomial(probabilities, shape=(10)))
print(nd.sample_multinomial(probabilities, shape=(5,10)))

# create a thousand samples.
rolls = nd.sample_multinomial(probabilities, shape=(1000))

# show how many time each side was repeated over the whole course of sampling.
counts = nd.zeros((6,1000))
# total count for each side.
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals

# total probability for each.
# should get more close to 1/6 as we throw in more samples.
print(totals / 1000)

print(counts)

# normalize the counts by the step number.
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
# show how probabilites evolved when given more samples.
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
print(estimates[:, -1])


# probs were mostly high at first but converged over time.
plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
# desired line
plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()

# X for die is a subset of this.
# $X \in \{1, 2, 3, 4, 5, 6\}$.

# likelihood
# someone's height falls into a given interval, say between 1.99 and 2.01 meters. 
# In these cases we quantify the likelihood that we see a value as a density
# The height of exactly 2.0 meters has no probability, but nonzero density. Between any two different heights we have nonzero probability.

# probability axioms

"""For any event $z$, the probability is never negative, i.e. $\Pr(Z=z) \geq 0$.
For any two events $Z=z$ and $X=x$ the union is no more likely than the sum of the individual events, i.e. $\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)$.
For any random variable, the probabilities of all the values it can take must sum to 1 $\sum_{i=1}^n P(Z=z_i) = 1$.
For any two mutually exclusive events $Z=z$ and $X=x$, the probability that either happens is equal to the sum of their individual probabilities that $\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=z)$."""
