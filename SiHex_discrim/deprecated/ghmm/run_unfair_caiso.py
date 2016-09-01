from ghmm import *
from UnfairCasino import train_seq, test_seq

# create the model
sigma = IntegerRange(1,7)
A = [[0.9, 0.1], [0.3, 0.7]]
efair = [1.0 / 6] * 6
print efair
eloaded = [3.0 / 13, 2.0 / 13, 2.0 / 13, 2.0 / 13, 2.0 / 13, 1.0 / 13]
B = [efair, eloaded]
pi = [0.5] * 2
m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)
print m

# generating sequences
obs_seq = m.sampleSingle(20)
print obs_seq
obs = map(sigma.external, obs_seq)
print obs

# learning from data
m.baumWelch(train_seq)
print m

# computing a Viterbi-path
v = m.viterbi(test_seq)
print v
my_seq = EmissionSequence(sigma, [1]*20 + [6]*10 + [1]*40)
print my_seq
print m.viterbi(my_seq)

