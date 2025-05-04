import itertools
import math
import numpy as np
import pandas as pd
import scipy
import sys
from threading import Thread

def distance(vector):
	output = 0
	for i in range(len(vector)):
		output = output + vector[i]**2
	return math.sqrt(output)

def calculatepayoff(dimensions, producers, consumers, cdata, tuples, payoffsets, i):
	distances = np.zeros([consumers,producers])
	for j in range(consumers):
		consumerposition = np.zeros([dimensions])
		for k in range(dimensions):
			consumerposition[k] = float(cdata.iloc[j,k+1])
		utilities = np.zeros([producers])
		maxutility = -sys.float_info.max
		for k in range(producers):
			distances[j][k] = distance(tuples[i][k]-consumerposition)
			utilities[k] = -distances[j][k]
			if (utilities[k] > maxutility):
				maxutility = utilities[k]
		for k in range(producers):
			if (utilities[k] == maxutility):
				payoffsets[i][k] = payoffsets[i][k] + 1
	aggregatedemand = np.sum(payoffsets[i])
	for j in range(producers):
		payoffsets[i][j] = payoffsets[i][j] / aggregatedemand
	for j in range(producers):
		avgdistance = np.mean(distances[:,j])
		maxdistance = np.max(distances[:,j])
		if (maxdistance > 0):
			payoffsets[i][j] = payoffsets[i][j] * (1 - avgdistance / maxdistance)


def calculatemixedpayoffs(producers, tuples, strategies, payoffs, _weights):
	_tupleweights = np.ones([len(tuples)])
	_mixedpayoffs = np.zeros(shape=[len(strategies),producers])
	for i in range(producers):
		for j in range(len(tuples)):
			for k in range(len(strategies)):
				if (strategies[k] == tuples[j][i]):
					_tupleweights[j] = _weights[k][i] * _tupleweights[j]
	for i in range(producers):
		for k in range(len(strategies)):
			for j in range(len(tuples)):
				if (strategies[k] == tuples[j][i]):
					_mixedpayoffs[k][i] = _mixedpayoffs[k][i] + _tupleweights[j]*payoffs[j,i]
	return _tupleweights, _mixedpayoffs

def score(parameters, _weights, producers, producer, tuples, strategies, payoffs):
	for i in range(len(strategies)):
		_weights[i,producer] = parameters[i]
	_tupleweights, _mixedpayoffs = calculatemixedpayoffs(producers, tuples, strategies, payoffs, _weights)
	'''payoffsum = 0
	for j in range(len(strategies)):
		payoffsum = payoffsum + _mixedpayoffs[j,producer]
	return -payoffsum'''
	sqrdifferences = []
	for i in range(producers):
		sqrdifference = 0
		if (i != producer):
			for j in range(len(strategies)):
				sqrdifference = sqrdifference + (_mixedpayoffs[j,i] - np.mean(_mixedpayoffs[:,i]))**2
		sqrdifferences.append(sqrdifference)
	return sum(sqrdifferences)

if (len(sys.argv) > 4):

	dimensions = int(sys.argv[1])
	locations = int(sys.argv[2])
	producers = int(sys.argv[3])

	if (dimensions > 0 and locations > 0 and producers > 0):

		strategies = []
		position = np.zeros([dimensions])
		position[dimensions-1] = -1
		loop = 1
		index = 0
		while(loop == 1):
			carry = 1
			index = index+1
			for i in range(dimensions-1,-1,-1):
				position[i] = position[i] + carry
				carry = 0
				if (position[i] == locations):
					carry = 1
					position[i] = 0
			strategies.append(position.tolist())
			ready = 1
			for i in range(dimensions):
				if (position[i] < locations-1):
					ready = 0
					break
			if (ready == 1):
				loop = 0
		print("")
		print("Strategies:")
		print(strategies);
		print("")

		tuples = [p for p in itertools.product(strategies, repeat=producers)]
		print("Tuples:")
		print(tuples)
		print("")

		cfile = sys.argv[4]
		cdata = pd.read_excel(cfile)
		consumers = cdata.shape[0]
		payoffs = np.zeros(shape=[len(tuples),producers])
		threads = []
		for i in range(len(tuples)):
			t = Thread(target=calculatepayoff, args=(dimensions, producers, consumers, cdata, tuples, payoffs, i,))
			t.start()
			threads.append(t)
		for i in range(len(tuples)):
			threads[i].join()
		print("Payoffs:")
		print(payoffs.tolist())
		print("")

		print("Initial mix:")
		weights = np.zeros(shape=[len(strategies),producers])
		for i in range(producers):
			#weightsum = 0
			for j in range(len(strategies)):
				weights[j,i] = 1 / len(strategies)
				#weights[j,i] = random.random()
				#weightsum = weightsum + weights[j,i]
			#for j in range(len(strategies)):
			#	weights[j,i] = weights[j,i] / weightsum
		print(weights)
		print("")

		print("Initial payoffs:")
		initialtupleweights, initialmixedpayoffs = calculatemixedpayoffs(producers, tuples, strategies, payoffs, weights)
		print(np.sum(initialmixedpayoffs,0))
		print("")

		print("Optimization:")
		constraint = scipy.optimize.LinearConstraint(np.ones([len(strategies)]), 1,1)
		boundary = []
		tmpweights = weights.copy()
		for i in range(len(strategies)):
			boundary.append((0, 1))
		for i in range(producers):
			minimization = scipy.optimize.minimize(score, tmpweights[:,i].copy(), args=(tmpweights.copy(), producers, i, tuples, strategies, payoffs), constraints=[constraint], bounds=boundary)
			print("Producer "+str(i)+" -> "+str(minimization.x)+" (Score="+str(minimization.fun)+")")
			weights[:,i] = minimization.x
		print("")



		print("Equilibrium mix:")
		print(weights)
		print("")

