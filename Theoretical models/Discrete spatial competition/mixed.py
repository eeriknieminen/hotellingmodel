import itertools
import math
import numpy as np
import pandas as pd
import random
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
	payoffsum = np.sum(payoffsets[i])
	for j in range(producers):
		payoffsets[i][j] = payoffsets[i][j] / payoffsum
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

def estimatetemixedpayoffgradient(producers, tuples, strategies, payoffs, _weights, producer):
	gradient = np.zeros([len(strategies)])
	step = 0.0001
	for i in range(len(strategies)):
		plus = _weights.copy()
		plus[i,producer] =  plus[i,producer]+step
		minus = _weights.copy()
		minus[i,producer] =  minus[i,producer]-step
		_tupleweightsplus, _mixedpayoffsplus = calculatemixedpayoffs(producers, tuples, strategies, payoffs, plus)
		_tupleweightsminus, _mixedpayoffsminus = calculatemixedpayoffs(producers, tuples, strategies, payoffs, minus)
		gradient[i] = (_mixedpayoffsplus[i,producer] - _mixedpayoffsminus[i,producer]) / (2*step)
	return gradient

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

		loop = 1
		iteration = 0
		iterations = 1000
		learningrate = 0.0001
		convergencethreshold = 1e-8
		while(loop == 1):
			newweights = np.zeros([weights.shape[0], weights.shape[1]])
			for i in range(producers):
				gradient = estimatetemixedpayoffgradient(producers, tuples, strategies, payoffs, weights, i)
				newweights[:,i] = newweights[:,i] + gradient*learningrate
				weightsum = 0
				for j in range(len(strategies)):
					weightsum = weightsum + newweights[j,i]
				newweights[:,i] = newweights[:,i] / weightsum
			ready = 1
			for i in range(producers):
				for j in range(len(strategies)):
					if (np.abs(weights[j,i] - newweights[j,i]) > convergencethreshold):
						ready = 0
						break
				if (ready == 0):
					break
			if (ready == 1):
				break
			else:
				weights = newweights
				iteration = iteration + 1

		print("Equilibrium mix:")
		print(weights)
		print("")

		print("Equilibrium payoffs:")
		equilibriumtupleweights, equilibriummixedpayoffs = calculatemixedpayoffs(producers, tuples, strategies, payoffs, weights)
		print(np.sum(equilibriummixedpayoffs,0))
		print("")