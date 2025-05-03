import itertools
import math
import matplotlib.pyplot as plt
from matplotlib import patches
import nashpy as nash
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
	for j in range(consumers):
		consumerposition = np.zeros([dimensions])
		for k in range(dimensions):
			consumerposition[k] = float(cdata.iloc[j,k+1])
		utilities = np.zeros([producers])
		maxutility = -sys.float_info.max
		for k in range(producers):
			distancetoproducer = distance(tuples[i][k]-consumerposition)
			utilities[k] = -distancetoproducer
			if (utilities[k] > maxutility):
				maxutility = utilities[k]
		for k in range(producers):
			if (utilities[k] == maxutility):
				payoffsets[i][k] = payoffsets[i][k] + 1
	payoffsum = sum(payoffsets[i])
	for k in range(producers):
		payoffsets[i][k] = payoffsets[i][k] / payoffsum

def calculateresponsecorrespondence(tuples, payoffs, dimensions, locations, producers, bestresponsestrategy, bestresponsepayoff, i, j, k):
	validmove = 1
	for l in range(producers):
		if (i != l):
			if (tuples[j][l] != tuples[k][l]):
				validmove = 0
				break;
	if (validmove == 1):
		bestresponsestrategy.append(tuples[k])
		bestresponsepayoff.append(payoffs[k])
		#print(str(tuples[k])+" "+str(payoffsets[k]))

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


		score = np.zeros([len(tuples)])
		print("Best responses:")
		for i in range(producers):
			bestresponsestrategies = []
			bestresponsepayoffs = []
			for j in range(len(tuples)):
				#print("Producer = "+str(i)+" , Pivot = "+str(tuples[j])+" :")
				responsestrategy = []
				responsepayoff = []
				threads = []
				for k in range(len(tuples)):
					t = Thread(target=calculateresponsecorrespondence, args=(tuples, payoffs, dimensions, locations, producers, responsestrategy, responsepayoff, i,j,k,))
					t.start()
					threads.append(t)
				for k in range(len(tuples)):
					threads[k].join()
				bestresponsepayoffforproducer = -sys.float_info.max
				bestresponseindices = []
				for k in range(len(responsepayoff)):
					if (responsepayoff[k][i] == bestresponsepayoffforproducer):
						bestresponseindices.append(k)
					elif (responsepayoff[k][i] > bestresponsepayoffforproducer):
						bestresponseindices = []
						bestresponseindices.append(k)
						bestresponsepayoffforproducer = responsepayoff[k][i]
				for k in range(len(bestresponseindices)):
					bestresponsestrategies.append(responsestrategy[bestresponseindices[k]])
					#print("Best response strategy = "+str(responsestrategy[bestresponseindices[k]]))
					bestresponsepayoffs.append(responsepayoff[bestresponseindices[k]])
					#print("Best response payoff = "+str(responsepayoff[bestresponseindices[k]]))
			uniquebestresponsestrategies = []
			uniquebestresponsepayoffs = []
			for j in range(len(bestresponsestrategies)):
				unique = 1
				for k in range(len(uniquebestresponsestrategies)):
					if (bestresponsestrategies[j] == uniquebestresponsestrategies[k]):
						unique = 0
						break;
				if (unique == 1):
					uniquebestresponsestrategies.append(bestresponsestrategies[j])
					for l in range(len(tuples)):
						if (bestresponsestrategies[j] == tuples[l]):
							score[l] = score[l] + 1
							break;
					print(str(i)+" : "+str(bestresponsestrategies[j])+" -> "+str(bestresponsepayoffs[j]))
		print("")

		print("Equilibrium points:")
		for i in range(len(tuples)):
			if(score[i] == producers):
				print(tuples[i])
		print("")