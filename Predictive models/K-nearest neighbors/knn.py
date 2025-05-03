import math
import numpy as np
import pandas as pd
import random
import sys
from threading import Thread

def distance(a,b):
	d = 0
	for i in range(N):
		d = d + (a[i]-b[i])**2
	d = math.sqrt(d)
	return d

if (len(sys.argv) > 3):

	trainingfile = sys.argv[1]
	trainingdata = pd.read_excel(trainingfile)
	trainingdata.set_index("Id", inplace=True)
	trainingclass = trainingdata["Class"].to_numpy()
	trainingposition = trainingdata.drop("Class",axis=1).to_numpy()
	M1 = trainingposition.shape[0]
	N1 = trainingposition.shape[1]

	predictionfile = sys.argv[2]
	predictiondata = pd.read_excel(predictionfile)
	predictiondata.set_index("Id", inplace=True)
	predictionclass = predictiondata["Class"].to_numpy()
	predictionposition = predictiondata.drop("Class",axis=1).to_numpy()
	M2 = predictionposition.shape[0]
	N2 = predictionposition.shape[1]

	K = int(sys.argv[3])


	if (N1 == N2):

		N = N1

		print(predictiondata)

		for i in range(M2):
			distances = np.zeros([M1])
			for j in range(M1):
				distances[j] = distance(predictionposition[i], trainingposition[j])
			indices1 = distances.argsort()
			classes = np.zeros([K])
			for j in range(K):
				classes[j] = trainingclass[indices1[j]]
			uniqueclasses = []
			for j in range(K):
				unique = 1
				for k in range(len(uniqueclasses)):
					if (classes[j] == uniqueclasses[k]):
						unique = 0
						break
				if (unique == 1):
					uniqueclasses.append(classes[j])
			C = len(uniqueclasses)
			tallies = np.zeros([C])
			for j in range(C):
				for k in range(K):
					if (classes[k] == uniqueclasses[j]):
						tallies[j] = tallies[j] + 1
			indices2 = tallies.argsort()
			predictionclass[i] = uniqueclasses[indices2[0]]

		predictiondata["Class"] = predictionclass

		print(predictiondata)
		predictiondata.to_excel(predictionfile)