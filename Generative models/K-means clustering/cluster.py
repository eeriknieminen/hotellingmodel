import numpy as np
import pandas as pd
import random
import sys

iterations = 10000
stoppingthreshold = 0

def distance(a,b):
	d = 0
	for i in range(len(a)):
		d = d + (a[i]-b[i])**2
	d = np.sqrt(d)
	return d

if (len(sys.argv) > 2):

	file = sys.argv[1]
	dataset = pd.read_excel(file)
	dataset.set_index("Id", inplace=True)
	data = dataset.to_numpy()
	N = dataset.shape[0]
	D = dataset.shape[1]

	minvalues = []
	maxvalues = []
	for i in range(D):
		minvalue = sys.float_info.max
		maxvalue = -sys.float_info.max
		for j in range(N):
			minvalue = min(minvalue,float(data[j,i]))
			maxvalue = max(maxvalue,float(data[j,i]))
		minvalues.append(minvalue)
		maxvalues.append(maxvalue)

	K = int(sys.argv[2])

	for i in range(1,K+1,1):

		print("")
		if (i == 1):
			print(str(i)+" cluster:")
		else:
			print(str(i)+" clusters:")

		means = np.zeros([i,D])
		for j in range(i):
			for k in range(D):
				rand = random.random()
				means[j,k] = rand * minvalues[k] + (1-rand) * maxvalues[k]

		for j in range(iterations):

			mask = np.zeros([N])
			counts = np.zeros([K])
			for k in range(N):
				mindistance = sys.float_info.max
				minindex = -1
				for l in range(i):
					d = distance(data[k],means[l])
					if (d < mindistance):
						mindistance = d
						minindex = l
				mask[k] = minindex
				counts[minindex] = counts[minindex] + 1

			newmeans = np.zeros([i,D])
			for k in range(i):
				mean = np.zeros([D])
				for l in range(N):
					if (mask[l] == k):
						mean = mean + data[l]/counts[k]
				newmeans[k] = mean

			change = np.sum(np.abs(newmeans-means))
			means = newmeans

			if (change <= stoppingthreshold):
				break

		print(means)